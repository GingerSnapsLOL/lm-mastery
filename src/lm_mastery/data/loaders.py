"""
Robust dataset loading utilities with fallback strategies
"""

import os
import glob
import json
from typing import Optional, List
from datasets import load_from_disk, load_dataset, Dataset, Features, Sequence, Value

def load_packed_dataset(dataset_path: str, 
                       dataset_name: Optional[str] = None,
                       split: str = "train") -> Dataset:
    """
    Load dataset with robust fallback strategies for 'List' feature type issues
    
    Args:
        dataset_path: Path to dataset directory
        dataset_name: Optional name for logging
        split: Dataset split to load
    
    Returns:
        Loaded dataset
    """
    if dataset_name is None:
        dataset_name = os.path.basename(dataset_path)
    
    print(f"[loader] Loading dataset: {dataset_name}")
    
    # 1) Try normal path (datasets v3)
    try:
        dataset = load_from_disk(dataset_path)
        print(f"[loader] Successfully loaded {dataset_name} with {len(dataset)} sequences")
        return dataset
    except Exception as e:
        print(f"[loader] load_from_disk failed -> {e}")
        
        # Try to fix the old 'List' feature type issue
        if "Feature type 'List' not found" in str(e):
            print(f"[loader] Attempting to fix 'List' feature type issue for {dataset_name}...")
            try:
                # Look for dataset_info.json and fix the feature type
                info_file = os.path.join(dataset_path, "dataset_info.json")
                if os.path.exists(info_file):
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    
                    print(f"[loader] Original features: {info.get('features', 'Not found')}")
                    
                    # Replace 'List' with 'Sequence' in features
                    if 'features' in info:
                        features_str = json.dumps(info['features'])
                        if '"List"' in features_str:
                            features_str = features_str.replace('"List"', '"Sequence"')
                            info['features'] = json.loads(features_str)
                            
                            # Write back the fixed info
                            with open(info_file, 'w') as f:
                                json.dump(info, f, indent=2)
                            
                            print(f"[loader] Fixed features: {info['features']}")
                            print(f"[loader] Fixed dataset_info.json for {dataset_name}, trying to load again...")
                            dataset = load_from_disk(dataset_path)
                            print(f"[loader] Successfully loaded {dataset_name} after fix with {len(dataset)} sequences")
                            return dataset
                        else:
                            print(f"[loader] No 'List' found in features for {dataset_name}, issue might be elsewhere")
                    else:
                        print(f"[loader] No features found in dataset_info.json for {dataset_name}")
                else:
                    print(f"[loader] dataset_info.json not found for {dataset_name}")
            except Exception as fix_e:
                print(f"[loader] Feature type fix failed for {dataset_name} -> {fix_e}")

    # 2) Try parquet fallback (if your packer kept shards)
    pq_dir = dataset_path + "_parquet"
    pq_files = glob.glob(os.path.join(pq_dir, "*.parquet"))
    if pq_files:
        print(f"[loader] Loading parquet shards fallback for {dataset_name}")
        try:
            # Try to load parquet with explicit features to avoid 'List' type issue
            dataset = load_dataset("parquet", 
                                 data_files=pq_files, 
                                 split=split, 
                                 features=Features({"input_ids": Sequence(Value("int32"))}))
            print(f"[loader] Successfully loaded {dataset_name} from parquet with {len(dataset)} sequences")
            return dataset
        except Exception as e:
            print(f"[loader] Parquet loading failed for {dataset_name} -> {e}")
            print(f"[loader] Trying to read parquet files directly for {dataset_name}...")
            
            # Try to read parquet files directly with pandas
            try:
                import pandas as pd
                all_rows = []
                for f in sorted(pq_files):
                    try:
                        df = pd.read_parquet(f)
                        if 'input_ids' in df.columns:
                            all_rows.extend(df['input_ids'].tolist())
                    except Exception as read_e:
                        print(f"[loader] Failed to read parquet {f} for {dataset_name}: {read_e}")
                        continue
                
                if all_rows:
                    print(f"[loader] Loaded {len(all_rows)} sequences from parquet files for {dataset_name}")
                    feats = Features({"input_ids": Sequence(Value("int32"))})
                    dataset = Dataset.from_dict({"input_ids": all_rows}, features=feats)
                    print(f"[loader] Successfully created dataset for {dataset_name} from parquet with {len(dataset)} sequences")
                    return dataset
            except ImportError:
                print(f"[loader] Pandas not available, skipping parquet direct read for {dataset_name}")
            except Exception as pd_e:
                print(f"[loader] Pandas parquet read failed for {dataset_name} -> {pd_e}")

    # 3) Last resort: read arrow file(s) directly and rebuild features explicitly
    print(f"[loader] Reading raw .arrow files and rebuilding dataset for {dataset_name}")
    arrow_files = glob.glob(os.path.join(dataset_path, "data", "*.arrow")) or glob.glob(os.path.join(dataset_path, "*.arrow"))
    
    if not arrow_files:
        raise FileNotFoundError(f"No .arrow files found under {dataset_path} for {dataset_name}")
    
    all_rows = []
    for f in sorted(arrow_files):
        try:
            import pyarrow.ipc as pa_ipc
            with open(f, "rb") as fh:
                rb = pa_ipc.open_file(fh)
                tbl = rb.read_all()
            col = tbl.column("input_ids")  # list<int32>
            all_rows.extend(col.to_pylist())  # list[list[int]]
        except Exception as e:
            print(f"[loader] Failed to read {f} for {dataset_name}: {e}")
            continue
    
    if not all_rows:
        raise ValueError(f"No data could be loaded from any source for {dataset_name}")
    
    print(f"[loader] Loaded {len(all_rows)} sequences from arrow files for {dataset_name}")
    
    # Use the correct feature type for current datasets version
    feats = Features({"input_ids": Sequence(Value("int32"))})
    dataset = Dataset.from_dict({"input_ids": all_rows}, features=feats)
    print(f"[loader] Successfully created dataset for {dataset_name} from arrow with {len(dataset)} sequences")
    
    return dataset

def load_train_dataset(dataset_name: str = "train_big", 
                      data_dir: Optional[str] = None) -> Dataset:
    """Load training dataset with default settings"""
    if data_dir is None:
        data_dir = "01-pretraining-pipeline/data/processed"
    
    dataset_path = os.path.join(data_dir, f"{dataset_name}.arrow")
    return load_packed_dataset(dataset_path, dataset_name, "train")

def load_val_dataset(dataset_name: str = "val_big", 
                    data_dir: Optional[str] = None) -> Dataset:
    """Load validation dataset with default settings"""
    if data_dir is None:
        data_dir = "01-pretraining-pipeline/data/processed"
    
    dataset_path = os.path.join(data_dir, f"{dataset_name}.arrow")
    return load_packed_dataset(dataset_path, dataset_name, "train")  # Use "train" split for validation data
