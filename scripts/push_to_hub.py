#!/usr/bin/env python3
"""
Push model to Hugging Face Hub
"""

import argparse
import sys
import os
from huggingface_hub import HfApi

def push_to_hub(repo_id: str, folder_path: str, commit_message: str, private: bool = False):
    """Push a folder to Hugging Face Hub"""
    
    api = HfApi()
    
    print(f"=== PUSHING TO HUB ===")
    print(f"Repository: {repo_id}")
    print(f"Folder: {folder_path}")
    print(f"Commit message: {commit_message}")
    print(f"Private: {private}")
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Create repository (will not fail if it already exists)
    print(f"Creating/checking repository: {repo_id}")
    api.create_repo(
        repo_id, 
        repo_type="model", 
        private=private, 
        exist_ok=True
    )
    
    # Upload folder
    print(f"Uploading folder: {folder_path}")
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        ignore_patterns=[
            "*.pt", "*.bin", "**/data/**", "**/checkpoints/**",  # safety
            "*.log", "*.tmp", "**/__pycache__/**", "**/.git/**"   # additional safety
        ]
    )
    
    print(f"✅ Successfully uploaded to: {repo_id}")
    print(f"🔗 View at: https://huggingface.co/{repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Push model to Hugging Face Hub")
    parser.add_argument("repo_id", help="Repository ID (e.g., 'username/model-name')")
    parser.add_argument("folder_path", help="Path to folder to upload")
    parser.add_argument("commit_message", help="Commit message for the upload")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    
    args = parser.parse_args()
    
    try:
        push_to_hub(
            repo_id=args.repo_id,
            folder_path=args.folder_path,
            commit_message=args.commit_message,
            private=args.private
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
