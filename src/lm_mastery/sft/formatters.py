"""
SFT dataset formatters for different instruction formats
"""

def alpaca_to_text(example):
    """Format Alpaca/Dolly example to prompt/completion format for SFTTrainer"""
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output = example.get("response", "").strip()  # Dolly uses 'response' not 'output'
    
    prompt = f"### Instruction:\n{instruction}\n"
    if input_text:
        prompt += f"\n### Input:\n{input_text}\n"
    prompt += "\n### Response:\n"
    
    return {"prompt": prompt, "completion": output}

def ultrachat_to_text(example):
    """Format UltraChat example to prompt/completion format for SFTTrainer"""
    messages = example.get("messages", [])
    if not messages:
        return {"prompt": "", "completion": ""}
    
    # UltraChat format: messages is a list of [{"role": "user/assistant", "content": "..."}]
    # For SFT, we want to create a conversation flow
    # Strategy: take first user message as prompt, first assistant response as completion
    
    user_messages = [msg for msg in messages if msg.get("role") == "user"]
    assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
    
    if user_messages and assistant_messages:
        # Simple approach: first exchange
        prompt = user_messages[0].get("content", "").strip()
        completion = assistant_messages[0].get("content", "").strip()
        return {"prompt": prompt, "completion": completion}
    elif user_messages:
        # Only user messages - use first as prompt
        prompt = user_messages[0].get("content", "").strip()
        return {"prompt": prompt, "completion": ""}
    elif assistant_messages:
        # Only assistant messages - use as completion
        return {"prompt": "", "completion": assistant_messages[0].get("content", "").strip()}
    else:
        # Fallback for any other format
        return {"prompt": "", "completion": ""}

def chat_to_text(example):
    """Generic chat format converter to prompt/completion"""
    messages = example.get("messages", [])
    if not messages:
        return {"prompt": "", "completion": ""}
    
    # Simple approach: first message as prompt, rest as completion
    if len(messages) >= 2:
        prompt = messages[0].get("content", "").strip()
        completion = messages[1].get("content", "").strip()
        return {"prompt": prompt, "completion": completion}
    else:
        # Single message case
        content = messages[0].get("content", "").strip()
        return {"prompt": content, "completion": ""}
