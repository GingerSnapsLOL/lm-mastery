def alpaca_to_text(ex):
    instr = (ex.get("instruction") or "").strip()
    inp   = (ex.get("input") or "").strip()
    out   = (ex.get("output") or "").strip()
    prompt = f"### Instruction:\n{instr}\n"
    if inp:
        prompt += f"\n### Input:\n{inp}\n"
    prompt += "\n### Response:\n" + out
    return {"text": prompt}

def messages_to_text(messages):
    lines=[]
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = (m.get("content") or "").strip()
        lines.append(f"<|{role}|>\n{content}\n")
    lines.append("<|END|>")
    return {"text":"\n".join(lines)}

def ultrachat_to_text(ex):
    # ultrachat_200k has "messages" list [{"role": "...", "content": "..."}...]
    return messages_to_text(ex["messages"])