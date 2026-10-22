def encode(messages, tok, limit=2048):
    prefix = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    p = tok(prefix, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    assert ids[:len(p)] == p, "Training/generation template prefix mismatch"
    if len(ids) > limit:
        raise ValueError(f"Full sequence {len(ids)} exceeds {limit}; no truncation")
    labels = [-100] * len(p) + ids[len(p):]
    assert len(p) < len(ids) and tok.eos_token_id in labels[len(p):]
    return {"input_ids": ids, "labels": labels, "prompt_ids": p}

def collate(items, pad):
    import torch
    size = max(len(x["input_ids"]) for x in items)
    return {
        "input_ids": torch.tensor([x["input_ids"] + [pad] * (size-len(x["input_ids"])) for x in items]),
        "labels": torch.tensor([x["labels"] + [-100] * (size-len(x["labels"])) for x in items]),
        "attention_mask": torch.tensor([[1]*len(x["input_ids"]) + [0]*(size-len(x["input_ids"])) for x in items]),
    }

def pad_inputs(rows,pad_id,device):
    import torch
    width=max(len(r['prompt_ids']) for r in rows)
    ids=torch.full((len(rows),width),pad_id,dtype=torch.long,device=device)
    mask=torch.zeros_like(ids)
    for j,r in enumerate(rows):
        n=len(r['prompt_ids']);assert n>0
        ids[j,-n:]=torch.tensor(r['prompt_ids'],device=device);mask[j,-n:]=1
    return ids,mask
