import time
from .batching import collate
from .losses import balanced_bce,loss_sums,joint_loss

def supervised_forward(model, batch, prompt_end=None):
    """Capture the student prompt state only, never a base-teacher forward."""
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    labels = batch['labels']
    positions = torch.nonzero(labels[0, 1:] != -100).flatten()
    targets = labels[0, positions + 1]
    states = []
    handle = None
    if prompt_end is not None:
        handle = model.get_base_model().model.norm.register_forward_hook(
            lambda module, args, output: states.append(output[0, prompt_end].clone()))
    try:
        with sdpa_kernel(SDPBackend.MATH):
            output = model(**{k: v for k, v in batch.items() if k != 'labels'},
                           use_cache=False, logits_to_keep=positions)
            logits = output.logits[0]
    finally:
        if handle is not None:
            handle.remove()
    return logits, targets, states[0] if states else None

def teacher_logprobs(model, batch):
    import torch
    import torch.nn.functional as F
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        logits, _, _ = supervised_forward(model, batch)
        result = F.log_softmax(logits.float(), dim=-1).detach()
    model.train()
    return result

def joint_update(model, tok, heads, features, opt, sched, scaler, rows):
    import torch
    params = [p for p in model.parameters() if p.requires_grad] + list(heads.parameters())
    opt.zero_grad(set_to_none=True)
    denominator = sum(sum(v != -100 for v in r['labels'][1:]) for r in rows)
    metrics = dict(sql_loss=0., kl_loss=0., table_loss=0., column_loss=0., edge_loss=0., teacher_forward_seconds=0.)
    for row in rows:
        batch = {k: v.to('cuda') for k, v in collate([row], tok.pad_token_id).items()}
        torch.cuda.synchronize(); begun = time.perf_counter()
        logp = teacher_logprobs(model, batch)
        torch.cuda.synchronize(); metrics['teacher_forward_seconds'] += time.perf_counter() - begun
        aux = row['aux']
        logits, targets, context = supervised_forward(model, batch, len(row['prompt_ids']) - 1 if aux['enabled'] else None)
        ce, kl = loss_sums(logits, logp, targets)
        lt = lc = le = logits.new_zeros(())
        if aux['enabled']:
            tables, columns = features[row['db_id']]
            edges = torch.tensor(aux['edge_items'], device='cuda', dtype=torch.long).reshape(-1, 2)
            pred = heads(context, tables, columns, edges)
            truth = [torch.tensor(aux[k], device='cuda') for k in ['table_labels', 'column_labels', 'edge_labels']]
            lt, lc, le = [balanced_bce(x, y) for x, y in zip(pred, truth)]
        loss = joint_loss(ce, kl, lt, lc, le, denominator, len(rows))
        assert all(bool(torch.isfinite(x)) for x in [ce, kl, lt, lc, le, loss])
        assert float(kl.detach()) >= -1e-3
        for key, val, norm in [('sql_loss', ce, denominator), ('kl_loss', kl, denominator),
                               ('table_loss', lt, len(rows)), ('column_loss', lc, len(rows)), ('edge_loss', le, len(rows))]:
            metrics[key] += float(val.detach()) / norm
        scaler.scale(loss).backward()
        del logits, targets, context, logp, ce, kl, lt, lc, le, loss, batch
        if aux['enabled']:
            del pred, truth
    scaler.unscale_(opt)
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
    assert any(p.grad is not None and bool((p.grad != 0).any()) for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
    lr = opt.param_groups[0]['lr']; before = scaler.get_scale()
    scaler.step(opt); scaler.update(); assert scaler.get_scale() >= before, 'Skipped update'
    sched.step(); torch.cuda.synchronize()
    metrics.update(weighted_kl=.1 * metrics['kl_loss'],
        weighted_structure=.1 * ((metrics['table_loss'] + metrics['column_loss']) / 2 + metrics['edge_loss']))
    metrics.update(loss=metrics['sql_loss'] + metrics['weighted_kl'] + metrics['weighted_structure'],
        grad_norm=float(norm), scale=scaler.get_scale(), learning_rate=lr, samples=len(rows),
        input_tokens=sum(len(r['input_ids']) for r in rows), supervised_tokens=denominator,
        peak_allocated_GiB=torch.cuda.max_memory_allocated() / 1024**3)
    return metrics
