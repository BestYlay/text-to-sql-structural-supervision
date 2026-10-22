"""Seven-group, two-stage ablation. No hashes, review, or background monitoring."""
import argparse
import copy
import json
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import pilot
from pilot import read, save
from structure_dpo_data import CONFIG as OLD_CONFIG
from structure_dpo_score import records, put, export

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiments/combined-ablation-v1'
OLD = ROOT / 'experiments/structure-dpo-v1'
KL = ROOT / 'experiments/D-kl-01-v1'
INC = ROOT / 'experiments/incremental-sql-v1'
EVAL = ROOT / 'experiments/hardpair-v1/inference-batch2-v1'
RUNS = ['joint-sft', 'full-dpo', 'no-kl-dpo', 'no-structure-dpo', 'matched-sft']
GROUPS = {
    'D': dict(run=None, decode='none'),
    'full': dict(run='full-dpo', decode='syntax-binding'),
    'no-kl': dict(run='no-kl-dpo', decode='syntax-binding'),
    'no-structure': dict(run='no-structure-dpo', decode='syntax-binding'),
    'no-dpo': dict(run='joint-sft', decode='syntax-binding'),
    'no-parser': dict(run='full-dpo', decode='none'),
    'matched-sft': dict(run='matched-sft', decode='syntax-binding'),
}
CONFIG = copy.deepcopy(OLD_CONFIG)
CONFIG['experiment'] = 'combined-ablation-v1'
CONFIG['kl'] = dict(weight=.1, direction='base || student', temperature=1., vocabulary='full',
                    positions='assistant and EOS', normalization='window supervised tokens')
CONFIG['preference']['initialization'] = 'branch-specific stage-one adapter'
CONFIG['preference']['reference'] = 'frozen branch-specific stage-one adapter'
CONFIG['groups'] = GROUPS
CONFIG['stage_boundary'] = 'KL/structure only in stage one; pure DPO or chosen-only SFT in stage two'
CONFIG['rules_source'] = str(INC)
SPLITS = {'main': 100, 'financial': 106}


def parent_adapter(run):
    if run == 'joint-sft':
        return None
    if run == 'no-kl-dpo':
        return OLD / 'structure/adapter'
    if run == 'no-structure-dpo':
        return KL / 'adapter'
    if run in ['full-dpo', 'matched-sft']:
        return OUT / 'runs/joint-sft/adapter'
    raise ValueError(run)


def run_spec(run):
    if run not in RUNS:
        raise ValueError(run)
    joint = run == 'joint-sft'
    return dict(run=run, phase='joint-sft' if joint else 'preference',
                objective='joint' if joint else ('sft-cont' if run == 'matched-sft' else 'dpo'),
                parent=str(parent_adapter(run)) if parent_adapter(run) else None,
                reference=str(parent_adapter(run)) if run.endswith('-dpo') else None,
                samples=800 if joint else 192, steps=50 if joint else 24,
                accumulation=16 if joint else 8, lr=2e-4 if joint else 5e-6,
                warmup=2 if joint else 1, kl_weight=.1 if joint else 0.,
                structure_weight=.1 if joint else 0., dpo_beta=.1)


def frozen(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('Existing configuration/data differs: ' + str(path))
    else:
        save(path, value)


def validate_binding(actual, expected):
    if actual != expected:
        raise ValueError('Run, parent, configuration, or data order mismatch')


def prepare():
    if (OUT / 'prepared.json').exists():
        validate_binding(read(OUT / 'config.json'), CONFIG)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    train = read(OLD / 'train.json')
    original = read(ROOT / 'experiments/data-mixture-v1/frozen/D/train.json')
    assert len(train) == 800 and [r['id'] for r in train] == [r['id'] for r in original]
    for a, b in zip(train, original):
        assert all(a[k] == b[k] for k in ['sql', 'input_ids', 'labels', 'prompt_ids'])
    preferences = read(OLD / 'preferences.json')
    assert len(preferences) == 192 and len({r['id'] for r in preferences}) == 192
    train_ids = {r['id'] for r in train}
    assert all(r['id'] in train_ids for r in preferences)
    eos = pilot.tokenizer().eos_token_id
    for r in train + [r[k] for r in preferences for k in ['chosen', 'rejected']]:
        assert len(r['input_ids']) <= 2048 and len(r['prompt_ids']) <= 1792
        n = len(r['prompt_ids'])
        assert r['input_ids'][:n] == r['prompt_ids']
        assert r['labels'][:n] == [-100] * n and len(r['labels']) == len(r['input_ids'])
        assert r['labels'][n:] == r['input_ids'][n:] and eos in r['labels'][n:]
    frozen(OUT / 'config.json', CONFIG)
    for name in ['train.json', 'preferences.json', 'structure-labels.json', 'schema-candidates.json']:
        frozen(OUT / name, read(OLD / name))
    for name in ['metadata.json', 'training-metadata.json', 'profile-prompts.json']:
        frozen(OUT / name, read(INC / name))
    # Freeze the already evaluated parser, without editing or re-deriving its rules.
    shutil.copyfile(INC / 'grammar.lark', OUT / 'grammar.lark')
    for split, count in SPLITS.items():
        rows = read(EVAL / f'{split}-prompts.json')
        assert len(rows) == count and len({r['id'] for r in rows}) == count
        assert sum(not r['context_covered'] for r in rows) == (13 if split == 'main' else 0)
        assert not ({r['db_id'] for r in rows} & {r['db_id'] for r in train})
        frozen(OUT / f'{split}-prompts.json', rows)
    for adapter in [parent_adapter('no-kl-dpo'), parent_adapter('no-structure-dpo')]:
        assert (adapter / 'adapter_model.safetensors').exists()
    for run in RUNS:
        frozen(OUT / 'runs' / run / 'binding.json', binding(run))
    for group in GROUPS:
        frozen(OUT / 'groups' / group / 'binding.json', infer_binding(group))
    code = OUT / 'code'
    code.mkdir(exist_ok=True)
    for name in ['combined_ablation.py', 'combined_ablation_score.py', 'test_combined_ablation.py',
                 'd_kl.py', 'structure_dpo.py', 'incremental_sql.py', 'incremental_constraints.py']:
        shutil.copyfile(ROOT / name, code / name)
    save(OUT / 'prepared.json', dict(completed=True, train=800, preferences=192, groups=GROUPS,
         reused=dict(D=str(EVAL / 'D'), structure=str(OLD / 'structure'), kl=str(KL)),
         no_hashes=True, no_extra_review=True))
    (OUT / 'RUN_GUIDE.md').write_text(
        '# 核心7组组合消融\n\nKL=0.1；结构权重0.1/0.1；DPO beta=0.1；语法＋绑定约束。\n'
        '800条/50步联合SFT，三个192对/24步DPO分支，一个匹配24步SFT续训控制。\n'
        '主100题、financial106题；batch2、greedy512。D预测直接复用。\n'
        '启动命令：`python combined_ablation.py pipeline`。同命令支持完整检查点及逐题恢复。\n'
        '首个正式更新确认后不持续监控；失败查看 status.json 指定日志。\n'
        '预期4–6小时GPU阶段，CPU评分另计；不自动增加组数或轮数。\n'
        '结果见 REPORT.md / comparison.json；每个分支使用自己的冻结DPO参考。\n', 'utf-8')


def items_for(run):
    if run == 'joint-sft':
        aux = {r['id']: r for r in read(OUT / 'structure-labels.json')}
        return [dict(r, aux=aux[r['id']]) for r in read(OUT / 'train.json')]
    return read(OUT / 'preferences.json')


def binding(run):
    return dict(config=CONFIG, spec=run_spec(run), ids=[r['id'] for r in items_for(run)])


def infer_binding(group):
    cfg = GROUPS[group]
    adapter = ROOT / 'experiments/data-mixture-v1/D/adapter' if group == 'D' else OUT / 'runs' / cfg['run'] / 'adapter'
    return dict(config=CONFIG, group=group, adapter=str(adapter), decode=cfg['decode'],
                ids={s: [r['id'] for r in read(OUT / f'{s}-prompts.json')] for s in SPLITS})


def joint_loss(ce, kl, table, column, edge, tokens, samples, kl_weight=.1, structure_weight=.1):
    if tokens <= 0 or samples <= 0:
        raise ValueError('Empty accumulation window')
    loss = ce / tokens
    if kl_weight:
        loss = loss + kl_weight * kl / tokens
    if structure_weight:
        loss = loss + structure_weight * ((table + column) / 2 + edge) / samples
    return loss


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
    import d_kl
    from structure_dpo import balanced_bce
    params = [p for p in model.parameters() if p.requires_grad] + list(heads.parameters())
    opt.zero_grad(set_to_none=True)
    denominator = sum(sum(v != -100 for v in r['labels'][1:]) for r in rows)
    metrics = dict(sql_loss=0., kl_loss=0., table_loss=0., column_loss=0., edge_loss=0., teacher_forward_seconds=0.)
    for row in rows:
        batch = {k: v.to('cuda') for k, v in pilot.collate([row], tok.pad_token_id).items()}
        torch.cuda.synchronize(); begun = time.perf_counter()
        logp = teacher_logprobs(model, batch)
        torch.cuda.synchronize(); metrics['teacher_forward_seconds'] += time.perf_counter() - begun
        aux = row['aux']
        logits, targets, context = supervised_forward(model, batch, len(row['prompt_ids']) - 1 if aux['enabled'] else None)
        ce, kl = d_kl.loss_sums(logits, logp, targets)
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


def load_model(run, training=False, reference=False):
    pilot.OUT = OUT / 'runs' / run / 'runtime'
    pilot.OUT.mkdir(parents=True, exist_ok=True)
    if training:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        model, tok = pilot.load(train=True)
        parent = parent_adapter(run)
        if parent:
            set_peft_model_state_dict(model, load_file(str(parent / 'adapter_model.safetensors')))
        return model, tok
    return pilot.load(train=False, adapter=parent_adapter(run) if reference else OUT / 'runs' / run / 'adapter')


def reference(run):
    import torch
    from structure_dpo import model_logp
    assert run_spec(run)['objective'] == 'dpo'
    b = binding(run); folder = OUT / 'runs' / run
    frozen(folder / 'reference-binding.json', b)
    path = folder / 'reference-logps.jsonl'
    done = {r['id']: r for r in records(path)}
    items = items_for(run)
    assert set(done) <= {r['id'] for r in items}
    model = tok = None
    for r in items:
        if r['id'] in done:
            continue
        if model is None:
            model, tok = load_model(run, reference=True)
            model.eval()
        with torch.no_grad():
            a, _ = model_logp(model, tok, r['chosen'])
            z, _ = model_logp(model, tok, r['rejected'])
        assert bool(torch.isfinite(a)) and bool(torch.isfinite(z))
        item = dict(id=r['id'], chosen=float(a), rejected=float(z), parent=str(parent_adapter(run)))
        put(path, item); done[r['id']] = item
        print('REFERENCE', run, len(done), len(items), flush=True)
    export(path)
    save(folder / 'reference-summary.json', dict(completed=True, count=len(done), binding=b))


def save_checkpoint(folder, model, heads, opt, sched, scaler, b, step, history):
    import torch
    import numpy as np
    from peft import get_peft_model_state_dict
    state = dict(binding=b, step=step, position=step * b['spec']['accumulation'], history=history,
        adapter={k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()},
        heads=heads.state_dict() if heads else None, optimizer=opt.state_dict(), scheduler=sched.state_dict(),
        scaler=scaler.state_dict(), rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
        python=random.getstate(), numpy=np.random.get_state()))
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / 'state.pt.tmp'; torch.save(state, tmp); tmp.replace(folder / 'state.pt')
    save(folder / 'complete.json', dict(step=step, position=state['position'], state='state.pt'))


def restore_optimizer_runtime(state, opt, sched, scaler):
    import torch
    import numpy as np
    opt.load_state_dict(state['optimizer']); sched.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
    torch.set_rng_state(state['rng']['torch']); torch.cuda.set_rng_state_all(state['rng']['cuda'])
    random.setstate(state['rng']['python']); np.random.set_state(state['rng']['numpy'])


def train(run, profile=False):
    import numpy as np
    import torch
    import bitsandbytes as bnb
    import structure_dpo as sd
    from peft import set_peft_model_state_dict
    from transformers import get_cosine_schedule_with_warmup
    b = binding(run); cfg = b['spec']; joint = run == 'joint-sft'
    root = OUT / 'runs' / run
    validate_binding(read(root / 'binding.json'), b)
    folder = root / 'profile' if profile else root
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'summary.json').exists():
        summary = read(folder / 'summary.json')
        assert summary['completed']; validate_binding(summary['binding'], b)
        return
    if not profile:
        assert read(root / 'profile/summary.json')['completed']
    items = items_for(run)
    if cfg['objective'] == 'dpo':
        validate_binding(read(root / 'reference-binding.json'), b)
        refs = {r['id']: r for r in records(root / 'reference-logps.jsonl')}
        assert set(refs) == {r['id'] for r in items}
        assert all(r['parent'] == cfg['parent'] for r in refs.values())
        items = [dict(r, reference_chosen=refs[r['id']]['chosen'], reference_rejected=refs[r['id']]['rejected']) for r in items]
    total = (3 if joint else 2) if profile else cfg['steps']
    accum = cfg['accumulation']
    if profile:
        longest = max(items, key=lambda r: len(r['input_ids']) if joint else len(r['chosen']['input_ids']) + len(r['rejected']['input_ids']))
        items = [longest] * (total * accum)
    assert len(items) == total * accum
    model, tok = load_model(run, training=True); model.train()
    heads = features = None
    if joint:
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(3407)
            heads = sd.StructureHeads(model.config.hidden_size).cuda().float()
        sd.OUT = OUT; features = sd.schema_features(model)
    params = [p for p in model.parameters() if p.requires_grad] + (list(heads.parameters()) if heads else [])
    opt = bnb.optim.AdamW8bit(params, lr=cfg['lr'], weight_decay=.01)
    sched = (torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.) if profile else
             get_cosine_schedule_with_warmup(opt, num_warmup_steps=cfg['warmup'], num_training_steps=total))
    scaler = torch.amp.GradScaler('cuda', init_scale=256)
    history = []; start = 0
    if not profile:
        checkpoints = sorted((root / 'checkpoints').glob('step-*/complete.json'))
        if checkpoints:
            state = torch.load(checkpoints[-1].parent / 'state.pt', map_location='cpu', weights_only=False)
            validate_binding(state['binding'], b)
            assert state['position'] == state['step'] * accum
            set_peft_model_state_dict(model, state['adapter'])
            if heads: heads.load_state_dict(state['heads'])
            restore_optimizer_runtime(state, opt, sched, scaler)
            history = state['history']; start = state['step']; del state
    begun = time.perf_counter()
    try:
        for step in range(start, total):
            before = time.perf_counter(); window = items[step * accum:(step + 1) * accum]
            m = (joint_update(model, tok, heads, features, opt, sched, scaler, window) if joint else
                 sd.one_update(model, tok, None, None, opt, sched, scaler, window, cfg['objective']))
            m.update(step=step + 1, position=(step + 1) * accum, seconds=time.perf_counter() - before)
            history.append(m); save(folder / 'metrics.json', history)
            print('STEP', run, 'profile' if profile else 'formal', step + 1, total, m, flush=True)
            elapsed = time.perf_counter() - begun
            (folder / 'progress.txt').write_text(f'{step+1}/{total} elapsed={elapsed:.1f}s estimated_remaining={elapsed/(step+1-start)*(total-step-1):.1f}s', 'utf-8')
            # Cosine warmup starts at LR=0. Confirm an effective optimizer update,
            # rather than calling the zero-learning-rate warmup step training-ready.
            if not profile and m['learning_rate'] > 0 and not (root / 'startup-ready.json').exists():
                save(root / 'startup-ready.json', dict(passed=True, run=run, finite_update=True, metrics=m))
                if joint: save(OUT / 'startup-ready.json', dict(passed=True, run=run, finite_update=True, metrics=m))
            if not profile and ((step + 1) % 10 == 0 or step + 1 == total):
                save_checkpoint(root / 'checkpoints' / f'step-{step+1:04d}', model, heads, opt, sched, scaler, b, step + 1, history)
        if not profile:
            model.save_pretrained(root / 'adapter', safe_serialization=True); tok.save_pretrained(root / 'adapter')
            if heads: torch.save(heads.state_dict(), root / 'structure-heads.pt')
        save(folder / 'summary.json', dict(completed=True, profile=profile, binding=b, steps=total,
            samples=sum(m['samples'] for m in history), seconds=sum(m['seconds'] for m in history),
            input_tokens=sum(m['input_tokens'] for m in history),
            supervised_tokens=sum(m.get('supervised_tokens', m.get('chosen_supervised_tokens', 0)) for m in history),
            teacher_seconds=sum(m.get('teacher_forward_seconds', 0) for m in history),
            peak_allocated_GiB=torch.cuda.max_memory_allocated() / 1024**3, skipped_updates=0))
    except BaseException as error:
        save(folder / 'failure.json', dict(error=repr(error), completed_steps=len(history)))
        raise


def load_group(group):
    b = infer_binding(group)
    validate_binding(read(OUT / 'groups' / group / 'binding.json'), b)
    assert group != 'D' and read(OUT / 'runs' / GROUPS[group]['run'] / 'summary.json')['completed']
    pilot.OUT = OUT / 'groups' / group / 'runtime'; pilot.OUT.mkdir(parents=True, exist_ok=True)
    return pilot.load(train=False, adapter=Path(b['adapter']))


def parser_profile():
    import incremental_sql as inc
    import torch
    inc.OUT = OUT
    model, tok = load_group('full')
    rows = read(OUT / 'profile-prompts.json'); meta = read(OUT / 'training-metadata.json')
    outputs = []
    for i in range(0, len(rows), 2):
        outputs += inc.generate(model, tok, rows[i:i+2], meta, 'syntax-binding')
    save(OUT / 'parser-profile.json', dict(passed=True, outputs=outputs,
         peak_allocated_GiB=torch.cuda.max_memory_allocated() / 1024**3))


def infer(group):
    if group == 'D': return
    import torch
    import incremental_sql as inc
    from hardpair_batch2 import batches, generate_batch
    inc.OUT = OUT
    assert read(OUT / 'parser-profile.json')['passed']
    b = infer_binding(group); folder = OUT / 'groups' / group
    validate_binding(read(folder / 'binding.json'), b)
    model, tok = load_group(group); meta = read(OUT / 'metadata.json'); summaries = {}
    for split, count in SPLITS.items():
        path = folder / f'{split}-predictions.jsonl'
        rows = read(OUT / f'{split}-prompts.json'); done = {r['id']: r for r in records(path)}
        assert set(done) <= {r['id'] for r in rows}
        for r in done.values():
            assert r['group'] == group and r['adapter'] == b['adapter'] and r['decode'] == b['decode']
        for r in rows:
            if not r['context_covered'] and r['id'] not in done:
                p = dict(id=r['id'], db_id=r['db_id'], group=group, adapter=b['adapter'], decode=b['decode'],
                    raw_output='', pred_sql='', token_ids=[], generated_tokens=0, prompt_tokens=len(r['prompt_ids']),
                    completed=False, generation_status='context_limit', seconds=0., events=[], interventions=0, parser_seconds=0.)
                put(path, p); done[r['id']] = p
        for pair in batches(rows):
            if all(r['id'] in done for r in pair): continue
            torch.cuda.synchronize()
            output = (inc.generate(model, tok, pair, meta, 'syntax-binding') if b['decode'] == 'syntax-binding'
                      else generate_batch(model, tok, pair))
            for p in output:
                if not p['completed'] and p['generation_status'] == 'ok': p['generation_status'] = 'output_incomplete'
                p.update(group=group, adapter=b['adapter'], decode=b['decode'])
                if p['id'] not in done: put(path, p); done[p['id']] = p
            save(OUT / 'progress.json', dict(group=group, split=split, completed=len(done), total=count))
            print('INFER', group, split, len(done), count, flush=True)
        assert len(done) == count; export(path)
        summaries[split] = dict(seconds=sum(r['seconds'] for r in done.values()),
            generated_tokens=sum(r['generated_tokens'] for r in done.values()),
            parser_seconds=sum(r.get('parser_seconds', 0) for r in done.values()),
            statuses=dict(Counter(r['generation_status'] for r in done.values())))
    save(folder / 'generation-summary.json', dict(completed=True, binding=b, splits=summaries,
         peak_allocated_GiB=torch.cuda.max_memory_allocated() / 1024**3))


def pipeline():
    prepare()
    jobs = [['profile', '--run', 'joint-sft'], ['train', '--run', 'joint-sft']]
    for run in RUNS[1:]:
        if run_spec(run)['objective'] == 'dpo': jobs.append(['reference', '--run', run])
        jobs += [['profile', '--run', run], ['train', '--run', run]]
    jobs += [['profile', '--group', 'full']]
    jobs += [['infer', '--group', g] for g in GROUPS if g != 'D']
    jobs += [['score', '--group', g] for g in GROUPS if g != 'D'] + [['report']]
    (OUT / 'logs').mkdir(exist_ok=True)
    for args in jobs:
        key = f'{int(time.time())}-{uuid.uuid4().hex[:6]}'
        log = OUT / 'logs' / (key + '-' + '-'.join(args) + '.log')
        begun = time.time(); save(OUT / 'status.json', dict(status='running', stage=args, log=str(log)))
        with log.open('w', encoding='utf-8') as stream:
            child = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), *args], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUTF8='1'),
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        save(OUT / 'stage-costs' / (key + '.json'), dict(args=args, seconds=time.time()-begun, returncode=child.returncode))
        if child.returncode:
            save(OUT / 'status.json', dict(status='failed', stage=args, log=str(log)))
            raise SystemExit(child.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'profile', 'reference', 'train', 'infer', 'score', 'report', 'pipeline'])
    parser.add_argument('--run', choices=RUNS)
    parser.add_argument('--group', choices=list(GROUPS))
    a = parser.parse_args()
    if a.stage in ['prepare', 'pipeline']: return globals()[a.stage]()
    validate_binding(read(OUT / 'config.json'), CONFIG)
    if a.stage == 'profile':
        if a.run: return train(a.run, profile=True)
        if a.group == 'full': return parser_profile()
        parser.error('profile requires --run or --group full')
    if a.stage in ['train', 'reference']:
        if not a.run: parser.error('--run is required')
        return globals()[a.stage](a.run)
    if a.stage == 'infer':
        if not a.group: parser.error('--group is required')
        return infer(a.group)
    import combined_ablation_score as scoring
    if a.stage == 'score':
        if not a.group: parser.error('--group is required')
        return scoring.score(a.group)
    return scoring.report()


if __name__ == '__main__':
    main()
