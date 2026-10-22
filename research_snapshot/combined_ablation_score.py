"""Serial primary scoring and six prespecified paired contrasts."""
import json
import shutil
import sys
from collections import Counter

from combined_ablation import ROOT, OUT, OLD, KL, INC, EVAL, RUNS, GROUPS, SPLITS, CONFIG, infer_binding, validate_binding
from pilot import read, save
from structure_dpo_score import records, put, export
sys.path.insert(0, str(ROOT.parent / 'shiverion-text2sql-reproduction'))
from reproduction_sqlite import execute_readonly, compare_rows
from src.evaluate import result_sets_match

NAMES = {'D': 'D', 'full': '完整组合', 'no-kl': '去掉KL', 'no-structure': '去掉结构监督',
         'no-dpo': '去掉DPO', 'no-parser': '去掉增量解析', 'matched-sft': '匹配SFT续训'}


def folder(group):
    return EVAL / 'D' if group == 'D' else OUT / 'groups' / group


def failed_score(p, ref):
    covered = p['generation_status'] != 'context_limit'
    status = p['generation_status']
    if status == 'ok': status = 'output_incomplete'
    return dict(id=p['id'], db_id=ref['db_id'], valid=False, status=status, error='',
        repository_ex=False, official_set_ex=False, bag_ex=False, context_covered=covered,
        score_source='generation status')


def score(group):
    if group == 'D': return
    summary = read(folder(group) / 'generation-summary.json')
    assert summary['completed']; validate_binding(summary['binding'], infer_binding(group))
    parents = [EVAL / 'baseline', EVAL / 'D', KL, ROOT / 'experiments/D-kl-v1']
    parents += [OLD / g for g in ['structure', 'dpo', 'sft-cont']]
    parents += [INC / m for m in ['syntax', 'syntax-binding']]
    parents += [folder(g) for g in GROUPS if g not in ['D', group]]
    caches = [INC / 'reference-results', OLD / 'reference-results', KL / 'reference-results']
    for split, count in SPLITS.items():
        refs = {r['id']: r for r in read(EVAL / f'evaluation-only/{split}-references.json')}
        path = folder(group) / f'{split}-scores.jsonl'; done = {r['id']: r for r in records(path)}
        reuse = {}
        for parent in parents:
            scores = {r['id']: r for r in records(parent / f'{split}-scores.jsonl')}
            for p in records(parent / f'{split}-predictions.jsonl'):
                if p['id'] in scores:
                    reuse[(p['id'], p['pred_sql'].strip(), p['completed'], p['generation_status'])] = (scores[p['id']], str(parent))
        predictions = records(folder(group) / f'{split}-predictions.jsonl')
        assert len(predictions) == count and {p['id'] for p in predictions} == set(refs)
        assert set(done) <= set(refs)
        for p in predictions:
            i = p['id']
            if i in done: continue
            r = refs[i]; key = (i, p['pred_sql'].strip(), p['completed'], p['generation_status'])
            if not p['completed'] or p['generation_status'] != 'ok' or not p['pred_sql'].strip():
                item = failed_score(p, r)
            elif key in reuse:
                old, source = reuse[key]
                item = dict(old, score_source='reused same sample/SQL/completion status', reused_from=source)
            else:
                cache = OUT / 'reference-results' / f'{split}-{i}.json'
                if not cache.exists():
                    existing = next((c / cache.name for c in caches if (c / cache.name).exists()), None)
                    if existing:
                        cache.parent.mkdir(exist_ok=True); shutil.copyfile(existing, cache)
                if cache.exists(): gold = read(cache)
                else: gold = execute_readonly(r['db_path'], r['sql']); save(cache, gold)
                assert gold['status'] == 'ok', (i, gold['status'])
                pred = execute_readonly(r['db_path'], p['pred_sql']); valid = pred['status'] == 'ok'
                cmp = compare_rows([tuple(x) for x in pred.get('rows', [])], [tuple(x) for x in gold['rows']]) if valid else dict(ex=False, bag=False)
                item = dict(id=i, db_id=r['db_id'], valid=valid, status=pred['status'], error=pred.get('error', ''),
                    repository_ex=valid and result_sets_match(pred['rows'], gold['rows']),
                    official_set_ex=cmp['ex'], bag_ex=cmp['bag'], context_covered=True,
                    score_source='primary readonly execution')
            item['group'] = group
            put(path, item); done[i] = item
            print('SCORE', group, split, len(done), count, item['repository_ex'], flush=True)
        assert len(done) == count; export(path)


def paired(first, second, clustered):
    if clustered:
        from generalization_runtime import contrast
        return contrast(first, second)
    a = {r['id']: r for r in first}; b = {r['id']: r for r in second}
    assert a.keys() == b.keys()
    delta = {i: int(b[i]['repository_ex']) - int(a[i]['repository_ex']) for i in a}
    return dict(delta=sum(delta.values()) / len(delta), repairs=sorted(i for i, v in delta.items() if v == 1),
                harms=sorted(i for i, v in delta.items() if v == -1))


def report():
    from generalization_runtime import summarize
    result = {}; contrasts = {}; recovery = {}; costs = {}; changes = []
    for split, count in SPLITS.items():
        ss = {g: records(folder(g) / f'{split}-scores.jsonl') for g in GROUPS}
        expected = {r['id'] for r in read(OUT / f'{split}-prompts.json')}
        assert all(len(v) == count and {r['id'] for r in v} == expected for v in ss.values())
        result[split] = {g: summarize(v) for g, v in ss.items()}
        contrasts[split] = {'full-' + g: paired(ss[g], ss['full'], split == 'main') for g in GROUPS if g != 'full'}
        maps = {g: {r['id']: r for r in v} for g, v in ss.items()}
        base = {r['id']: r for r in records(EVAL / 'baseline' / f'{split}-scores.jsonl')}
        lost = sorted(i for i in expected if base[i]['repository_ex'] and not maps['D'][i]['repository_ex'])
        recovery[split] = {g: dict(original_losses=lost, recovered=[i for i in lost if maps[g][i]['repository_ex']],
            new_regressions=sorted(i for i in expected if maps['D'][i]['repository_ex'] and not maps[g][i]['repository_ex']),
            versus_D=paired(ss['D'], ss[g], split == 'main')) for g in GROUPS}
        if split == 'financial': assert len(lost) == 18
        costs[split] = {}
        for g in GROUPS:
            pp = records(folder(g) / f'{split}-predictions.jsonl')
            seconds = sum(p['seconds'] for p in pp); tokens = sum(p['generated_tokens'] for p in pp)
            costs[split][g] = dict(seconds=seconds, tokens=tokens, tokens_per_second=tokens / seconds if seconds else 0.,
                parser_seconds=sum(p.get('parser_seconds', 0) for p in pp),
                interventions=sum(p.get('interventions', 0) for p in pp),
                intervention_questions=sum(p.get('interventions', 0) > 0 for p in pp),
                rejected_candidates=sum(p.get('rejected_candidates', 0) for p in pp),
                reasons=dict(Counter(e['original_top_reason'].split(':')[0] for p in pp for e in p.get('events', []))),
                errors=dict(Counter(r.get('error', '').split(':')[0] for r in ss[g] if r.get('error'))),
                historical_timing=g == 'D')
        for comparison, d in contrasts[split].items():
            for direction in ['repairs', 'harms']:
                changes += [dict(split=split, comparison=comparison, direction=direction, id=i) for i in d[direction]]
    training = {r: read(OUT / 'runs' / r / 'summary.json') for r in RUNS}
    parser_delta = {s: costs[s]['full']['seconds'] - costs[s]['no-parser']['seconds'] for s in SPLITS}
    save(OUT / 'comparison.json', dict(groups=result, contrasts=contrasts, recoveries=recovery, costs=costs,
        training=training, parser_extra_seconds=parser_delta, config=CONFIG,
        stages=[read(p) for p in sorted((OUT / 'stage-costs').glob('*.json'))]))
    save(OUT / 'changed-outcomes.json', changes)
    lines = ['# KL＋结构监督＋DPO＋增量解析：7组消融', '',
        'KL=0.1，仅第一阶段使用KL与结构损失；DPO以各自第一阶段模型为冻结参考。结构头不用于第二阶段或推理。',
        '主比较使用最终检查点；D复用原预测，其他六组新生成。全部数据为已查看开发集，seed3407。']
    for split in SPLITS:
        lines += ['', '## ' + split, '', '|组别|EX|可执行率|set EX|bag EX|覆盖率|', '|---|---:|---:|---:|---:|---:|']
        for g, r in result[split].items():
            lines.append(f"|{NAMES[g]}|{r['ex']:.1%}|{r['valid']:.1%}|{r['set_ex']:.1%}|{r['bag_ex']:.1%}|{r['context_coverage']:.1%}|")
        for g in GROUPS:
            if g == 'full': continue
            d = contrasts[split]['full-' + g]
            line = f"完整组合 − {NAMES[g]}：{d['delta']*100:+.1f}pp；修复{len(d['repairs'])}题，退步{len(d['harms'])}题。"
            if split == 'main': line += f" 数据库聚类配对bootstrap95%区间[{d['ci95'][0]*100:+.1f}, {d['ci95'][1]*100:+.1f}]pp。"
            lines += ['', line]
        if split == 'financial':
            lines += ['', '|组别|原18道退步恢复|相对D新增退步|', '|---|---:|---:|']
            for g, r in recovery[split].items():
                lines.append(f"|{NAMES[g]}|{len(r['recovered'])}/18|{len(r['new_regressions'])}|")
        lines += ['', '|组别|生成分钟|token/s|约束CPU秒|介入题数|', '|---|---:|---:|---:|---:|']
        for g, c in costs[split].items():
            lines.append(f"|{NAMES[g]}|{c['seconds']/60:.2f}|{c['tokens_per_second']:.2f}|{c['parser_seconds']:.2f}|{c['intervention_questions']}|")
        lines += ['', f'同一完整组合adapter，增量解析额外生成耗时：{parser_delta[split]:+.2f}秒。',
                  '', '失败状态：' + json.dumps({g: r['statuses'] for g, r in result[split].items()}, ensure_ascii=False)]
    lines += ['', '## 新增训练成本', '', '|任务|优化步|训练分钟|输入token|峰值显存GiB|', '|---|---:|---:|---:|---:|']
    for run, t in training.items():
        lines.append(f"|{run}|{t['steps']}|{t['seconds']/60:.2f}|{t['input_tokens']}|{t['peak_allocated_GiB']:.2f}|")
    delta = contrasts['main']['full-matched-sft']['delta']
    lines += ['', '## 解释边界', '',
        '完整组合对匹配SFT续训控制的主评测EX差为' + f'{delta*100:+.1f}pp。' +
        ('DPO未超过匹配续训控制，不能把组合收益归因于偏好学习。' if delta <= 0 else '正向点估计仍须结合配对区间，不能据此认定稳定提升。'),
        '完整组合对去掉DPO的差异包含追加训练；DPO和匹配控制步数及正样本相同，但负答案和参考计算令FLOPs不同。',
        '七组只反映完整组合下各组件的条件作用，不完整识别交互，也不证明各组件独立有效。六项区间为逐项区间，不是同时置信区间；financial单库不作数据库聚类推断。',
        '若只提高可执行率而EX不变，结论限定为减少非法SQL。绑定合法不保证关联路径、条件或问题语义正确。单种子开发评测不代表通用泛化。',
        '主100题13道超长题、解析死路、输出不完整均保留分母。沿用作者EX/set/bag与只读30秒/100000行/64MiB限制，不做事后纠错或回退搜索。',
        'D耗时为历史参考；新增组采用一致计时。预检、参考概率、加载与评分阶段另见stage-costs；正式训练成本不包含旧组已有成本。',
        '无哈希核对、独立复核或旧模型重放。正式启动自检后后台串行执行，未持续外部监控。',
        '逐步SQL、未加权/加权KL、表列与关联损失、DPO margin、梯度、token、显存见runs/*/metrics.json。逐题约束介入与错误见groups/*。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', 'utf-8')
    save(OUT / 'status.json', dict(status='completed', report=str(OUT / 'REPORT.md')))
    print('RESULT', json.dumps(result, ensure_ascii=False), flush=True)
