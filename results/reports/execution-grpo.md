# D＋执行奖励 GRPO 对照报告

从同一D继续训练；GRPO与普通SFT使用共同200题，各50个更新机会。

## main

|组别|EX|可执行率|set EX|bag EX|
|---|---:|---:|---:|---:|
|D|42.0%|68.0%|44.0%|43.0%|
|sft-cont|43.0%|70.0%|45.0%|44.0%|
|grpo|43.0%|71.0%|45.0%|44.0%|

grpo-D：+1.0pp，修复2、退步1。 数据库聚类配对bootstrap95%区间[+0.0,+3.1]pp。

grpo-sft-cont：+0.0pp，修复2、退步2。 数据库聚类配对bootstrap95%区间[-3.0,+3.0]pp。

sft-cont-D：+1.0pp，修复1、退步0。 数据库聚类配对bootstrap95%区间[+0.0,+3.1]pp。

生成耗时（秒；D为历史参考）：{"D": {"seconds": 296.3470509999679, "tokens": 4931, "historical": true}, "sft-cont": {"seconds": 247.3499302000564, "tokens": 4509, "historical": false}, "grpo": {"seconds": 237.32856239991088, "tokens": 4370, "historical": false}}

## financial

|组别|EX|可执行率|set EX|bag EX|
|---|---:|---:|---:|---:|
|D|28.3%|82.1%|29.2%|28.3%|
|sft-cont|26.4%|79.2%|27.4%|26.4%|
|grpo|28.3%|82.1%|29.2%|28.3%|

grpo-D：+0.0pp，修复0、退步0。

grpo-sft-cont：+1.9pp，修复2、退步0。

sft-cont-D：-1.9pp，修复0、退步2。

grpo：原18道退步恢复0道，新增退步0道。

sft-cont：原18道退步恢复0道，新增退步2道。

生成耗时（秒；D为历史参考）：{"D": {"seconds": 355.27477369998815, "tokens": 6652, "historical": true}, "sft-cont": {"seconds": 352.9767268999858, "tokens": 6687, "historical": false}, "grpo": {"seconds": 348.5227056000149, "tokens": 6640, "historical": false}}

## 训练信号与成本

{"passed": true, "questions": 32, "candidates": 128, "mixed_groups": 15, "all_correct": 3, "all_wrong": 14, "truncated": 0, "correct": 46, "success_rate": 0.359375, "generation_seconds": 393.3769893999706, "generation_tokens": 7676, "execution_seconds": 44.70031190002919, "criteria": {"min_mixed": 8, "max_truncated": 12}}
两组匹配问题数与计划更新机会，不匹配实际token/FLOPs。策略零优势窗口与数值跳步分开计数。

grpo: 50窗口，实际更新50次，有非零策略优势窗口47；总阶段69.88分钟；峰值显存5.46GiB。
generation: 2530.48秒。
execution: 418.96秒。
old_logp: 219.46秒。
reference_logp: 219.63秒。
反向更新: 795.91秒。

sft-cont: 50窗口，实际更新50次，有非零策略优势窗口0；总阶段2.43分钟；峰值显存5.08GiB。
generation: 0.00秒。
execution: 0.00秒。
old_logp: 0.00秒。
reference_logp: 0.00秒。
反向更新: 145.72秒。

## 解释

GRPO超过D，但未超过普通SFT续训，不能将收益归因于执行奖励GRPO。
单种子、已查看开发集；不证明通用泛化。主100题13道超长继续计失败，financial为辅助诊断。两项主要区间是逐项描述，不是同时置信区间。
训练奖励使用严格单SQL提取，最终评测沿用历史提取规则；KL参考为冻结D，不能与旧基座KL实验直接等同。未使用解析约束、纠错、QDoRA或best-of-4。
无哈希核对、额外复核或旧模型重放。检查点与逐窗口rollout可用于恢复；仅比较最终adapter。
