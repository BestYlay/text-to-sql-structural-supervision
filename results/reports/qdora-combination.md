# QDoRA 联合监督及续训对照

单种子、已查看开发集。GRPO与匹配SFT的数据、步数和计算量不同，分支差异不用于归因GRPO优于SFT。
GRPO信号：通过


## main

|组|EX|可执行|set EX|bag EX|
|---|---:|---:|---:|---:|
|D|42.0%|68.0%|44.0%|43.0%|
|old-no-dpo|44.0%|75.0%|46.0%|45.0%|
|old-matched-sft|45.0%|74.0%|48.0%|46.0%|
|joint|44.0%|75.0%|46.0%|45.0%|
|matched-sft|44.0%|76.0%|48.0%|45.0%|
|grpo|42.0%|75.0%|46.0%|43.0%|

joint - old-no-dpo: +0.0pp；修复1，退步1。 数据库聚类95%区间 [-0.030303030303030304, 0.029411764705882353]。

matched-sft - old-matched-sft: -1.0pp；修复0，退步1。 数据库聚类95%区间 [-0.03, 0.0]。

matched-sft - joint: +0.0pp；修复1，退步1。 数据库聚类95%区间 [-0.029411764705882353, 0.030303030303030304]。

grpo - joint: -2.0pp；修复1，退步3。 数据库聚类95%区间 [-0.06930693069306931, 0.02040816326530612]。

grpo - matched-sft: -2.0pp；修复0，退步2。 数据库聚类95%区间 [-0.04854368932038835, 0.0]。

耗时（旧组为历史参考）：
D：4.94分钟，解析0.00秒，生成4931 token。
old-no-dpo：10.84分钟，解析84.76秒，生成7740 token。
old-matched-sft：10.68分钟，解析84.25秒，生成7795 token。
joint：34.45分钟，解析93.17秒，生成8585 token。
matched-sft：29.64分钟，解析77.07秒，生成7267 token。
grpo：32.15分钟，解析83.27秒，生成7709 token。

## financial

|组|EX|可执行|set EX|bag EX|
|---|---:|---:|---:|---:|
|D|28.3%|82.1%|29.2%|28.3%|
|old-no-dpo|29.2%|84.9%|29.2%|29.2%|
|old-matched-sft|28.3%|84.0%|28.3%|28.3%|
|joint|27.4%|82.1%|27.4%|27.4%|
|matched-sft|28.3%|82.1%|28.3%|28.3%|
|grpo|26.4%|81.1%|26.4%|26.4%|

joint - old-no-dpo: -1.9pp；修复0，退步2。

matched-sft - old-matched-sft: +0.0pp；修复2，退步2。

matched-sft - joint: +0.9pp；修复1，退步0。

grpo - joint: -0.9pp；修复1，退步2。

grpo - matched-sft: -1.9pp；修复1，退步3。
D：原18题恢复0；相对D新增退步0。
old-no-dpo：原18题恢复2；相对D新增退步2。
old-matched-sft：原18题恢复2；相对D新增退步3。
joint：原18题恢复1；相对D新增退步3。
matched-sft：原18题恢复1；相对D新增退步2。
grpo：原18题恢复0；相对D新增退步4。

耗时（旧组为历史参考）：
D：5.92分钟，解析0.00秒，生成6652 token。
old-no-dpo：15.21分钟，解析114.30秒，生成11338 token。
old-matched-sft：15.59分钟，解析121.96秒，生成11810 token。
joint：48.76分钟，解析123.93秒，生成11971 token。
matched-sft：51.58分钟，解析131.36秒，生成12495 token。
grpo：50.60分钟，解析130.48秒，生成12295 token。

## 训练成本
joint-sft：26.49分钟，峰值5.90 GiB。
matched-sft：4.46分钟，峰值5.56 GiB。
grpo：205.04分钟，峰值5.86 GiB。
GRPO generation: 9412.24秒。
GRPO execution: 493.79秒。
GRPO old_logp: 493.87秒。
GRPO reference_logp: 492.98秒。
GRPO backward: 1399.63秒。

GRPO采样、执行等待、反向更新见grpo-stage/grpo/metrics.json；逐题约束日志见groups。
主评测13道超长题及其他失败均保留分母。四项主要比较的区间为逐项描述；financial单库不作数据库聚类推断。
无哈希核对、额外复核或旧模型重放。不按结果挑检查点。
