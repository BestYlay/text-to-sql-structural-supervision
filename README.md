# Text-to-SQL：结构监督微调与约束解码

基于 **Qwen3-4B-Instruct-2507**，在单张 **Tesla V100 16GB** 上探索低显存 Text-to-SQL 优化：

**QLoRA + 冻结基座 KL + SQL AST 结构监督 → 增量语法/绑定约束解码 → 静态诊断后一次重生成。**

在同一份已查看的 BIRD **100题开发子集**上，相对未微调模型，SQL执行正确率由 **36% → 44%**，可执行率由 **71% → 77%**。这是整套方案的结果，不是仅靠一次纠错或单独微调的收益，也不是完整 BIRD 榜单成绩。

## 结果与贡献边界

| 方案 | 主100题正确率 | 主100题可执行率 | financial 106题正确率 | financial可执行率 |
|---|---:|---:|---:|---:|
| 未微调原模型 | 36.0% | 71.0% | 40.6% | 84.0% |
| D：普通 QLoRA SFT | 42.0% | 68.0% | 28.3% | 82.1% |
| 联合监督＋约束解码（旧“去掉DPO”） | 44.0% | 75.0% | 29.2% | 84.9% |
| ＋静态检查与一次纠错 | **44.0%** | **77.0%** | **31.1%** | **89.6%** |

- 一次纠错并未增加主100题正确数：可执行题从75增至77；financial正确题从31增至33。
- financial原18道“原模型正确、D错误”的题目仍恢复2道；本次额外修复的2题属于其他题。
- 单个训练种子 `3407`。主100题有13题超出固定输入窗口，继续计入分母；无失败样本剔除。
- 这里的“正确率”采用原参考项目的执行比较口径：忽略行顺序、保留重复行、单元格转成字符串后比较。另公开官方集合比较及严格类型的重复行敏感结果，避免混淆。
- 这些对照经过开发集反复观察，**不能作为未见测试集泛化结论**。带 KL、结构损失与约束解码同时变化的比较，不能证明某个组件单独贡献全部提升。

原始逐题成功/失败标记、数据库ID及指标见 [`results/`](results/)，其中没有数据库内容、参考SQL或权重。多种失败和负向实验也保留于 [`results/reports/`](results/reports/)。

## 核心方法

### 1. 完成部分监督的 QLoRA

- NF4双重量化，FP16矩阵计算；adapter与非量化参数FP32。
- LoRA rank/alpha=`16/16`，注意力与MLP七类投影，dropout=0。
- 800条训练记录，1 epoch、50个优化步；micro-batch=1、累积16。
- AdamW8bit，峰值学习率`2e-4`、weight decay=0.01、cosine、warmup=2步。
- SQL/EOS参与监督，提示和padding不参与；按累积窗口实际监督token数归一化。
- SDPA MATH、非重入梯度检查点、FP16 GradScaler及完整恢复状态。

### 2. 基座 KL 与 AST 结构辅助损失

```math
L=L_{SQL}+0.1D_{KL}(p_{base}\Vert p_{student})
 +0.1\frac{L_{table}+L_{column}}{2}+0.1L_{edge}
```

- 关闭adapter、停止梯度，在相同参考SQL前缀上得到冻结量化基座的分布。
- 温度1，**完整词表**FP32 log-softmax；SQL/EOS位置使用前向KL，不是top-k近似。
- SQLGlot AST与作用域解析提取表、列、物理列等值关联边，支持CTE和派生列回溯。
- 三个平衡BCE辅助任务使用提示末token隐藏状态；不会读取答案位置隐藏状态。
- schema候选向量来自冻结词嵌入；辅助头只在训练使用。无法可靠解析的标签单独屏蔽，SQL监督保留。

实现见 [`text2sql/schema.py`](text2sql/schema.py)、[`losses.py`](text2sql/losses.py)、[`joint.py`](text2sql/joint.py)。

### 3. 增量语法及作用域绑定约束

- llguidance按token原始字节计算语法掩码，支持SQLite查询、CTE、子查询、JOIN、聚合、窗口函数等。
- 影子词法解析器维护关键词/标识符边界；字符串、引用、Unicode不会通过简单逐token拼接判断。
- 表列及别名作用域检查支持SELECT先于FROM的延迟绑定；只拒绝已确定的非法绑定。
- 在语法允许的token中按模型分数寻找合法候选，不把固定top-k耗尽误认为无解。
- batch=2独立状态，greedy、最多512新token；约束死路及未完成输出保留为失败。

实现见 [`text2sql/constraints.py`](text2sql/constraints.py)、[`decoding.py`](text2sql/decoding.py)及冻结语法。

### 4. 静态检查后一次纠错

只对有检查问题的记录触发。复用原问题、完整schema与evidence，附加字段归属、候选外键路径或语法/完整性诊断，开启**新system/user会话，不传入旧SQL**。

重新生成仍使用同一adapter和约束解码；最多一个候选。仅在完整、正常、非空且无静态硬错误时接受；否则保留原输出。检查和接受逻辑均不访问参考答案或执行正确性。

本轮主集13题、financial22题触发；额外生成共约9.51分钟。静态合法不保证语义正确，非外键JOIN也不自动视为错误。

实现见 [`text2sql/checks.py`](text2sql/checks.py)、[`feedback.py`](text2sql/feedback.py)。

## 快速开始：无需GPU的合成示例

Python 3.12：

```bash
python -m pip install -e .
python -m text2sql demo
python -m unittest tests.test_core -v
```

示例展示为什么 `client.client_id = account.account_id` 虽然引用合法字段，却可能错误关联不同实体，以及如何生成不含旧SQL的候选关系诊断。

## 新训练与推理

本仓库提供从实验代码拆分出的可移植入口，详见 [`docs/RUNNING.md`](docs/RUNNING.md)。GPU原始环境为 PyTorch2.8.0/cu128、Transformers4.57.6、PEFT0.18.0、bitsandbytes0.50.2、SQLGlot27.29.0、llguidance1.9.1。V100使用FP16，**不使用BF16或FlashAttention**。

```bash
python -m text2sql prepare --input data/train.jsonl --metadata data/schema.json \
  --model models/Qwen3-4B-Instruct-2507 --output runs/prepared
python -m text2sql train --prepared runs/prepared \
  --model models/Qwen3-4B-Instruct-2507 --output runs/joint
python -m text2sql infer --input data/eval-prompts.jsonl --metadata data/schema.json \
  --model models/Qwen3-4B-Instruct-2507 --adapter runs/joint/adapter \
  --output runs/predictions.jsonl --repair
python -m text2sql score --predictions runs/predictions.jsonl \
  --references data/evaluation-only/references.jsonl --output runs/scores.jsonl
```

`infer`不读取参考SQL。评分采用独立只读SQLite子进程、30秒、100,000行和64MiB限制，拒绝写操作、ATTACH及扩展加载。

**复现状态**：历史指标来自原实验产物；本次代码整理只运行CPU组件测试，不重新训练或重新生成历史成绩。可移植入口尚未执行完整GPU回归，不能保证重构入口与历史脚本逐token一致。逐次实验原始实现保留在 [`research_snapshot/`](research_snapshot/)，其中工作区相对路径、历史输入依赖须按说明恢复。

## 数据、模型与来源

- 模型：[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)，revision `cdbee75f17c01a7cc42f958dc650907174af0554`。
- 数据：[BIRD过滤训练集](https://huggingface.co/datasets/birdsql/bird23-train-filtered)及官方数据库；训练、主评测和financial数据库互斥。
- 作者配方与执行比较参考：[Shiverion/text2sql-finetuning](https://github.com/Shiverion/text2sql-finetuning)。这是适配与后续方法实验，不宣称逐位复现作者结果。
- 增量约束实现使用 [llguidance](https://github.com/guidance-ai/llguidance)，思路参考 [PICARD](https://github.com/ServiceNow/picard)。

数据库、训练文本、权重及第三方仓库全文均不随此仓库分发。请从原来源取得资源，并遵循各自许可。实现来源及整理边界见 [`NOTICE.md`](NOTICE.md)。
