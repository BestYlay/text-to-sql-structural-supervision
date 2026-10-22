# 运行与复现边界

## 资源与环境

原实验：Windows、Tesla V100-SXM2 16GB、Python3.12、PyTorch2.8.0+cu128。

新环境先安装适合本机驱动与GPU的PyTorch，再安装依赖；不要在已有实验环境直接升级：

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[gpu]'
```

模型从Hugging Face按固定revision下载到本地（需要足够空间及网络）：

```python
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B-Instruct-2507',
    revision='cdbee75f17c01a7cc42f958dc650907174af0554',
    local_dir='models/Qwen3-4B-Instruct-2507')
```

加载器只读本地模型。FP32未量化参数用于保持实验精度口径一致；不是全模型FP32训练。

## 输入格式

`examples/train.jsonl`是合成格式示例。真实训练JSONL每行包含：

```json
{"id":"sample-id","db_id":"database-id","messages":[{"role":"system","content":"..."},{"role":"user","content":"完整schema、evidence、问题"}],"sql":"SELECT ..."}
```

- 顺序即训练顺序；准备阶段不额外打乱。
- 模板使用Qwen3原生非思考模板；只对assistant SQL代码块和EOS监督。
- 上限：输入1792、完整训练2048；超长记录会写入排除清单，不静默截断。
- 元数据格式见`examples/schema.json`，按db_id索引，保留完整表列类型、主键和外键。生成输入中不得根据参考SQL裁剪schema。
- 评测JSONL仅含`id/db_id/messages`，不能包含参考答案。评分参考文件另外提供`id/db_id/db_path/sql`。

历史800条及主100/financial106的数据库、ID和顺序见`results/split-ids.json`，其中**不包含训练文本或参考SQL**。完整历史数据选择还依赖原工作区的已冻结清单及上游官方数据，不能只凭一个随机种子从官方全量数据恢复。

因此：可移植入口可以训练和评测按上述格式准备的新输入；历史结果由发布的逐题标记支持，但本仓库不是一条命令从零完整复现所有历史数据构造的脚本。

## 训练

`train`默认从基础模型初始化LoRA，一轮、micro-batch1、累积16、峰值2e-4、cosine、warmup2（极小合成输入时缩短）。800条对应50步。最后不足16条按实际样本与监督token数归一化。

每10步及最终保存adapter、结构头、优化器、调度器、GradScaler、Python/NumPy/Torch/CUDA随机数状态和位置；相同命令读取`resume.pt`继续。不同配置或训练ID顺序拒绝恢复。仅恢复自己生成的受信任检查点。

`metrics.json`分别记录SQL/KL/表/列/边损失、加权分量、梯度、缩放值、token与显存。NaN、Inf、数值跳步或OOM停止，不自动换精度、缩短输入或删题。

## 推理与评分

推理固定batch2、greedy、最多512新token。`--repair`开启静态检查后一次重生成，仍保持语法和绑定约束。

输出按批次原子保存，已有完整样本保留；清理输出或换组时使用新文件。原实验的逐候选持久化恢复流程保留在`research_snapshot/no_dpo_static_repair.py`；便携入口的恢复粒度是完成批次。

评分参考答案仅在`score`读取。资源限制、语法/输出失败保留分母；参考SQL异常停止并报告，不能删掉后继续报分。

## 测试状态

```bash
python -m unittest tests.test_core -v
python -m unittest tests.test_losses -v  # 已安装PyTorch时
```

这些测试验证AST标签、CTE、JOIN告警、新会话无旧SQL、接受/回退、重复行口径、只读/多语句拒绝/超时、KL方向、辅助梯度和padding。

它们不是完整GPU训练与重载一致性的替代证明。本次整理未触发新的模型训练或历史评测。
