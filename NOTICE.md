# 来源与发布说明

作者：Liheng Huang（GitHub: BestYlay）。

此仓库整理了本人的本地Text-to-SQL实验实现：AST辅助标签、冻结基座KL、结构辅助头、约束解码、静态诊断、只读执行及实验报告。

`text2sql/`由这些实际实验函数抽取并整理为不依赖个人目录的模块；`research_snapshot/`保留相关历史脚本的原始实现。历史脚本使用的其他工作区模块和冻结输入未全部发布，应按`docs/RUNNING.md`理解运行边界。

作者配方、提示词和历史执行比较口径参考：
https://github.com/Shiverion/text2sql-finetuning

执行比较兼容该项目的“单元格转字符串后比较行多重集合”规则；这与官方BIRD集合比较并不相同。SQL提取也保留其先截取首条语句的行为。不得把此处的小规模开发结果表述为官方完整BIRD成绩。

主要依赖Qwen、PyTorch、Transformers、PEFT、bitsandbytes、SQLGlot、llguidance，各自代码、模型和数据的许可由原项目提供。本仓库不重新许可或随附第三方模型、BIRD数据库、训练语料或上游仓库全文。

发布结果是已保存历史产物的整理；代码重构后的完整GPU逐token等价性尚未重新验证。没有为了本次发布重跑旧模型或改变实验分母。
