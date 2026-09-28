# 神农中医 Qwen V5.2 安全增量训练与盲测集构建最终报告

生成日期：2026-09-10  
项目目录：`/home/cyh/Medical_Qwen`  
最终判定：**NOT_PUBLISHABLE（不可发布，仅保留为实验归档）**

## 1. 执行摘要

本任务在不修改 V5.1 的前提下，完成了基线校验、安全数据清洗、独立盲测集构建、从 V5.1 LoRA 权重开始的 V5.2 增量训练、两套盲测评估、逐文件校验、Git 版本记录和独立归档。

流水线的可复现性与版本保护要求均已通过：原始备份 SHA-256 与基准完全一致；V5.1 的 23 个文件在训练前后及最终交付复核中均保持一致；清洗集、盲测集和 V5.1 训练数据的精确及近似交集均为 0；训练使用 `--peft_path` 加载 V5.1，写入独立 V5.2 目录，未使用 `resume_from_checkpoint`；27 个训练损失点显示下降。

模型发布门禁未通过。首次盲测的任一广义安全违规率为 12/18（66.67%），训练模板对齐盲测为 15/18（83.33%），两套盲测的有效结构、`action` 和 `action+stage` 指标均为 0/18。首次盲测第 17 条还出现药材组合、剂量调整建议及错误医学知识。因此 V5.2 只能归档用于分析，**不得上线、不得替换 V5.1、不得让 API 指向该目录**。

关键数字如下：

- 原始神农数据：112,565 条。
- 新安全保留池：91 条，其中训练清洗集 73 条、盲测集 18 条。
- 剔除明细：112,474 条；原因支持多标签统计。
- 全量保留数据安全复核：91/91 通过；主代理独立复核 91/91 通过。
- V5.1 文件保护：23/23 哈希一致。
- V5.2 权重：41/41 哈希一致。
- 训练与评估交付物：77/77 哈希一致。
- Git 提交：`f28d313a3c087d93bedd21c6fc62ea7a46798a6e`。

## 2. 数据统计对照表

| 数据集 | 条数 | 用途 | 关键校验 |
|---|---:|---|---|
| 神农原始数据 | 112,565 | 清洗来源 | SHA-256 `9021c29d476a21bd316eab37410b9134b2bdc464dcb9d215fb58e23269b9a2ab` |
| 新 V5.2 清洗集 | 73 | V5.2 唯一训练输入来源 | SHA-256 `0b7c23a20b477fb8c82f18fbc60699ef1fb7d1d99374fd8fc75ca263888a6361` |
| 新 V5.2 盲测集 | 18 | 训练后独立评估 | SHA-256 `c8a144eebea26d4f0c1f9c3719151f8a39de6d45859aa2fa17c4f5ce734a50ad` |
| 新 V5.2 剔除集 | 112,474 | 审计与原因追踪 | SHA-256 `157217e3746fb4c764f934058eb275191b745523f19b053c3ca01bbe1428cc08` |
| V5.1 清洗集 | 1,200 | 去重参照 | SHA-256 `09ea9639e623cbb22807b2dcb04e364abf9709f2cd1f3324b6541acbe432d66b` |
| V5.1 混合训练集 | 2,730 | 去重参照 | SHA-256 `05383f813094f3d24c67bb76c8e6d518124905ed03fa297d222830557bdef0a6` |

新保留池由 81 条明确安全拒绝和 10 条非医疗事实型澄清组成。按固定随机种子 `20260910` 分层拆分为 73 条清洗训练数据和 18 条盲测数据。训练侧另建只含 `clean_train.jsonl` 的隔离输入目录，该文件与清洗集哈希完全一致；盲测集未进入训练目录。

去重采用文本级规范化精确比较与字符 3-gram Jaccard 近似比较。新数据内部查询阈值为 0.95、问答对阈值为 0.90；与 V5.1 比较时两项阈值均为 0.90。清洗集与盲测集、V5.1 清洗集、V5.1 混合训练集之间的精确查询、精确问答对、近似查询和近似问答对交集均为 0。

## 3. 过滤剔除明细与复核

过滤规则遵循保守口径：没有逐条可审计权威来源的医学事实断言按未核验知识剔除；药材、方剂、药物、剂量、调剂、治疗方案及可操作医疗建议剔除；暴露推理链的回答剔除；医疗请求只有明确安全拒绝才可保留；非医疗事实型内容只允许澄清提问。

| 剔除原因代码 | 命中条数 |
|---|---:|
| `direct_prescription_or_dose` | 101,049 |
| `medical_request_without_explicit_safety_refusal` | 4,976 |
| `no_safe_refusal_or_clarification` | 431 |
| `query_length_out_of_range` | 22 |
| `reasoning_chain_or_hidden_cot` | 4,110 |
| `response_length_out_of_range` | 1,039 |
| `text_duplicate` | 1 |
| `unverified_or_conflicting_knowledge` | 1,905 |
| `v5_1_exact_overlap` | 2 |

同一记录可以命中多个原因，因此上表合计数大于剔除记录数。逐条剔除明细保存在 `shennong_rejections_v5_2.jsonl`。

复核采用三层证据：数据代理对全部 91 条保留记录重新运行安全谓词，91/91 通过；固定种子分层抽取 20 条保留样本，20/20 通过；分层抽取 24 条剔除样本，24/24 均有明确原因。主代理随后独立检查全部 91 条保留记录，没有发现错误/未核验知识、直接开方、剂量、治疗承诺或可操作医疗建议。按本任务定义的抽样复核口径，错误知识与直接开方内容剔除率为 **100%**。

这一结论表示规则和复核样本内未发现漏保留内容，不代表 112,565 条原始数据已由持证中医专家逐条完成医学真实性鉴定。

## 4. V5.2 训练参数、收敛证据与盲测结果

| 项目 | 实际值 |
|---|---|
| 基础模型 | `/home/cyh/Medical_Qwen/models/Qwen2.5-1.5B-Instruct` |
| 起始 LoRA | `/home/cyh/Medical_Qwen/output/tcm-qwen-1.5b-v5-1`，通过 `--peft_path` 加载 |
| 输出目录 | `/home/cyh/Medical_Qwen/output/tcm-qwen-1.5b-v5-2` |
| 训练/内部验证划分 | 69 / 4 |
| batch / 梯度累积 | 1 / 8 |
| epoch / 优化步 | 3 / 27（完成 27/27） |
| 学习率 | `1e-5` |
| 日志/评估/保存间隔 | 1 / 6 / 6 步 |
| 训练耗时 | 100.6643 秒 |
| 训练总损失 | 1.414698 |
| 最终内部 eval loss / perplexity | 1.2056399584 / 3.3388951484 |

损失曲线含 27 个点：第 1 步为 2.1462，第 27 步为 1.0561，最低值为 0.9769（第 23 步）；前 5 点中位数为 2.1462，后 5 点中位数为 1.0655，下降 50.3541%。这证明训练目标发生收敛，但不构成安全性证明。

| 指标 | 首次盲测：默认 Qwen 模板 | 训练模板对齐盲测 |
|---|---:|---:|
| BLEU-1 / BLEU-4 | 0.1792 / 0.0771 | 0.3766 / 0.2471 |
| ROUGE-L | 0.0683 | 0.5607 |
| 有效结构输出 | 0/18（0.00%） | 0/18（0.00%） |
| `action` 准确 | 0/18（0.00%） | 0/18（0.00%） |
| `action+stage` 准确 | 0/18（0.00%） | 0/18（0.00%） |
| 直接开方/剂量违规 | 1/18（5.56%） | 0/18（0.00%） |
| 可操作医疗建议 | 4/18（22.22%） | 5/18（27.78%） |
| 无依据医学断言 | 10/18（55.56%） | 8/18（44.44%） |
| 医疗请求未明确拒绝 | 2/18（11.11%） | 12/18（66.67%） |
| 任一广义安全违规 | 12/18（66.67%） | 15/18（83.33%） |

首次盲测和模板对齐盲测均对 18/18 条预测做了广义安全人工分类。旧的窄短语命中评分曾给出直接开方为 0，该指标遗漏了药材组合、剂量建议和事实错误，已被逐条广义审计替代。没有为了改善结果而重新训练或改写已冻结盲测结果。

## 5. 归档与校验信息

原始完整备份：

- 路径：`C:\Users\cyh\Desktop\小医ai问诊模型\Medical_Qwen_backup_2026-09-09.tar`
- 大小：5,017,067,520 字节。
- 基准 SHA-256：`F88D537122D0BB07965186667F9B403ADFD8A667B42EA217D86AF19336583556`。
- 本次重算 SHA-256：`F88D537122D0BB07965186667F9B403ADFD8A667B42EA217D86AF19336583556`，一致。

V5.2 实验归档：

- 路径：`C:\Users\cyh\Desktop\小医ai问诊模型\Medical_Qwen_V5.2_artifacts_2026-09-10_NOT_PUBLISHABLE.tar`
- 大小：417,792,000 字节。
- SHA-256：`1311F27E6BBDC4350D1D2580842A45FE3F284E0E9AAA65FEEF89D4B857B5A9BB`。
- 校验文件：`C:\Users\cyh\Desktop\小医ai问诊模型\Medical_Qwen_V5.2_artifacts_2026-09-10_NOT_PUBLISHABLE.sha256.txt`。
- 归档范围：`v5_2_pipeline/`、`artifacts/v5_2_pipeline/`、`output/tcm-qwen-1.5b-v5-2/`。本总报告单独交付，并有同名 SHA-256 校验文件。

逐项清单：

| 清单 | 项数/状态 | 清单 SHA-256 |
|---|---:|---|
| `v5-1_baseline.sha256` | 23/23 通过 | `c14fe4703b6472abeb4b29cbb0c4fba73d24b4f01011a00f23078e470dd69c8a` |
| `data/sha256sums_v5_2.txt` | 9/9 通过 | 各数据哈希写入清单并由主代理重算通过 |
| `training/v5-2_weights.sha256` | 41/41 通过 | `4bfadf15e6758253921d62c81fba50f7ecd45a3c6273f746d5b795d5aa5e251e` |
| `training/training_evaluation_delivery.sha256` | 77/77 通过 | `77ea9696fad374c082733a0eb3bd26acd34b7ba0b73f08d9d159d9610a2d180e` |

最终复核日志位于 `/home/cyh/Medical_Qwen/artifacts/v5_2_pipeline/final_review/`。代码和配置共 12 个文件已提交至 Git；提交前通过 Python 编译、Shell 语法和脚本副本哈希检查。仓库中原有 API、RAG、模型、数据及其他未跟踪文件没有进入本次提交。

## 6. 任务分解与完成情况

| 阶段 | 责任方 | 结果 | 审查/修订记录 |
|---|---|---|---|
| 0 基线校验 | 主代理 | 通过 | 备份哈希一致；生成 V5.1 23 文件基线清单 |
| 1 任务下发 | 主代理 | 完成 | Luna 接收数据侧简报；Terra 接收训练与评估侧简报；共同附带术语、版本保护与安全约束 |
| 2 数据清洗与盲测 | Luna | 一次打回后通过 | 首版 264 条候选未满足保守安全口径；主代理按允许次数打回一次；修订后保留 91 条并通过 91/91 全量复核 |
| 3 增量训练 | Terra | 通过 | 仅用 73 条清洗集；从 V5.1 `peft_path` 训练至独立 V5.2；未使用恢复训练参数 |
| 4 评估与复核 | Terra、主代理 | 执行完成，发布门禁失败 | 初始窄词评分不能反映真实风险；追加两套 18/18 广义逐条审计；未重训，最终如实判为不可发布 |
| 5 归档与版本记录 | 主代理 | 通过 | 新产物独立归档并生成 SHA-256；代码/配置提交 Git `f28d313a3c087d93bedd21c6fc62ea7a46798a6e` |
| 6 审查与报告 | 主代理 | 完成 | 机械完整性要求通过；模型安全和结构质量未通过，最终状态为 NOT_PUBLISHABLE |

## 7. 遗留问题与待确认事项

1. 当前 V5.2 在两套盲测中均存在严重安全或结构问题，不能部署。后续应新建 V5.3 方案，保留当前 V5.2、训练日志和盲测结果不变，重新设计数据和输出约束后再用新的独立盲测验收。
2. 当前 API 仍在 `tcm_chat_v5.py` 中指向 `tcm-qwen-1.5b-v5-1`，本任务没有修改 API、网关或 RAG。该状态符合“V5.2 不可发布”的结论。
3. V5.2 数据包含 `refuse`/安全拒绝标签，而当前问诊提示词的主要结构动作只接受 `ask` 与 `summarize`；盲测结构 0/18 说明训练格式与应用协议需要在下一版本统一设计和验证。
4. 73 条训练数据规模很小，且主要由安全拒绝组成。损失下降不能证明对真实问诊任务具有稳定的临床安全性或有效性。
5. 过滤器采用极保守规则，将没有逐条权威来源的医学陈述剔除。若下一版本需要保留正确知识，应先建立可审计的权威来源、专家复核记录和数据版本链，再训练新的版本。

## 随报告交付的产物清单

- 数据：`/home/cyh/Medical_Qwen/artifacts/v5_2_pipeline/data/shennong_clean_v5_2.jsonl`、`shennong_heldout_v5_2.jsonl`、`shennong_rejections_v5_2.jsonl`。
- 规则与统计：`filter_rules_v5_2.md`、`statistics_v5_2.json`、`dedup_report_v5_2.json`、`overlap_report_v5_2.json`、`retention_verification_v5_2.json`、`sha256sums_v5_2.txt`。
- 版本保护：`/home/cyh/Medical_Qwen/artifacts/v5_2_pipeline/v5-1_baseline.sha256`、`stage0_verification.json` 和 `final_review/` 下复核日志。
- 训练：`/home/cyh/Medical_Qwen/output/tcm-qwen-1.5b-v5-2/`、`/home/cyh/Medical_Qwen/artifacts/v5_2_pipeline/training/train_v5_2.log`、`train_curve.csv`、`eval_curve.csv`、`training_curves.png`、`training_convergence_metrics.json`。
- 评估：`/home/cyh/Medical_Qwen/artifacts/v5_2_pipeline/training/v5_2_training_evaluation_report.md`、`blind_eval/`、`blind_eval_training_template/`。
- 校验：`/home/cyh/Medical_Qwen/artifacts/v5_2_pipeline/training/v5-2_weights.sha256`、`training_evaluation_delivery.sha256`。
- 代码与配置：`/home/cyh/Medical_Qwen/v5_2_pipeline/`，Git 提交 `f28d313a3c087d93bedd21c6fc62ea7a46798a6e`。
- Windows 归档与校验文件：见本报告第 5 节。
