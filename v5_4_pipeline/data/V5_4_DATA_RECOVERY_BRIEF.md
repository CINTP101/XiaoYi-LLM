# V5.4 数据恢复子任务简报

## 目标

在不使用第 1 批数据文本、不读取 V5.3 最终盲测文本的前提下，重新构建自然、紧凑、适合手机端真实输入分布的 V5.4 数据集。输出到独立的 `artifacts/v5_4_pipeline/data_v2/`，不得覆盖已拒收的 `data/`。

## 固定规模与协议

- train：1,600（ask/initial 1,200；summarize/summary 400）。
- protocol_dev：240（ask/initial 180；summarize/summary 60）。
- heldout：240（ask/initial 180；summarize/summary 60）。
- 每行仅含一个 human/gpt 对，目标严格使用冻结的 V5.3 扁平 JSON 合同。
- ask 必须有 3 项 questions，第一项明确拒绝诊断、开方、药材、药物和剂量决定，后两项为安全澄清问题。
- summary 必须使用冻结中性辨证短语；key_findings 非空、唯一且均为 human 原文中的完整子串。

## 自然性硬门禁

- human 文本中禁止出现随机 hash、UUID、样本编号、nonce、V54 标识、用于区分 split/category 的固定尾句。
- 禁止加入与问诊意图无关的颜色、地图、灯塔、邮票、折纸、音乐、天气、桌面物件等填充信息。
- 禁止出现“附加记录：”“元信息：”“附加观察：”等审计型标记。
- train 与 protocol_dev 的每条文本必须可以被解释为真实用户会在手机问诊中直接输入的一条消息。
- train 和 protocol_dev 字符长度中位数必须在 20–120，P95 不超过 180，超过 240 字的比例必须为 0。
- 单一模板骨架占比不得超过 5%；不得靠同义词机械替换或无关尾句制造差异。
- Luna 对 train 分层抽检 160 条、protocol_dev 全量 240 条；自然性、安全性和目标一致性必须 100% 通过并保存逐条记录。heldout 仅程序化扫描，不在报告中输出样例。

## 去重与隔离硬门禁

- 三个 split 原文、Unicode NFKC 后规范化文本的交集均为 0。
- 只对跨 split 进行近重复泄露审计：字符 3-gram Jaccard 必须 <0.90，本地 BGE 余弦必须 <0.985。
- split 内允许相同安全意图，但原文和规范化精确重复仍必须为 0。
- 所有 human 文本 SHA-256 与 V5.3 最终盲测 180 条 SHA denylist 的交集为 0；禁止打开历史盲测文本、预测或答案。
- 训练、开发、heldout 的来源场景和模板族应按组隔离，避免同一近似句只替换症状后跨 split 分配。

## 交付

- `train_v5_4_v2.jsonl`
- `protocol_dev_v5_4_v2.jsonl`
- `heldout_v5_4_v2.jsonl`
- 可复现构造器 `v5_4_pipeline/data/build_v5_4_data_v2.py`
- 来源追踪、严格合同扫描、安全扫描、prompt manifest、跨 split 去重报告、自然性逐条审计、统计报告、SHA-256 清单。

所有门禁通过后停止，等待主代理和独立审计验收；不得启动训练或访问任何模型权重。
