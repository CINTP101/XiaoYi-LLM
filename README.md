# 小医大语言模型

小医是一个还在持续完善的中医问诊模型项目。你可以用它做症状信息采集、追问和陈述摘要，也可以通过 API 把这些能力接进 App。它不会替你做个体诊断，也不会直接开处方。

仓库放的是**代码和文档**。模型权重、训练归档、BGE 模型和 RAG 索引需要另外准备；只克隆这个仓库，服务还不能完整启动。

## 技术概览

小医当前以 **Qwen2.5-1.5B-Instruct** 为基础，使用 **LoRA 参数高效微调**适配中医问诊场景；对外提供的模型版本是 **V5.4-R1**。它的设计重点是把自由文本变成可控的问诊交互，而不是让模型独自决定诊断和治疗。

一轮请求会先经过安全分流和意图识别：急症或转诊场景走专门的提示路径；知识问题可走 BGE/RAG 检索；普通问诊由模型生成后，再经过 Candidate H v2 适配器整理为追问或事实摘要。App 收到的是带有 `action`、`message` 和签名会话状态的结构化结果，而不是直接暴露的原始生成文本。

服务层使用 FastAPI 提供接口，并在启动时核对冻结权重和适配器；`/readyz` 用来确认整套模型是否就绪。数据清洗、训练与评估的流程代码及部分阶段报告保留在仓库中，方便追溯。**这些工程措施不等于临床有效性验证**，小医目前仍应按信息采集与研究演示工具使用。

## 从哪里开始

- 想把小医接进 App：看 [API 接入说明](docs/API_V54_APP.md)。
- 想知道代码和模型文件分别放在哪里：看 [仓库与发布说明](GITHUB_PUBLISH_GUIDE.md)。
- 想了解数据清洗和训练过程：看仓库中的 V5.2/V5.3 报告及 `v5_2_pipeline/`、`v5_3_pipeline/`、`v5_4_pipeline/`。
- 想试一个最简单的客户端：看 [Python 调用示例](examples/app_api_client.py)。

## 小医现在能做什么

- 根据用户描述继续追问，并把已收集到的事实整理成摘要。
- 对急症风险给出分流提示，对需要线下处理的情况给出就医建议。
- 对部分纯知识问题走现有检索流程。知识检索依赖另行准备的 BGE 模型和 RAG 索引。
- 通过 `POST /v1/tcm/process` 与 App 交换消息和会话状态。

这些结果用于信息整理和交互演示，不能当作医疗诊断或治疗方案。尤其不要把模型回复当作处方使用。

## 版本怎么理解

**小医**是对外使用的模型名称。当前 API 对应的模型版本是 **V5.4-R1**，底座是 **Qwen2.5-1.5B-Instruct**，再加载本项目的 LoRA 权重和运行时适配器。接口返回中的版本号、权重目录和历史报告文件名保留技术名称，方便核对和复现。

本仓库基于 [Medical_Qwen](https://github.com/scuterGuoyulong/Medical_Qwen) 扩展；其中也保留了部分来自 [MedicalGPT](https://github.com/shibing624/MedicalGPT) 的通用训练示例。它们是来源和技术参考，不是小医当前版本的别名。

## 在本机试运行

先准备好 `tcm_llm` 环境、基础模型、V5.4-R1 权重、冻结清单和适配器。缺少这些文件时，`/readyz` 不会显示模型就绪。进入项目目录后运行：

```bash
conda activate tcm_llm
pip install -r requirements-api.txt
python tcm_api.py --host 127.0.0.1 --port 8008
```

另开终端检查：

```bash
curl http://127.0.0.1:8008/healthz
curl http://127.0.0.1:8008/readyz
```

接口文档在 <http://127.0.0.1:8008/docs>。向小医发送一轮消息：

```bash
curl http://127.0.0.1:8008/v1/tcm/process \
  -H 'Content-Type: application/json' \
  -d '{"text":"我这阵子觉得嘴巴发干。","state":null,"new_session":true,"request_id":"demo-001"}'
```

第一次请求时，`state` 用 `null`；下一轮把响应里的 `result.state` 原样传回。更多字段、鉴权和手机接入方式见 [API 接入说明](docs/API_V54_APP.md)。

## 仓库里有什么

| 位置 | 用途 |
|---|---|
| `tcm_api.py`、`tcm_v54_service.py` | 小医当前的 HTTP 接口和模型服务 |
| `tcm_gateway.py`、`tcm_consultation_state.py` | 请求路由和会话状态 |
| `rag/` | 知识检索代码；索引文件不在仓库里 |
| `docs/API_V54_APP.md` | App 接入、错误码和部署边界 |
| `v5_2_pipeline/`、`v5_3_pipeline/`、`v5_4_pipeline/` | 历史数据与训练流程 |
| `V5_2_FINAL_REPORT.md`、`V5_3_FINAL_REPORT.md` | 已归档的阶段报告 |

历史报告和校验文件保留发布时的原文与文件名。通用训练教程见 `docs/`，其中部分内容沿用上游示例；使用前请核对是否适合小医当前版本。

## 想一起改进？

欢迎提 issue 或 pull request。修改接口时，请同时更新文档和相关测试；涉及医学知识时，请提供可核查的来源，不要提交患者隐私、密钥或模型大文件。详见 [贡献说明](CONTRIBUTING.md)。

代码许可见 [LICENSE](LICENSE)。
