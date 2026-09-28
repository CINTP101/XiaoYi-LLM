# 小医（XiaoYi）大语言模型

中文 | [English](README_EN.md) | [App 接入文档](docs/API_V54_APP.md)

小医是面向**中医问诊信息采集**的实验性大语言模型项目。它基于 Qwen2.5-1.5B-Instruct 进行 LoRA 微调，结合安全分流、会话状态管理和可选的 BGE/RAG 检索，帮助 App 追问症状、整理用户陈述。小医不提供个体诊断、辨证结论或处方。

本仓库公开代码、接口说明和部分训练记录。**Qwen 基础模型、V5.4-R1 权重、BGE 模型及现有 RAG 索引尚未在公开仓库提供**；仅克隆代码不能让完整服务就绪。文件准备方式见 [仓库与发布说明](GITHUB_PUBLISH_GUIDE.md)。

## 概述

| 项目 | 当前情况 |
|---|---|
| 对外名称 | 小医（XiaoYi） |
| 模型版本 | V5.4-R1 |
| 基础模型 | [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，约 1.5B 参数 |
| 适配方式 | LoRA 参数高效微调 + Candidate H v2 运行时适配器 |
| 接口 | FastAPI，`POST /v1/tcm/process` |
| 知识检索 | 可选 BGE + RAG；需要另行准备有权使用的索引 |
| 适用范围 | 问诊交互、信息整理与研究演示；尚无临床有效性验证 |

一轮请求先经过安全分流和意图识别。急症风险和转诊情形由专门路径处理；知识问题可进入检索；普通问诊由模型生成，再由运行时适配器整理成结构化追问或事实摘要。App 获取 `action`、`message` 和签名会话状态，不直接接收模型原始生成。

## 核心能力

- **症状采集与追问**：围绕用户已经提供的信息继续提问，避免把猜测写成事实。
- **事实摘要**：整理本次会话中由用户陈述的症状与背景。
- **安全分流**：对急症风险或需要线下就医的情况给出相应提示。
- **知识检索**：在单独配置 BGE 模型与 RAG 索引后，处理部分纯知识问题。
- **App 接入**：通过 HTTP 接口返回可显示的消息与下一轮需要回传的状态。

这些能力描述的是当前接口设计，并不表示已经通过临床验证。完整行为和边界见 [App 接入文档](docs/API_V54_APP.md)。

## 使用示例

启动服务后，向小医发送一轮消息：

```bash
curl http://127.0.0.1:8008/v1/tcm/process \
  -H 'Content-Type: application/json' \
  -d '{"text":"我最近口干，而且晚上会出汗。","state":null,"new_session":true,"request_id":"demo-001"}'
```

把响应中的 `result.message` 展示给用户；下一轮将 `result.state` 原样放入请求的 `state`，并设置 `new_session=false`。新建问诊时设置 `new_session=true`。急症、转诊和知识路径可能不调用生成模型，客户端应依据 `result.action` 处理结果。

## 模型与数据

小医的技术版本名保留为 V5.4-R1，以便和训练日志、冻结清单及历史报告对应。基础模型来自 Qwen；LoRA 权重与 Candidate H v2 适配器需配套使用。训练与清洗流程保留在 `v5_2_pipeline/`、`v5_3_pipeline/` 和 `v5_4_pipeline/`；阶段统计可查阅 [V5.2 报告](V5_2_FINAL_REPORT.md)与 [V5.3 报告](V5_3_FINAL_REPORT.md)。

完整神农原始数据和训练归档不随本仓库发布。现有 RAG 元数据含国家标准、药典及教材来源的正文片段，公开再分发权限尚未核清，因此索引也未上传。代码开放不等于训练数据、模型权重或知识文本自动获得相同许可。

## 下载与部署

先在本机准备 `tcm_llm` 环境，并将 Qwen 基础模型、V5.4-R1 权重、冻结清单和适配器放入项目要求的路径。缺少这些文件时，`GET /readyz` 不会报告模型就绪。然后运行：

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

交互式接口文档位于 <http://127.0.0.1:8008/docs>。`/healthz` 只说明进程仍在；`/readyz` 才检查整套模型是否可用。云端和手机接入还需要 HTTPS、鉴权及业务后端代理，具体见 [App 接入文档](docs/API_V54_APP.md)。

## 训练与评估

仓库保留数据清洗、训练、评估流程代码及部分阶段报告，用于追溯技术版本。历史指标应以对应报告和测试条件为准；目前没有足以支持临床准确率或疗效宣传的验证结果。若研究者复现实验，请先核对训练数据来源、医学内容和使用许可，不要把未过滤的原始语料直接用于训练。

## 仓库导航

| 路径 | 内容 |
|---|---|
| `tcm_api.py`、`tcm_v54_service.py` | HTTP 接口与 V5.4-R1 模型服务 |
| `tcm_gateway.py`、`tcm_consultation_state.py` | 请求路由与会话状态 |
| `rag/` | 知识检索代码；现有索引未公开发布 |
| `docs/API_V54_APP.md` | App 请求格式、错误码和部署边界 |
| `examples/app_api_client.py` | Python 客户端示例 |
| `v5_2_pipeline/`、`v5_3_pipeline/`、`v5_4_pipeline/` | 数据、训练与评估流程 |

## 致谢

小医在 [Medical_Qwen](https://github.com/scuterGuoyulong/Medical_Qwen) 的基础上扩展，仓库也保留了部分来自 [MedicalGPT](https://github.com/shibing624/MedicalGPT) 的通用训练示例。基础模型来自 [Qwen](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，检索组件采用 [BGE](https://huggingface.co/BAAI/bge-small-zh-v1.5)。这些上游项目的名称和许可应与小医自身版本区分。

欢迎提交 issue 或 pull request。涉及医学知识时请附可核查来源，不要提交患者隐私、密钥或未经授权的大文件；详见 [贡献说明](CONTRIBUTING.md)。

## 声明

小医用于信息采集和研究演示，回复可能不完整或错误，不能代替执业医师的判断，也不能作为诊断或用药依据。遇到急症请及时寻求线下医疗帮助。详见 [免责声明](DISCLAIMER)。

## 许可

**小医项目中由维护者有权授权的原创代码和文档采用 [MIT 许可](LICENSE)。** 仓库沿用的 Medical_Qwen、MedicalGPT 代码及原版权声明仍受其原有 Apache-2.0 条款约束；第三方模型、示例数据、训练语料和知识文本不因仓库根目录的 MIT 许可而改为 MIT。详见 [第三方来源与许可说明](THIRD_PARTY_NOTICES.md)及保留的 [Apache-2.0 许可全文](LICENSES/Apache-2.0.txt)。
