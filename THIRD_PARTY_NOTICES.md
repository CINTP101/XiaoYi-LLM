# 第三方来源与许可范围

仓库根目录的 [MIT 许可](LICENSE)适用于小医维护者有权授权的原创代码和文档。它不改变下列上游作品、模型或数据的权利归属。使用者应同时遵守各来源的许可与版权声明。

| 来源 | 仓库中的关系 | 许可与处理 |
|---|---|---|
| [Medical_Qwen](https://github.com/scuterGuoyulong/Medical_Qwen) | 本仓库的上游项目，保留部分训练与评估代码、文档及示例数据 | 原仓库采用 Apache-2.0；原许可全文保存在 [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)，原文件内版权声明继续保留 |
| [MedicalGPT](https://github.com/shibing624/MedicalGPT) | 部分通用训练示例及文档的更早来源 | 其代码按 Apache-2.0 发布；原作者标记应保留。其模型和数据可能另有用途限制，不纳入小医的 MIT 授权 |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 小医使用的基础模型；文件未在本公开仓库提供 | 模型卡标注 Apache-2.0，使用时以模型发布页及附带文件为准 |
| [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) | 可选知识检索模型；文件未在本公开仓库提供 | 模型卡标注 MIT，使用时以模型发布页及附带文件为准 |
| `data/`、`role_play_data/` 等历史示例 | 部分由上游仓库带入 | 不能仅凭本仓库的 MIT 文件认定这些数据可再分发、训练或商业使用；逐项核对来源 |
| 本地训练权重与 RAG 索引 | 当前未在公开仓库提供 | 模型训练数据及 RAG 正文的公开再分发权利需单独核实 |

本仓库曾直接继承上游的 `DISCLAIMER`。原文另存于 [LICENSES/UPSTREAM_DISCLAIMER.txt](LICENSES/UPSTREAM_DISCLAIMER.txt)，用于保留来源记录；它不应被误读为小医原创 MIT 代码的新许可条款。上游对其模型和数据的声明仍需分别核查。

以上清单说明已知来源，并非逐文件的版权鉴定。特别是混合或改写过的上游文件，仍须保留适用的 Apache-2.0 许可、原作者声明及修改记录。若要公开发布权重、训练数据或 RAG 正文，应先完成对应素材的授权核对。
