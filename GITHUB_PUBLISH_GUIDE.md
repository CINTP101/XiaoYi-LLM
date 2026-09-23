# XiaoYi-LLM 发布说明

目标仓库：<https://github.com/CINTP101/XiaoYi-LLM>。本地 `origin` 仍指向上游 `scuterGuoyulong/Medical_Qwen`；`personal` 指向自己的仓库。

本次从本地 `Medical_Qwen` 创建 `codex/github-publish`，提交 App API 代码、文档和必要的 RAG 源码。随后在隔离工作树 `codex/github-publish-merged` 合并目标仓库原有的 README、V5.2/V5.3 报告和 SHA-256 文件。两个仓库原本没有共同 Git 历史，合并保留了目标仓库的报告和校验文件，并采用本地较完整的 README。V5.3 报告采用与已有 SHA-256 文件一致的版本。

发布分支推送到目标仓库的 `main`：

```bash
git push -u personal codex/github-publish-merged:main
```

如果在 WSL 中提示缺少 GitHub 凭据，可以使用已登录的 Git for Windows / Git Credential Manager 推送。此机器的 Windows 系统代理是 `127.0.0.1:7892`；代理端口变化时应按实际设置调整。不要将访问令牌写进命令、仓库或文档。

## 仓库包含与不包含的内容

本次 API 提交只加入了 17 个明确审查过的代码、文档和配置文件，没有加入本机模型权重、训练归档、RAG 索引、会话密钥或新生成的数据。原上游仓库已跟踪的少量 `data/` 示例文件仍随原有 Git 历史保留；`.gitignore` 不能移除既有历史。原仓库中未提交的脚本和工作区修改也仍保留在本地，没有并入这次发布。

本机 `models/` 约 3.1 GB、`output/` 约 4.6 GB。完整运行还需要另行准备基础模型、LoRA 权重、BGE 模型与 RAG 索引；代码仓库本身不等于完整可运行的模型备份。大文件仍应依照现有归档与 SHA-256 流程保存。若要通过 GitHub 分发权重或神农数据，需先确认再分发权限，并单独规划 Git LFS 存储和费用。

App 接入说明见 [docs/API_V54_APP.md](docs/API_V54_APP.md)。
