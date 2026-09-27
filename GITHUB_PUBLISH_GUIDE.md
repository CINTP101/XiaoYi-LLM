# 小医仓库：代码怎么更新，模型文件放哪里

小医的代码在 <https://github.com/CINTP101/XiaoYi-LLM>。这里放的是源码、使用说明和部分历史报告；模型权重、训练归档、BGE 模型、RAG 索引和服务密钥没有放进普通 Git 仓库。所以，在 GitHub 上看到代码，不代表已经拿到了可直接运行的完整模型。

## 改完文件后怎么上传

先进入你正在修改的那份本地仓库，运行 `git remote -v`。确认要推送的远端确实是 `CINTP101/XiaoYi-LLM`，再操作。不同电脑或工作树可能把它叫作 `origin` 或 `personal`，不要只凭远端名字判断。

这份 WSL 发布工作树使用 `personal`。例如只修改了 README，可以这样做：

```bash
cd ~/Medical_Qwen_publish
git status
git add README.md
git commit -m "Update XiaoYi README"
git push personal HEAD:main
```

改的是其他文件，就把 `README.md` 换成实际路径。推送前看一眼 `git status`，尽量逐个添加要提交的文件。保存文件并不会自动同步到 GitHub。

## 哪些文件要单独保存

`.gitignore` 会挡住常见的模型目录、生成数据、密钥和 `*.tar` 归档。但忽略规则**不会删除已经进入 Git 历史的文件**：上游原本跟踪的一些小型示例数据，仍会跟随代码历史保留。

完整运行小医，需要另行准备 Qwen 基础模型、V5.4-R1 LoRA 权重、冻结清单与适配器；启用知识检索还需要 BGE 模型和 RAG 索引。请继续用既有的“归档 + SHA-256”方式保存大文件。若打算公开分发权重或神农数据，先核对再分发权限和存储成本，别直接运行 `git add .`。

App 接入方式见 [小医 API 说明](docs/API_V54_APP.md)。V5.2/V5.3 的报告和校验文件属于历史发布记录，保留原文，便于日后核对。
