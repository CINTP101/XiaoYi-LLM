# 将 Medical_Qwen 放到自己的 GitHub

已检查本地项目：`/home/cyh/Medical_Qwen` 是 Git 仓库，`master` 比原仓库 `origin/master` 多 5 个提交。`origin` 指向 `https://github.com/scuterGuoyulong/Medical_Qwen.git`，不能把它当作自己的仓库地址。下面默认创建**私有、代码为主**的新仓库，保留上游 `origin`。

## 1. 在 GitHub 创建空仓库

登录 GitHub，打开 <https://github.com/new>，仓库名 `XiaoYi-LLM`，可见性选 **Private**。不要勾选自动创建 README、`.gitignore` 或 License，因为本地仓库已经有这些文件。复制已提供的 HTTPS 地址 `https://github.com/CINTP101/XiaoYi-LLM.git`。

## 2. 在 WSL 中单独提交这次 API 代码

逐行运行；下面只暂存当前使用的三个 RAG 代码文件，不包括历史快照和 RAG 索引。

```bash
cd ~/Medical_Qwen
git switch codex/github-publish
git add .gitignore requirements-api.txt tcm_api.py tcm_v54_service.py \
  tcm_gateway.py tcm_consultation_state.py tcm_chat_v5.py \
  safety_router.py intent_router.py docs/API_V54_APP.md \
  examples/app_api_client.py tests/test_api_v54.py \
  tests/smoke_api_v54.py rag/embedder.py rag/field_reranker.py \
  rag/retriever.py GITHUB_PUBLISH_GUIDE.md
git diff --cached --stat
git diff --cached --name-only
git diff --cached --check
```

检查文件列表，只应包含代码、文档和配置。确认没有模型文件、训练数据、评估数据、患者内容、密钥或归档后，再提交：

```bash
git commit -m "Add V5.4 R1 App API and GitHub publishing guide"
```

已有的 `evaluate_sft_qwen.py`、`requirements.txt` 工作区修改没有放进这次暂存清单，也不会被这次提交包含。以后审查后可以另行提交。

## 3. 加上自己的远端并上传

用户已提供目标仓库地址。执行推送前，先确认它已创建且当前登录账号可以访问。

```bash
git remote add personal https://github.com/CINTP101/XiaoYi-LLM.git
git remote -v
git push -u personal HEAD:main
```

这里的 `personal` 是你自己的远端；`origin` 仍指向原作者仓库。不要运行 `git push origin master`。如果命令行要求认证，使用浏览器登录 Git Credential Manager / GitHub CLI，或输入 GitHub Personal Access Token；Git 操作不能用 GitHub 登录密码。

完成后，打开 `https://github.com/CINTP101/XiaoYi-LLM` 检查文件和仓库的 **Private** 状态。

## 哪些内容不进入普通 Git 仓库

本机的 `.gitignore` 已排除 `models/`、`output/`、`artifacts/`、新增的 `data/` 文件、`rag/index/`、`.runtime/` 和模型、归档、密钥文件。原仓库已经跟踪的 `data/` 文件仍然会随历史提交上传；忽略规则不会从既有 Git 历史中删除文件。

当前项目本地 `models/` 约 3.1 GB、`output/` 约 4.6 GB。GitHub 普通 Git 拒绝超过 100 MiB 的单个文件；大文件可使用 Git LFS，但会占用单独的存储和下载额度。完整备份（约 5 GB）以及模型、数据更适合放在有权限控制的对象存储，保留现有 SHA-256 文件；代码仓库中写清恢复方法与所需文件清单。

如确实要在 GitHub 存储模型权重，应先确认基础模型、神农数据及其他资料的再分发权限，再单独设计 Git LFS 和预算，不要直接 `git add .`。
