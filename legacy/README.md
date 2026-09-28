# 历史训练与演示脚本

这里集中放置从上游继承的通用 PT、SFT、DPO、PPO、量化和演示示例，以及对应配置和 Notebook。部分脚本保留了上游机器的绝对路径，使用前需要自行替换模型与数据路径；它们不是小医 V5.4-R1 的在线服务入口。

请从**仓库根目录**运行这些脚本，以便相对数据路径和配置路径正常解析。例如：

```bash
bash legacy/run_pt.sh
python -m legacy.inference --help
python -m legacy.merge_peft_adapter --help
```

小医当前的 API 入口仍在根目录：`python tcm_api.py --host 127.0.0.1 --port 8008`。正式版本训练与评估流程在 `v5_2_pipeline/`、`v5_3_pipeline/`、`v5_4_pipeline/`，版本报告在 `reports/`。
