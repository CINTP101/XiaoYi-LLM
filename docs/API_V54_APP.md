# V5.4 R1 App API 接入说明

接口地址：`POST /v1/tcm/process`。请求/响应兼容 XiaoMedInsight 当前的 `TcmGatewayClient`。
API 版本 V6.21.0；模型版本 V5.4-R1；运行时同时加载冻结的 LoRA 权重和 Candidate H v2。

## 1. 启动与检查

在 WSL 中逐行运行：

```bash
cd ~/Medical_Qwen
conda activate tcm_llm
python tcm_api.py --host 127.0.0.1 --port 8008
```

`Ctrl+C` 是键盘组合键，不是一条命令。后台服务已经占用端口时，不要重复启动。

- 接口文档：<http://127.0.0.1:8008/docs>
- 进程健康：`GET /healthz`
- 模型就绪：`GET /readyz`，应返回 `status=ready`、`model_version=V5.4-R1` 和 `stack_verification.status=PASS`。
- 启动时重新校验 51 个权重文件、基础权重、提示词源文件和适配器；校验或加载失败会中止启动。

本次本机服务监听 127.0.0.1，未启用 Bearer 鉴权；地址仅供这台电脑和可访问宿主机的模拟器调试。
单进程单 GPU；不要增加 uvicorn workers，否则每个进程会重复加载权重。

## 2. App 请求格式

发送 JSON 对象：

```json
{
  "text": "我这阵子觉得嘴巴发干。",
  "state": null,
  "new_session": true,
  "request_id": "app-001"
}
```

```bash
curl http://127.0.0.1:8008/v1/tcm/process \
  -H 'Content-Type: application/json' \
  -d '{"text":"我这阵子觉得嘴巴发干。","state":null,"new_session":true,"request_id":"app-001"}'
```

Swagger 中点击 Try it out 后，**全选并替换**请求框内容；只保留一个 `{...}`，不能把两个 JSON 对象接在一起。

响应字段：

| 字段 | App 用途 |
|---|---|
| `request_id` | 和本轮请求对应；不是幂等键，重试由 App 决定 |
| `result.message` | 直接显示给用户 |
| `result.action` | `ask` 追问、`summarize` 事实摘要、`urgent` 急症分流、`refer` 线下面诊建议；知识路径保留 `answer` / `knowledge_not_found` |
| `result.state` | 原样保存，下轮原样回传；不要编辑里面的字段 |
| `result.complete` | 本轮任务是否完成 |
| `result.meta.model_version` | `V5.4-R1` |
| `result.meta.model_used` | 本轮是否调用模型；急症、转诊和知识检索路径可绕过模型 |
| `result.meta.protocol` | 问诊路径的结构化追问/摘要结果 |

第二轮：将上一轮完整的 `result.state` 放进 `state`，`new_session=false`，`text` 填新的回答。
用户点击“新建问诊”时设置 `new_session=true`。每个用户、每个会话独立保存状态。
旧 V6.17 状态首次接入会重置并返回 `meta.session_event=legacy_state_reset`。

状态在客户端保存，服务端用 `.runtime/state.key` 签名；该文件不进入 Git 或交付包。
保留该密钥，服务重启后状态仍有效。多实例部署必须共享通过环境变量设置的 `TCM_API_STATE_SECRET`（至少 32 字节），并由 App 后端负责用户身份校验和会话归属。
状态包含用户陈述，不应记录到通用日志或跨用户共享。

## 3. XiaoMedInsight 配置

当前 Debug 配置默认为：

```text
http://10.0.2.2:8008/v1/tcm/process
```

该地址用于 Android 模拟器访问宿主电脑。现有客户端已会显示 message 并保存 state，无需因 V5.4 更改这两个字段的处理。

真机的 `127.0.0.1` 指向手机本身。真机需填写手机可访问的电脑地址或云服务地址，在 App 的 `local.properties` 设置后重新构建：

```properties
TCM_API_URL=https://你的域名/v1/tcm/process
```

Release 版本只接受 HTTPS。WSL NAT 下局域网真机访问可能还需 Windows 端口转发和防火墙配置；本轮未修改 Windows 网络配置，也没有完成真机安装联调。

## 4. 云端及鉴权

云端需要本项目代码、基础模型、R1 权重、冻结清单和适配器；若保留知识检索，还需要既有 BGE 模型与 RAG 索引。完整项目备份和 R1 归档包含这些历史产物，本次 API 代码包不重复包含大权重。

```bash
export TCM_API_KEY='替换为随机生成的服务端密钥'
export TCM_API_STATE_SECRET='替换为至少32字节的随机会话签名密钥'
python tcm_api.py --host 0.0.0.0 --port 8008
```

对外通过 HTTPS 反向代理。生产架构使用“App → 已有业务后端（用户登录鉴权）→ 模型 API”；业务后端持有密钥，并在访问模型 API 时携带：

```http
Authorization: Bearer <TCM_API_KEY>
Content-Type: application/json
```

不要把模型服务的长期共享密钥写进发布 APK。现有 XiaoMedInsight 客户端不发送此 Bearer 头，因此启用服务端密钥后应接到业务后端代理地址；本轮没有实现你的业务系统登录代理。
Swagger 右上角 Authorize 可以填入密钥进行接口调试。

受控局域网临时调试可显式添加 `--allow-unauthenticated`；不要将此模式暴露到公网。
网页客户端按需设置 `TCM_API_CORS_ORIGINS=https://你的网页域名`，原生 Android 网络调用不需要 CORS。
公网域名、云主机和 HTTPS 尚未部署，本机运行不等于手机随时联网可用。

## 5. 上限与错误

| 状态码 | 含义 / 客户端处理 |
|---|---|
| 200 | 查看 `result.action`，急症分流也是正常结果 |
| 401 | Bearer 凭据缺失或错误 |
| 413 | 文本/状态/会话太长；缩短输入或新建问诊 |
| 422 | JSON 格式错误、非法字段或 state 签名无效 |
| 429 | 队列满或排队超过 10 秒；遵循 Retry-After，避免自动连发 |
| 500 | 服务内部错误；使用 request_id 排查 |
| 503 | 模型未就绪 |

text 最多 4096 字符，HTTP body 最多 320 KiB，state 最多 256 KiB。
模型沿用冻结的 768 输入 token 上限（含提示词、已有事实和本轮文本），超限明确拒绝，不静默截断；每次生成最多 256 token。
最多 8 个在途请求，单次会话最多 24 个问诊轮次，实际可能先触及 token 上限。
客户端建议 connect timeout 15 秒、read timeout 60 秒、总超时 75 秒；已有 XiaoMedInsight 配置满足。

## 6. 能力范围和验证

这是结构化症状采集和用户陈述摘要 API，不输出个体诊断、辨证、处方或剂量。
H v2 按用户文本决定最终结果，不采用原始模型输出决定字段；HTTP 不返回原始生成。
问诊 API 新增的会话组合逻辑使用合成用例回归；未重新读取或使用已封存的盲测集。
纯知识问法沿用现有 intent_router 和 RAG，例如“什么是阴阳学说？”；不明确的问法可能被归入问诊。

回归命令：

```bash
python tests/test_api_v54.py
# 服务已启动时执行，实际调用模型：
python tests/smoke_api_v54.py
```

验证产物：`artifacts/api_v54_20260919/` 中的 `tests.log`、`live_smoke.json`、`sample_responses.json`、`openapi.json`。
本轮修改前的 API 备份在同目录 `prechange/tcm_api.py`。保留旧版本和冻结发布记录，不重写原有发布结论。
