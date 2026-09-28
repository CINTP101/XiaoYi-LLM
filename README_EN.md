# XiaoYi (小医)

XiaoYi, written **小医** in Chinese, is a work-in-progress TCM consultation model. It can collect symptom details, ask follow-up questions, summarize what a user has said, and expose those flows through an API for an app. It is **not** a diagnostic or prescribing service.

This repository contains source code and documentation. Model weights, training archives, the BGE model, and the RAG index are distributed separately. Cloning the repo alone will not start the full model service.

## Technical overview

The current release, **V5.4-R1**, adapts **Qwen2.5-1.5B-Instruct** to TCM consultation with **LoRA parameter-efficient fine-tuning**. Its goal is a controlled intake workflow rather than free-form diagnosis or treatment generation.

Requests pass through safety routing and intent detection first. Urgent or referral cases follow dedicated response paths; knowledge questions can use BGE/RAG retrieval; consultation requests use the model and then a Candidate H v2 adapter to produce a follow-up question or a factual summary. The app receives structured `action`, `message`, and signed session-state fields, not the raw model generation.

The FastAPI service checks the frozen weights and adapters at startup, and `/readyz` reports whether the stack is ready. Workflow code and selected reports from data preparation, training, and evaluation are kept here for traceability. **These engineering checks are not clinical validation**; XiaoYi remains an information-collection and research-demo tool.

## Start here

- [App API guide](docs/API_V54_APP.md) — requests, session state, authentication, and deployment notes.
- [Publishing and file guide](GITHUB_PUBLISH_GUIDE.md) — what is in Git and what must be restored separately.
- [Python client example](examples/app_api_client.py) — a small request example.
- `v5_2_pipeline/`, `v5_3_pipeline/`, and `v5_4_pipeline/` — historical data and training work.

## What XiaoYi does today

The current API can ask structured follow-up questions, summarize user-provided facts, route urgent cases, suggest an in-person visit when appropriate, and answer some knowledge questions through the existing retrieval path. Its main endpoint is `POST /v1/tcm/process`.

**XiaoYi** is the user-facing name. The current model release is **V5.4-R1**. It loads a **Qwen2.5-1.5B-Instruct** base model with this project's LoRA weights and runtime adapters. Technical version strings and archived file names remain unchanged so results can still be checked against the release records.

## Try the local API

First restore the base model, V5.4-R1 weights, release manifest, and adapters, and prepare the `tcm_llm` environment. From the project root:

```bash
conda activate tcm_llm
pip install -r requirements-api.txt
python tcm_api.py --host 127.0.0.1 --port 8008
```

Open <http://127.0.0.1:8008/docs>, or check `/healthz` and `/readyz`. Then send a request:

```bash
curl http://127.0.0.1:8008/v1/tcm/process \
  -H 'Content-Type: application/json' \
  -d '{"text":"My mouth has felt dry lately.","state":null,"new_session":true,"request_id":"demo-001"}'
```

For the next turn, send `result.state` back unchanged. The [API guide](docs/API_V54_APP.md) explains the response fields, app connection, and production security requirements.

## Background and limits

XiaoYi builds on [Medical_Qwen](https://github.com/scuterGuoyulong/Medical_Qwen). Some generic training examples in this repository originated with [MedicalGPT](https://github.com/shibing624/MedicalGPT). Those names identify upstream work; they are not names for XiaoYi's current release.

Historical reports and checksum files are kept under their original names. The API is for information collection and research demos, not individual diagnosis, treatment, or prescriptions. Please do not upload patient data, credentials, or large model files when contributing.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [LICENSE](LICENSE) for contribution and license details.
