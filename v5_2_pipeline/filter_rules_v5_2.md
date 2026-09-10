# V5.2 ShenNong safety filter and split rules

This document is the reusable, auditable policy for the V5.2 data build. The executable implementation is `v5_2_pipeline/data_pipeline.py`; the run configuration is `v5_2_pipeline/config.json`. The V5.1 paths are read only.

## Scope and inputs

- Raw source: `/home/cyh/Medical_Qwen/data/shennong/ChatMed_TCM-v0.2.json` (expected 112,565 JSON lines).
- Leakage references: `/home/cyh/Medical_Qwen/data/shennong/shennong_clean_v5_1.jsonl` (expected 1,200 lines) and `/home/cyh/Medical_Qwen/data/sft_v5_1/train.jsonl` (expected 2,730 lines).
- V5.2 outputs are written only below `/home/cyh/Medical_Qwen/artifacts/v5_2_pipeline/data/`.

The raw ShenNong file has no per-record auditable authority citation. The conservative authority rule is therefore: any response that makes a medical or TCM factual claim is treated as unverified and excluded. This avoids relying on model intuition to decide whether a claim is correct. The retained set teaches safe handling of underspecified or treatment-seeking requests through refusal, professional referral, or a request for missing context.

## Deterministic safety rules

Rules are evaluated in this order.

1. Parse each line as a JSON object with non-empty `query` and `response` strings. Normalize with Unicode NFKC and collapse whitespace. Reject malformed, missing, or out-of-range records (query 4–200 characters; response 15–650 characters).
2. A response is eligible only when every sentence is a short, complete refusal/referral or a pure request for missing context. A clarification may ask for context but may not promise later diagnosis, treatment, analysis, recommendations, prescribing, appointment booking, document operations, or general assistance. Courtesy text is allowed only when it does not add a claim or a promise. The exact refusal, clarification, and promise marker lists live in the executable rule file.
3. Query-side medical gate: if the query requests diagnosis, confirmation, syndrome differentiation, treatment, medical advice, Chinese medicine, formulas, herbs, drugs, prescriptions, medication, taking medicine, acupuncture, massage, food therapy, or dose, the item is eligible only when the answer explicitly refuses the requested medical service with language such as “不能/无法/没有资格/没有能力进行医学诊断” or “不能/无法提供医疗/医学建议或治疗建议”. Short or incomplete clinical prompts containing symptom, syndrome, patient, examination, tongue/pulse, pain, bleeding, fever, pregnancy, or similar clinical context are treated the same way. Such a medical request may never be retained as an `ask`/clarification, even when the answer only asks for symptoms. The refusal must not contain a follow-up question or operation promise. A query that is clearly documentary (paper/book/title/article editing or retrieval) is exempt only when it does not also request a clinical action.
4. Reject a response containing any actionable treatment or prescription content: herbs, formulas, medicines, prescriptions, opening a prescription, taking or decocting medicine, dose or units (`克`, `毫克`, `mg`, `每日`, `每次`), treatment instructions, efficacy, or recommendations.
5. Reject a response containing unverified factual medical content: diagnostic or syndrome claims, pathogenesis (`病机`), causes (`病因`/`原因`), symptom lists/classifications, disease/infection/inflammation claims, pharmacology, ingredients, or claims using markers such as “可能”, “通常”, “例如”, or “包括”. Meta-mentions inside a complete refusal (for example, “无法进行医学诊断”) are allowed only after the sentence-level refusal check removes those meta words; any additional factual clause fails. A clause containing factual linkers such as “是一种”, “可以分为”, “表示”, “根据中医”, or “由于” together with medical terms also fails.
6. Reject any response that exposes reasoning or chain-of-thought markers (`推理过程`, `思考过程`, `详细推理`, `一步步的推理`, `逐步推理`), or promises to reason after more context. A query may ask for reasoning, but no reasoning content or future reasoning promise may appear in the retained response.

An item that fails any rule is written once to `shennong_rejections_v5_2.jsonl` with its source line, original query/response, SHA-256 values, stage, and one or more reason codes. That file is an audit-only rejection log and is never passed to training or evaluation.

## Text and semantic deduplication

Exact matching uses Unicode NFKC/casefold text with whitespace and punctuation removed. Semantic near-duplicate matching uses sets of compact character 3-grams and Jaccard similarity `|A∩B|/|A∪B|`.

- Within the new source pool, reject a candidate when query Jaccard is at least `0.95` or full query/response pair Jaccard is at least `0.90`.
- Against the V5.1 mixed training reference, reject a candidate when query Jaccard is at least `0.90` or full pair Jaccard is at least `0.90`. Exact query and pair intersections are always rejected first.

This is a deterministic lexical semantic proxy; no unpinned embedding model is used. The report records all thresholds and counts.

## Blind split

After safety filtering, source deduplication, and V5.1 leakage removal, the remaining pool is stratified by response class (`safety_refusal` and `clarification`). With seed `20260910`, 20% of each class (at least one when a class has more than one row) is reserved as `shennong_heldout_v5_2.jsonl`; the remainder is `shennong_clean_v5_2.jsonl`. Files are sorted by source line after the split. The pipeline rechecks exact and semantic intersections between the two outputs and requires zero.

## Reproducible audit sampling

The same fixed seed selects up to 5 accepted examples per acceptance class and up to 3 rejected examples per primary rejection reason. Every sampled rejection must carry a reason code; `statistics_v5_2.json` records the sample size and measured rejection rate. The expected rejection rate for the rejection sample is exactly 100%. The same statistics object also records the independent retained-set full scan and the fixed 10-per-class retained safety sample; those safety checks are the acceptance criterion for retained rows.

## Required evidence

`statistics_v5_2.json`, `dedup_report_v5_2.json`, `overlap_report_v5_2.json`, and `run_evidence_v5_2.json` record line counts, inputs, thresholds, split seed, intersections, invariants, and runtime hashes. `sha256sums_v5_2.txt` records SHA-256 values for all material V5.2 data and audit outputs. The V5.1 directory is not written by this pipeline.

`retention_verification_v5_2.json` re-reads both output JSONL files and applies the query-side medical gate and response predicate to every retained assistant message/question. It reports the full-scan pass count, medical-request/refusal counts, unapproved hard-block/factual/promise-term counts, and a separate fixed-seed sample of 10 rows per retained response class. Delivery requires a 100% full-scan pass rate and a 100% sample pass rate; a nonzero violation count is a failed run.
