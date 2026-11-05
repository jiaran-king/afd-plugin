# E2E Scenario Matrix and Proposed Changes (RFC Appendix)

[简体中文](e2e_case_matrix.md) | English

Status: design and coverage proposal; tests and CI have not been changed. This is an appendix to the [E2E framework RFC](e2e_test_framework.en.md), for reviewing retention, replacement relationships and migration acceptance. It is not an execution log or a device-pass report.

Baseline: AFD `main@e778192d4c1cfd47e97be1881047532bfce7a7c0` (2026-11-05). This translates the proposal published at `2209c5ca5ada5056abfa69c59c825ba3e7340240` without changing its design or scenario dispositions.

## Reading guide and counting rules

Navigation: [A is the current baseline](#current-matrix); [removal conditions gate deletion](#deletion-gates); [B shows conditional target changes](#target-matrix), with the [complete target list folded](#full-target-matrix); [C lists independent entries, device-specific tests and host checks](#other-entries), outside the 57 service scenarios.

> [!WARNING]
> **Do not remove G17/G37 unless their conditions hold. Retain G02 this round.**
> Removal conditions are not yet satisfied. The target table does not mean removals have occurred or replacement validation has passed.

- Count current service scenarios by model, backend, pytest scenario parameter and pod layout: **57 entries**. GPU/NPU implementations of an identical node ID count separately; pod count does not multiply one multi-pod scenario.
- List device scripts without pytest entries, operator acceptance, the external two-stream driver and **37 host checks** separately. They are not service-scenario counts.
- Current acceptance describes what code requires, not device results obtained in this work. CI selection is scheduling information; lack of selection does not remove retention value.
- Service cases retain current liveness, cancellation and cleanup rules. Tables emphasize distinguishing checks. NPU cleanup exceptions and FFN graph's lack of graceful-shutdown qualification are stated separately.
- `graph` follows the current roles' CUDA/ACL Graph configuration; it is distinct from FFN `FULL`.
- G/N/P references connect the RFC's tables, not proposed pytest IDs. Confirm new scenario names during implementation.

| Proposed action | Meaning | Current rows |
| --- | --- | ---: |
| Retain and migrate | Preserve the behavior; shared-code migration does not weaken acceptance | 47 |
| Retain limited scope | Keep functional smoke and existing path checks without claiming answer correctness | 2 |
| Enhance | Keep the scenario and add missing execution-path evidence | 6 |
| Conditional removal | Candidate for leaving fixed regression coverage; replacement, risk acceptance and execution conditions remain pending | 2 |

Retain G02's minimal two-device pure-eager service and diagnostic value. It is outside current CI, so removal has no direct pipeline benefit. G17/G37 are conditional candidates, not equivalent replacements or a ceiling on trimming. Future decisions use actual value and cost, not a deletion quota.

<a id="current-matrix"></a>

## A. All current service scenarios and proposed dispositions

Retain current GSM8K profiles: the V2-Lite smoke threshold is 0.25; larger samples and other models generally use 0.27. Exact floors, generation settings, full runs and local override rules remain in the [current contract](module/e2e_testing.md#accuracy-gate). Table sample counts assume no debugging override.

### GPU single-host / DeepSeek-V2-Lite / test_deepseek_v2_lite

Entry: [tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py](../../tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py), function `test_deepseek_v2_lite`. Append each scenario parameter below to this function's pytest node ID.

| ID / scenario parameter | Deployment and modes | Current workload and acceptance | Existing CI selection | Proposed action | Independent value, replacement or condition |
| --- | --- | --- | --- | --- | --- |
| G01 / `baseline-graph` | 4 GPU; native DP4×TP1; graph | GSM8K, 7 questions | ready/merge | Retain and migrate | Native, non-AFD reference for the same model. |
| G02 / `afd-v2-eager-1a1f` | 2 GPU; A/F DP1×TP1; eager | GSM8K, 7 questions | Not selected by existing CI | Retain and migrate | Minimal two-device full-service eager entry and diagnostic path. G05 introduces graph and G03 requires four devices; neither is equivalent. Retain this round and reassess later using actual use and maintenance cost. |
| G03 / `afd-v2-eager-dp2` | 4 GPU; A/F DP2×TP1; eager | GSM8K, 128 questions / 12 concurrent requests | ready/merge | Retain and migrate | Pure-eager DP2 requests and transport; retain the same 128-question workload as G08/G09 for comparison. |
| G04 / `afd-v2-eager-tp2` | 4 GPU; A/F DP1×TP2; eager | GSM8K, 7 questions | ready/merge | Retain and migrate | Pure-eager P2P integration with TP2 on both roles. |
| G05 / `afd-v2-graph-1a1f` | 2 GPU; A/F DP1×TP1; graph | GSM8K, 7 questions; per-rank graph | Not selected by existing CI | Retain and migrate | Minimal DP1/TP1 deployment with real graph execution; preserve the minimal process/topology path. |
| G06 / `afd-v2-graph-dp2` | 4 GPU; A/F DP2×TP1; graph | GSM8K, 7 questions; per-rank graph | ready/merge | Retain and migrate | DP2 graph replay across all A/F ranks. |
| G07 / `afd-v2-graph-tp2` | 4 GPU; A/F DP1×TP2; graph | GSM8K, 7 questions; per-rank graph | ready/merge | Retain and migrate | TP2 graph replay on both roles. |
| G08 / `afd-v2-eager-dbo-dp2` | 4 GPU; A/F DP2×TP1; eager+DBO | GSM8K, 128 questions / 12 concurrent requests; per-rank two-stage layouts | ready/merge | Retain and migrate | Pure-eager live DBO, not replaced by graph+DBO. |
| G09 / `afd-v2-graph-dbo-dp2` | 4 GPU; A/F DP2×TP1; graph+DBO | GSM8K, 128 questions / 12 concurrent requests; per-rank consecutive graph replay + two-stage layouts | ready/merge | Retain and migrate | DP2 graph execution combined with two-stage layouts. |
| G10 / `afd-v2-graph-dbo-tp2` | 4 GPU; A/F DP1×TP2; graph+DBO | GSM8K, 24 questions / 12 concurrent requests; per-rank consecutive graph replay + two-stage layouts | Not selected by existing CI | Retain and migrate | DP1×TP2+DBO exercises the GPU ubatching compatibility path without a DP handshake. |
| G11 / `afd-eager-2a1f` | 3 GPU; A DP2×TP1 / F DP1×TP1; eager | GSM8K, 7 questions | Not selected by existing CI | Retain and migrate | Pure-eager transport merging multiple Attention inputs into one FFN. |
| G12 / `afd-graph-2a1f` | 3 GPU; A DP2×TP1 / F DP1×TP1; graph | GSM8K, 7 questions; per-rank graph | Not selected by existing CI | Retain and migrate | Ordinary single-stage graph on the same asymmetric topology. |
| G13 / `afd-graph-dbo-2a1f` | 3 GPU; A DP2×TP1 / F DP1×TP1; graph+DBO | GSM8K, 24 questions / 12 concurrent requests; per-rank consecutive graph replay + two-stage layouts | weekly | Retain and migrate | Asymmetric topology combined with graph+DBO. |

### GPU single-host / DeepSeek-V2-Lite / test_deepseek_v2_lite_deepep_m2n

Entry: [tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py](../../tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py), function `test_deepseek_v2_lite_deepep_m2n`. Append each scenario parameter below to this function's pytest node ID.

| ID / scenario parameter | Deployment and modes | Current workload and acceptance | Existing CI selection | Proposed action | Independent value, replacement or condition |
| --- | --- | --- | --- | --- | --- |
| G14 / `afd-deepep-m2n-eager-2a2f` | 4 GPU; A/F DP2×TP1; eager | GSM8K, 7 questions | Not selected by existing CI | Retain and migrate | Complete pure-eager integration of DeepSeek shared experts and Attention-side routing. |
| G15 / `afd-deepep-m2n-graph-2a2f` | 4 GPU; A/F DP2×TP1; graph | GSM8K, 7 questions; per-rank graph | Not selected by existing CI | Retain and migrate | Ordinary DeepEP graph path for the same model. |
| G16 / `afd-deepep-m2n-graph-dbo-2a2f` | 4 GPU; A/F DP2×TP1; graph+DBO | GSM8K, 24 questions / 12 concurrent requests; per-rank consecutive graph replay + two-stage layouts | Not selected by existing CI | Retain and migrate | DeepEP graph/DBO combined with this model's shared-expert integration. |
| G17 / `afd-deepep-m2n-eager-tp2-2a2f` | 4 GPU; A DP1×TP2 / F DP2×TP1; eager | GSM8K, 7 questions | Not selected by existing CI | Conditional removal | Candidate for trimming the TP2×eager full-model combination; G14, G18 and D01 share the remaining responsibilities. See removal conditions below. |
| G18 / `afd-deepep-m2n-graph-tp2-2a2f` | 4 GPU; A DP1×TP2 / F DP2×TP1; graph | GSM8K, 7 questions; per-rank graph | Not selected by existing CI | Retain and migrate | Graph integration of shared experts and TP token slicing for this model. |

### GPU single-host / DeepSeek-V4-Flash / test_deepseek_v4_flash_gpu

Entry: [tests/e2e/models/deepseek_v4_flash/test_gpu.py](../../tests/e2e/models/deepseek_v4_flash/test_gpu.py), function `test_deepseek_v4_flash_gpu`. Append each scenario parameter below to this function's pytest node ID.

| ID / scenario parameter | Deployment and modes | Current workload and acceptance | Existing CI selection | Proposed action | Independent value, replacement or condition |
| --- | --- | --- | --- | --- | --- |
| G19 / `baseline-dsv4-flash-eager` | 2 GPU; native DP2×TP1; eager | Two concurrent answer rounds + GSM8K, 7 questions | Not selected by existing CI | Retain and migrate | Native eager reference for the same DSV4 configuration. |
| G20 / `baseline-dsv4-flash-graph` | 2 GPU; native DP2×TP1; graph | Two concurrent answer rounds + GSM8K, 7 questions; per-rank graph | Not selected by existing CI | Retain and migrate | Native graph reference for the same DSV4 configuration. |
| G21 / `afd-dsv4-flash-eager-2a2f` | 4 GPU; A/F DP2×TP1; eager | Two concurrent answer rounds + GSM8K, 7 questions; Hash pairing/freshness | Not selected by existing CI | Retain and migrate | DSV4 AFD eager Hash-input and result consistency. |
| G22 / `afd-dsv4-flash-graph-2a2f` | 4 GPU; A/F DP2×TP1; graph | Two concurrent answer rounds + GSM8K, 7 questions; Hash pairing/freshness; per-rank graph | Not selected by existing CI | Retain and migrate | DSV4 AFD graph Hash pairing and cross-round freshness. |

### GPU single-host / DeepSeek-V4-Flash / test_deepseek_v4_flash_gpu_dbo

Entry: [tests/e2e/models/deepseek_v4_flash/test_gpu.py](../../tests/e2e/models/deepseek_v4_flash/test_gpu.py), function `test_deepseek_v4_flash_gpu_dbo`. Append each scenario parameter below to this function's pytest node ID.

| ID / scenario parameter | Deployment and modes | Current workload and acceptance | Existing CI selection | Proposed action | Independent value, replacement or condition |
| --- | --- | --- | --- | --- | --- |
| G23 / `baseline-dsv4-flash-eager-dbo` | 2 GPU; native DP2×TP1; eager+DBO | Two concurrent answer rounds + GSM8K, 24 questions; per-rank two-stage layouts | Not selected by existing CI | Retain and migrate | Native eager DBO reference and real two-stage execution; graph DBO does not replace it. |
| G24 / `baseline-dsv4-flash-graph-dbo` | 2 GPU; native DP2×TP1; graph+DBO | Two concurrent answer rounds + GSM8K, 24 questions; per-rank consecutive graph replay + two-stage layouts | Not selected by existing CI | Retain and migrate | Native graph DBO reference with evidence of real two-substage execution. |
| G25 / `afd-dsv4-flash-eager-dbo-2a2f` | 4 GPU; A/F DP2×TP1; eager+DBO | Two concurrent answer rounds + GSM8K, 24 questions; Hash pairing/freshness; per-rank two-stage layouts | Not selected by existing CI | Retain and migrate | Hash and two-stage execution under AFD pure-eager DBO. |
| G26 / `afd-dsv4-flash-graph-dbo-2a2f` | 4 GPU; A/F DP2×TP1; graph+DBO | Two concurrent answer rounds + GSM8K, 24 questions; Hash pairing/freshness; per-rank consecutive graph replay + two-stage layouts | Not selected by existing CI | Retain and migrate | Per-window Hash checks and consecutive replay under AFD graph+DBO. |

### GPU single-host / Qwen3.6 / test_qwen3_6

Entry: [tests/e2e/models/qwen3_6/test_qwen3_6.py](../../tests/e2e/models/qwen3_6/test_qwen3_6.py), function `test_qwen3_6`. Append each scenario parameter below to this function's pytest node ID.

| ID / scenario parameter | Deployment and modes | Current workload and acceptance | Existing CI selection | Proposed action | Independent value, replacement or condition |
| --- | --- | --- | --- | --- | --- |
| G27 / `baseline-graph` | 4 GPU; native DP4×TP1; graph | GSM8K, 7 questions | weekly | Retain and migrate | Native reference for the GDN model. |
| G28 / `afd-eager-2a1f` | 3 GPU; A DP2×TP1 / F DP1×TP1; eager | GSM8K, 7 questions | weekly | Retain and migrate | Pure-eager state path for the GDN model. |
| G29 / `afd-graph-2a1f` | 3 GPU; A DP2×TP1 / F DP1×TP1; graph | GSM8K, 7 questions; per-rank graph | weekly | Retain and migrate | GDN graph-replay state path; does not restore GDN DBO. |

### GPU single-host / Qwen3 MoE / test_qwen3_moe

Entry: [tests/e2e/models/qwen3_moe/test_qwen3_moe.py](../../tests/e2e/models/qwen3_moe/test_qwen3_moe.py), function `test_qwen3_moe`. Append each scenario parameter below to this function's pytest node ID.

| ID / scenario parameter | Deployment and modes | Current workload and acceptance | Existing CI selection | Proposed action | Independent value, replacement or condition |
| --- | --- | --- | --- | --- | --- |
| G30 / `baseline-graph` | 4 GPU; native DP4×TP1; graph | GSM8K, 7 questions | weekly | Retain and migrate | Native reference for the Qwen3 model. |
| G31 / `afd-eager-2a1f` | 3 GPU; A DP2×TP1 / F DP1×TP1; eager | GSM8K, 7 questions | weekly | Retain and migrate | Qwen3 P2P pure-eager model integration. |
| G32 / `afd-graph-2a1f` | 3 GPU; A DP2×TP1 / F DP1×TP1; graph | GSM8K, 7 questions; per-rank graph | weekly | Retain and migrate | Ordinary single-stage P2P graph for the same model. |
| G33 / `afd-graph-dbo-2a1f` | 3 GPU; A DP2×TP1 / F DP1×TP1; graph+DBO | GSM8K, 24 questions / 12 concurrent requests; per-rank consecutive graph replay + two-stage layouts | weekly | Retain and migrate | P2P graph+DBO for the same model. |

### GPU single-host / Qwen3 MoE / test_qwen3_moe_deepep_m2n

Entry: [tests/e2e/models/qwen3_moe/test_qwen3_moe.py](../../tests/e2e/models/qwen3_moe/test_qwen3_moe.py), function `test_qwen3_moe_deepep_m2n`. Append each scenario parameter below to this function's pytest node ID.

| ID / scenario parameter | Deployment and modes | Current workload and acceptance | Existing CI selection | Proposed action | Independent value, replacement or condition |
| --- | --- | --- | --- | --- | --- |
| G34 / `afd-deepep-m2n-eager-2a2f` | 4 GPU; A/F DP2×TP1; eager | GSM8K, 7 questions | Not selected by existing CI | Retain and migrate | Qwen DeepEP pure-eager integration without shared experts, distinct from DeepSeek integration. |
| G35 / `afd-deepep-m2n-graph-2a2f` | 4 GPU; A/F DP2×TP1; graph | GSM8K, 7 questions; per-rank graph | Not selected by existing CI | Retain and migrate | Single-stage DeepEP graph for the same model. |
| G36 / `afd-deepep-m2n-graph-dbo-2a2f` | 4 GPU; A/F DP2×TP1; graph+DBO | GSM8K, 24 questions / 12 concurrent requests; per-rank consecutive graph replay + two-stage layouts | Not selected by existing CI | Retain and migrate | DeepEP graph+DBO for the same model. |
| G37 / `afd-deepep-m2n-eager-tp2-2a2f` | 4 GPU; A DP1×TP2 / F DP2×TP1; eager | GSM8K, 7 questions | Not selected by existing CI | Conditional removal | Candidate for trimming the TP2×eager full-model combination; G34, G38 and D01 share the remaining responsibilities. See removal conditions below. |
| G38 / `afd-deepep-m2n-graph-tp2-2a2f` | 4 GPU; A DP1×TP2 / F DP2×TP1; graph | GSM8K, 7 questions; per-rank graph | Not selected by existing CI | Retain and migrate | Graph integration of Qwen TP2 sharding and routing. |

### NPU single-host / DeepSeek-V2-Lite / test_deepseek_v2_lite

Entry: [tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py](../../tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py), function `test_deepseek_v2_lite`. Append each scenario parameter below to this function's pytest node ID.

| ID / scenario parameter | Deployment and modes | Current workload and acceptance | Existing CI selection | Proposed action | Independent value, replacement or condition |
| --- | --- | --- | --- | --- | --- |
| N01 / `baseline-graph` | 4 NPU; native DP4×TP1; graph | GSM8K, 7 questions | Not selected by existing CI | Retain and migrate | Native Ascend reference; a CUDA baseline does not replace it. |
| N02 / `afd-eager-2a2f` | 4 NPU; A/F DP2×TP1; eager | GSM8K, 7 questions | Not selected by existing CI | Retain and migrate | Ascend CAMP2P pure-eager service lifecycle. |
| N03 / `afd-graph-2a2f` | 4 NPU; A/F DP2×TP1; graph | GSM8K, 7 questions; per-rank graph | Not selected by existing CI | Retain and migrate | Ordinary single-stage Ascend graph and communication. |
| N04 / `afd-graph-dbo-2a2f` | 4 NPU; A/F DP2×TP1; graph+DBO | GSM8K, 24 questions / 12 concurrent requests; per-rank consecutive graph replay + two-stage layouts | Not selected by existing CI | Retain and migrate | Ascend graph+DBO and two-stage execution. |
| N05 / `afd-eager-2a1f` | 3 NPU; A DP2×TP1 / F DP1×TP1; eager | GSM8K, 7 questions | Not selected by existing CI | Retain and migrate | Asymmetric NPU merging of multiple Attention inputs, in pure eager. |
| N06 / `afd-graph-2a1f` | 3 NPU; A DP2×TP1 / F DP1×TP1; graph | GSM8K, 7 questions; per-rank graph | Not selected by existing CI | Retain and migrate | Ordinary graph on an asymmetric NPU topology. |
| N07 / `afd-graph-dbo-2a1f` | 3 NPU; A DP2×TP1 / F DP1×TP1; graph+DBO | GSM8K, 24 questions / 12 concurrent requests; per-rank consecutive graph replay + two-stage layouts | Not selected by existing CI | Retain and migrate | Graph+DBO on an asymmetric NPU topology. |

### NPU single-host / DeepSeek-V4-Flash / test_deepseek_v4_flash_async_cam

Entry: [tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py](../../tests/e2e/models/deepseek_v4_flash/test_async_cam_npu.py), function `test_deepseek_v4_flash_async_cam`. Append each scenario parameter below to this function's pytest node ID.

| ID / scenario parameter | Deployment and modes | Current workload and acceptance | Existing CI selection | Proposed action | Independent value, replacement or condition |
| --- | --- | --- | --- | --- | --- |
| N08 / `afd-dsv4-flash-async-cam-dp2tp4-ep8` | 16 NPU; A DP2×TP4 / F DP8×TP1; A/F eager; Attention configured with DBO, activation not qualified | 10 concurrent answers with request overlap; does not qualify FFN FULL | Not selected by existing CI | Retain and migrate | Existing 16-NPU async deployment and concurrent answers; topology and purpose differ from the 8-NPU FFN FULL test. |

### NPU single-host / DeepSeek-V4-Flash / test_deepseek_v4_flash_sync_camp2p

Entry: [tests/e2e/models/deepseek_v4_flash/test_sync_camp2p_npu.py](../../tests/e2e/models/deepseek_v4_flash/test_sync_camp2p_npu.py), function `test_deepseek_v4_flash_sync_camp2p`. Append each scenario parameter below to this function's pytest node ID.

| ID / scenario parameter | Deployment and modes | Current workload and acceptance | Existing CI selection | Proposed action | Independent value, replacement or condition |
| --- | --- | --- | --- | --- | --- |
| N09 / `afd-dsv4-flash-sync-camp2p-2a2f` | A5, 4 NPU; A/F DP2×TP1; graph, DBO disabled | 10 concurrent nonempty completions + per-rank graph; no answer validation | Not selected by existing CI | Retain limited scope | Retain A5 graph and concurrent plumbing; answer correctness remains unqualified. |
| N10 / `afd-dsv4-flash-sync-camp2p-8a8f` | A3, 16 NPU; A DP2×TP4 / F DP8×TP1; eager | 10 concurrent nonempty completions; no answer validation | Not selected by existing CI | Retain limited scope | Retain A3 large-topology concurrent plumbing; answer correctness remains unqualified. |

### GPU multi-pod / DeepSeek-V2-Lite / test_multi_pod

Entry: [tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite_multi_pod.py](../../tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite_multi_pod.py), function `test_multi_pod`. Append each scenario parameter below to this function's pytest node ID.

| ID / scenario parameter | Deployment and modes | Current workload and acceptance | Existing CI selection | Proposed action | Independent value, replacement or condition |
| --- | --- | --- | --- | --- | --- |
| P01 / `afd-graph-2a2f-2pod-role-split` | 4 GPU; A DP2×TP1 / F DP2×TP1; role-split pods; graph | GSM8K, coordination and cleanup; actual graph/DBO execution unverified | Not selected by existing CI | Enhance | Retain role-split deployment; add cross-pod graph execution proof. |
| P02 / `afd-graph-2a2f-2pod-interleaved` | 4 GPU; A DP2×TP1 / F DP2×TP1; interleaved layout; graph | GSM8K, coordination and cleanup; actual graph/DBO execution unverified | Not selected by existing CI | Enhance | Retain interleaved deployment and its communication boundary; add graph proof. |
| P03 / `afd-graph-dbo-2a2f-2pod-role-split` | 4 GPU; A DP2×TP1 / F DP2×TP1; role-split pods; graph+DBO | GSM8K, coordination and cleanup; actual graph/DBO execution unverified | Not selected by existing CI | Enhance | Retain two-stage execution with role separation; add cross-pod graph+DBO proof. |
| P04 / `afd-graph-dbo-2a2f-2pod-interleaved` | 4 GPU; A DP2×TP1 / F DP2×TP1; interleaved layout; graph+DBO | GSM8K, coordination and cleanup; actual graph/DBO execution unverified | Not selected by existing CI | Enhance | Retain interleaved two-stage execution; add cross-pod graph+DBO proof. |

### GPU multi-pod / Qwen3 MoE / test_multi_pod_deepep_m2n

Entry: [tests/e2e/models/qwen3_moe/test_qwen3_moe_multi_pod.py](../../tests/e2e/models/qwen3_moe/test_qwen3_moe_multi_pod.py), function `test_multi_pod_deepep_m2n`. Append each scenario parameter below to this function's pytest node ID.

| ID / scenario parameter | Deployment and modes | Current workload and acceptance | Existing CI selection | Proposed action | Independent value, replacement or condition |
| --- | --- | --- | --- | --- | --- |
| P05 / `afd-deepep-m2n-eager-2a2f-2pod-role-split` | 4 GPU; A DP2×TP1 / F DP2×TP1; role-split pods; eager | GSM8K, default 100 questions, coordination and cleanup; graph/DBO proof not required | Not selected by existing CI | Retain and migrate | Retain DeepEP cross-pod pure-eager services. |
| P06 / `afd-deepep-m2n-graph-2a2f-2pod-role-split` | 4 GPU; A DP2×TP1 / F DP2×TP1; role-split pods; graph | GSM8K, default 100 questions, coordination and cleanup; actual graph/DBO execution unverified | Not selected by existing CI | Enhance | Retain DeepEP cross-pod graph; add per-rank proof. |
| P07 / `afd-deepep-m2n-graph-dbo-2a2f-2pod-role-split` | 4 GPU; A DP2×TP1 / F DP2×TP1; role-split pods; graph+DBO | GSM8K, default 100 questions, coordination and cleanup; actual graph/DBO execution unverified | Not selected by existing CI | Enhance | Retain DeepEP cross-pod graph+DBO; add matching-window and layout proof. |

### NPU multi-pod / DeepSeek-V4-Flash / test_deepseek_v4_flash_multi_pod

Entry: [tests/e2e/models/deepseek_v4_flash/test_deepseek_v4_flash_multi_pod.py](../../tests/e2e/models/deepseek_v4_flash/test_deepseek_v4_flash_multi_pod.py), function `test_deepseek_v4_flash_multi_pod`. Append each scenario parameter below to this function's pytest node ID.

| ID / scenario parameter | Deployment and modes | Current workload and acceptance | Existing CI selection | Proposed action | Independent value, replacement or condition |
| --- | --- | --- | --- | --- | --- |
| P08 / `2pod-role-split` | 16 NPU; A DP2×TP4 / F DP8×TP1; role-split pods; A/F eager; Attention configured with DBO, activation not qualified | 10 concurrent answers, coordination and cleanup; no graph/DBO execution-path proof | Not selected by existing CI | Retain and migrate | Retain async CAM cross-pod role separation and answer checks; no FFN FULL claim. |
| P09 / `2pod-interleaved` | 16 NPU; A DP2×TP4 / F DP8×TP1; interleaved layout; A/F eager; Attention configured with DBO, activation not qualified | 10 concurrent answers, coordination and cleanup; no graph/DBO execution-path proof | Not selected by existing CI | Retain and migrate | Retain interleaved placement with local TP groups and answer checks; no FFN FULL claim. |

<a id="deletion-gates"></a>

### Conditional removal requirements

Retain G02 this round. G05's `FULL_DECODE_ONLY` and G03's DP2 eager service do not replace it. The following conditions apply only to G17/G37.

| Candidate | Retained or substitute coverage | Conditions required before removal |
| --- | --- | --- |
| G17 | G14: same-model DeepEP pure eager; G18: same-model TP2 graph; D01: GPU TP2-specific validation | Retained G14/G18 pass; `D01-TP2-4GPU` actually records both passes with zero skips and has execution ownership. Otherwise retain G17. Explicitly accept no longer independently regressing the complete V2-Lite+DeepEP+TP2+pure-eager service combination |
| G37 | G34: same-model DeepEP pure eager; G38: same-model TP2 graph; D01: GPU TP2-specific validation | Retained G34/G38 pass; the same D01 task has a valid zero-skip record and owner. DeepSeek E2E does not replace Qwen integration. Otherwise retain G37. Explicitly accept no longer independently regressing the complete Qwen3+DeepEP+TP2+pure-eager service combination |

Passing TP1 pure-eager service, TP2 graph service and TP2 eager device tests independently does not prove equivalent full model/scheduler/lifecycle combination coverage. The implementation PR must accept the retired scope and stop claiming independent full-service coverage for that combination. Reassess and restore service cases as needed if combination-specific regressions appear or substitutes weaken or stop running.

D01's [TP-slicing tests](../../tests/gpu/test_deepep_m2n_gpu.py) use real GPUs and cover TP2, different token counts, empty DP and single/two-lane steps. They do not start full vLLM services and must complement retained model E2E. Current CI does not select them; insufficient hardware skips them. Code existence alone is not replacement qualification.

<a id="d01-manual-task"></a>

#### Manual task `D01-TP2-4GPU`

This RFC defines the replacement task for G17/G37. It has not been executed or qualified here, and no Buildkite task has been added. Obtain a valid single-host allocation of four SM90+ GPUs, prepare pinned runtime versions under the [GPU test requirements](../../tests/gpu/README.md#requirements), and run from the repository root using allocated device visibility.

Select only `test_attention_tp_peers_dispatch_disjoint_slices` with `2F-1DPx2TP`: Attention DP1×TP2 and FFN DP2×TP1, with BF16 and FP8 payloads. Do not include the six-device `2F-2DPx2TP` subset in this four-device task.

```bash
set -e
export AFD_D01_RESULTS="$(mktemp -d /tmp/afd-d01-tp2-4gpu.XXXXXX)"
python -m pytest -sv -m gpu \
  "tests/gpu/test_deepep_m2n_gpu.py::test_attention_tp_peers_dispatch_disjoint_slices[2F-1DPx2TP-bf16]" \
  "tests/gpu/test_deepep_m2n_gpu.py::test_attention_tp_peers_dispatch_disjoint_slices[2F-1DPx2TP-fp8]" \
  --junitxml="$AFD_D01_RESULTS/junit.xml" \
  --basetemp="$AFD_D01_RESULTS/pytest"

python - <<'PY'
import os
import xml.etree.ElementTree as ET
from pathlib import Path

cases = ET.parse(Path(os.environ["AFD_D01_RESULTS"]) / "junit.xml").findall(".//testcase")
assert len(cases) == 2, "D01 must execute exactly the two selected GPU cases"
assert all(
    case.find(tag) is None
    for case in cases
    for tag in ("failure", "error", "skipped")
), "D01 requires two passes and zero failures, errors or skips"
PY
```

Execution ownership: the removal PR author arranges the initial run and supplies records. Authors of subsequent PRs affecting DeepEP Attention TP slicing, dispatch/combine or payload paths rerun the task and provide records. Without resources and accountable execution, G17/G37 remain.

Record tested AFD/runtime versions, hardware, full commands and exit status. Retain JUnit, console and rank logs as task artifacts and link records matching the implementation PR's code/environment in its description. Successful pytest exit with skips, collection only, missing dependencies or absent run records do not satisfy removal conditions. The command definition is not evidence of a device run in this work.

These conditions are still pending. Do not just delete the two candidate node IDs or withdraw their product configurations. Neither candidate is selected by current required gates. Existing ready/merge and weekly required selections are retained.

<a id="target-matrix"></a>

## B. Target changes after conditions hold (proposal)

This table lists only 2 conditional removals, 3 new entries and 6 existing acceptance enhancements. The other 49 current service scenarios migrate with retained coverage, including N09/N10's limited smoke scope. Removal conditions above remain binding. A always preserves the complete current baseline.

Retain G02. If G17/G37 both qualify for removal, 57 current scenarios minus 2 fixed regression combinations plus 3 entries gives **58 target scenarios**. Restore any candidate whose conditions do not hold; retaining both gives 60. Counts are a result, not a target.

Preserve **15 unique node IDs: ready and merge share 7; weekly adds 8**. Resources, frequency and budget for new hardware tasks remain to be settled. Proposed periodic/targeted execution is not an existing pipeline or device record. Implement in batches; additions need not land with the first framework extraction.

| Action | ID / scenario | Conditional target change | Conditions / resources |
| --- | --- | --- | --- |
| Conditional removal | G17 / V2-Lite DeepEP eager TP2 | Retain G14/G18 integration; D01 complements TP2 eager communication coverage | Same-model retained cases pass, D01 has zero-skip records and an owner, and the retired full-combination risk is accepted; otherwise retain |
| Conditional removal | G37 / Qwen3 DeepEP eager TP2 | Retain G34/G38 integration; D01 complements TP2 eager communication coverage | Same-model retained cases pass, D01 has zero-skip records and an owner, and the retired full-combination risk is accepted; otherwise retain |
| Add | G39 / V2-Lite graph 3A2F | Native-reference then 3A2F graph deployments; per-request comparison, length/workload variation, FFN-group participation and cleanup | Single host, 5 GPU; proposed periodic/targeted task outside the four-device PR gate |
| Add entry | N11 / V2-Lite eager DBO 2A2F | Stable pytest entry for an existing manual configuration; native 32/512 thresholds; prove live two-stage execution in the specified prefill-heavy window, without decode DBO qualification | Single host, 4 NPU; resources and frequency pending |
| Add | N12 / DSV4 async CAM FFN FULL 4A4F | Self-contained eager/FULL comparison; length, repeated inputs, real concurrency, deployment capture, per-window replay increments, tokens and graceful shutdown of both runs | Single host, 8 NPU; `cleanup_profile=strict`; shutdown remains unqualified |
| Enhance | P01 / V2-Lite graph, role-split pods | Add cross-pod graph-window evidence for every A/F rank to current result checks | 4 GPU, two pods; first implement evidence transport and coordination |
| Enhance | P02 / V2-Lite graph, interleaved layout | Same enhancement while retaining the interleaved communication boundary | 4 GPU, two pods; first implement evidence transport and coordination |
| Enhance | P03 / V2-Lite graph+DBO, role-split pods | Add matching cross-pod two-stage layouts and consecutive graph replay | 4 GPU, two pods; first implement evidence transport and coordination |
| Enhance | P04 / V2-Lite graph+DBO, interleaved layout | Same enhancement while retaining the interleaved communication boundary | 4 GPU, two pods; first implement evidence transport and coordination |
| Enhance | P06 / Qwen3 DeepEP graph, role-split pods | Add per-A/F-rank cross-pod graph proof to current GSM8K result acceptance | 4 GPU, two pods; first implement evidence transport and coordination |
| Enhance | P07 / Qwen3 DeepEP graph+DBO, role-split pods | Add matching cross-pod two-stage layouts and consecutive graph replay | 4 GPU, two pods; first implement evidence transport and coordination |

<a id="full-target-matrix"></a>

### Complete target list (58 entries when conditions hold)

<details>
<summary>Expand 37 single-host GPU, 12 single-host NPU and 9 multi-pod entries</summary>

Retain G02 and assume G17/G37 removal conditions hold. Otherwise bring the corresponding candidate back from A.

#### GPU single-host: 37 cases

| ID / target scenario | Model, deployment and modes | Target coverage | Scheduling proposal | Origin and change |
| --- | --- | --- | --- | --- |
| G01 / `baseline-graph` | DeepSeek-V2-Lite; 4 GPU; native DP4×TP1; graph | GSM8K, 7 questions | Retain ready/merge | Preserve coverage; migrate shared framework |
| G02 / `afd-v2-eager-1a1f` | DeepSeek-V2-Lite; 2 GPU; A/F DP1×TP1; eager | GSM8K, 7 questions | Manual/diagnostic; affected-change trigger | Retain the minimal two-device full-service eager path |
| G03 / `afd-v2-eager-dp2` | DeepSeek-V2-Lite; 4 GPU; A/F DP2×TP1; eager | GSM8K, 128 questions / 12 concurrent requests | Retain ready/merge | Preserve coverage; migrate shared framework |
| G04 / `afd-v2-eager-tp2` | DeepSeek-V2-Lite; 4 GPU; A/F DP1×TP2; eager | GSM8K, 7 questions | Retain ready/merge | Preserve coverage; migrate shared framework |
| G05 / `afd-v2-graph-1a1f` | DeepSeek-V2-Lite; 2 GPU; A/F DP1×TP1; graph | GSM8K, 7 questions; per-rank graph | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G06 / `afd-v2-graph-dp2` | DeepSeek-V2-Lite; 4 GPU; A/F DP2×TP1; graph | GSM8K, 7 questions; per-rank graph | Retain ready/merge | Preserve coverage; migrate shared framework |
| G07 / `afd-v2-graph-tp2` | DeepSeek-V2-Lite; 4 GPU; A/F DP1×TP2; graph | GSM8K, 7 questions; per-rank graph | Retain ready/merge | Preserve coverage; migrate shared framework |
| G08 / `afd-v2-eager-dbo-dp2` | DeepSeek-V2-Lite; 4 GPU; A/F DP2×TP1; eager+DBO | GSM8K, 128 questions / 12 concurrent requests; per-rank two-stage layouts | Retain ready/merge | Preserve coverage; migrate shared framework |
| G09 / `afd-v2-graph-dbo-dp2` | DeepSeek-V2-Lite; 4 GPU; A/F DP2×TP1; graph+DBO | GSM8K, 128 questions / 12 concurrent requests; per-rank consecutive graph replay + two-stage layouts | Retain ready/merge | Preserve coverage; migrate shared framework |
| G10 / `afd-v2-graph-dbo-tp2` | DeepSeek-V2-Lite; 4 GPU; A/F DP1×TP2; graph+DBO | GSM8K, 24 questions / 12 concurrent requests; per-rank consecutive graph replay + two-stage layouts | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G11 / `afd-eager-2a1f` | DeepSeek-V2-Lite; 3 GPU; A DP2×TP1 / F DP1×TP1; eager | GSM8K, 7 questions | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G12 / `afd-graph-2a1f` | DeepSeek-V2-Lite; 3 GPU; A DP2×TP1 / F DP1×TP1; graph | GSM8K, 7 questions; per-rank graph | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G13 / `afd-graph-dbo-2a1f` | DeepSeek-V2-Lite; 3 GPU; A DP2×TP1 / F DP1×TP1; graph+DBO | GSM8K, 24 questions / 12 concurrent requests; per-rank consecutive graph replay + two-stage layouts | Retain weekly | Preserve coverage; migrate shared framework |
| G14 / `afd-deepep-m2n-eager-2a2f` | DeepSeek-V2-Lite; 4 GPU; A/F DP2×TP1; eager | GSM8K, 7 questions | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G15 / `afd-deepep-m2n-graph-2a2f` | DeepSeek-V2-Lite; 4 GPU; A/F DP2×TP1; graph | GSM8K, 7 questions; per-rank graph | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G16 / `afd-deepep-m2n-graph-dbo-2a2f` | DeepSeek-V2-Lite; 4 GPU; A/F DP2×TP1; graph+DBO | GSM8K, 24 questions / 12 concurrent requests; per-rank consecutive graph replay + two-stage layouts | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G18 / `afd-deepep-m2n-graph-tp2-2a2f` | DeepSeek-V2-Lite; 4 GPU; A DP1×TP2 / F DP2×TP1; graph | GSM8K, 7 questions; per-rank graph | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G19 / `baseline-dsv4-flash-eager` | DeepSeek-V4-Flash; 2 GPU; native DP2×TP1; eager | Two concurrent answer rounds + GSM8K, 7 questions | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G20 / `baseline-dsv4-flash-graph` | DeepSeek-V4-Flash; 2 GPU; native DP2×TP1; graph | Two concurrent answer rounds + GSM8K, 7 questions; per-rank graph | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G21 / `afd-dsv4-flash-eager-2a2f` | DeepSeek-V4-Flash; 4 GPU; A/F DP2×TP1; eager | Two concurrent answer rounds + GSM8K, 7 questions; Hash pairing/freshness | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G22 / `afd-dsv4-flash-graph-2a2f` | DeepSeek-V4-Flash; 4 GPU; A/F DP2×TP1; graph | Two concurrent answer rounds + GSM8K, 7 questions; Hash pairing/freshness; per-rank graph | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G23 / `baseline-dsv4-flash-eager-dbo` | DeepSeek-V4-Flash; 2 GPU; native DP2×TP1; eager+DBO | Two concurrent answer rounds + GSM8K, 24 questions; per-rank two-stage layouts | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G24 / `baseline-dsv4-flash-graph-dbo` | DeepSeek-V4-Flash; 2 GPU; native DP2×TP1; graph+DBO | Two concurrent answer rounds + GSM8K, 24 questions; per-rank consecutive graph replay + two-stage layouts | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G25 / `afd-dsv4-flash-eager-dbo-2a2f` | DeepSeek-V4-Flash; 4 GPU; A/F DP2×TP1; eager+DBO | Two concurrent answer rounds + GSM8K, 24 questions; Hash pairing/freshness; per-rank two-stage layouts | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G26 / `afd-dsv4-flash-graph-dbo-2a2f` | DeepSeek-V4-Flash; 4 GPU; A/F DP2×TP1; graph+DBO | Two concurrent answer rounds + GSM8K, 24 questions; Hash pairing/freshness; per-rank consecutive graph replay + two-stage layouts | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G27 / `baseline-graph` | Qwen3.6; 4 GPU; native DP4×TP1; graph | GSM8K, 7 questions | Retain weekly | Preserve coverage; migrate shared framework |
| G28 / `afd-eager-2a1f` | Qwen3.6; 3 GPU; A DP2×TP1 / F DP1×TP1; eager | GSM8K, 7 questions | Retain weekly | Preserve coverage; migrate shared framework |
| G29 / `afd-graph-2a1f` | Qwen3.6; 3 GPU; A DP2×TP1 / F DP1×TP1; graph | GSM8K, 7 questions; per-rank graph | Retain weekly | Preserve coverage; migrate shared framework |
| G30 / `baseline-graph` | Qwen3 MoE; 4 GPU; native DP4×TP1; graph | GSM8K, 7 questions | Retain weekly | Preserve coverage; migrate shared framework |
| G31 / `afd-eager-2a1f` | Qwen3 MoE; 3 GPU; A DP2×TP1 / F DP1×TP1; eager | GSM8K, 7 questions | Retain weekly | Preserve coverage; migrate shared framework |
| G32 / `afd-graph-2a1f` | Qwen3 MoE; 3 GPU; A DP2×TP1 / F DP1×TP1; graph | GSM8K, 7 questions; per-rank graph | Retain weekly | Preserve coverage; migrate shared framework |
| G33 / `afd-graph-dbo-2a1f` | Qwen3 MoE; 3 GPU; A DP2×TP1 / F DP1×TP1; graph+DBO | GSM8K, 24 questions / 12 concurrent requests; per-rank consecutive graph replay + two-stage layouts | Retain weekly | Preserve coverage; migrate shared framework |
| G34 / `afd-deepep-m2n-eager-2a2f` | Qwen3 MoE; 4 GPU; A/F DP2×TP1; eager | GSM8K, 7 questions | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G35 / `afd-deepep-m2n-graph-2a2f` | Qwen3 MoE; 4 GPU; A/F DP2×TP1; graph | GSM8K, 7 questions; per-rank graph | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G36 / `afd-deepep-m2n-graph-dbo-2a2f` | Qwen3 MoE; 4 GPU; A/F DP2×TP1; graph+DBO | GSM8K, 24 questions / 12 concurrent requests; per-rank consecutive graph replay + two-stage layouts | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G38 / `afd-deepep-m2n-graph-tp2-2a2f` | Qwen3 MoE; 4 GPU; A DP1×TP2 / F DP2×TP1; graph | GSM8K, 7 questions; per-rank graph | Targeted/local; not added to PR gate yet | Preserve coverage; migrate shared framework |
| G39 / `afd-v2-graph-3a2f` | DeepSeek-V2-Lite; up to 5 GPU; native DP4×TP1 reference → A DP3×TP1 / F DP2×TP1 graph target | Per-request comparison with this run's native reference; fixed-input serial/concurrent windows; target FFN-group participation and per-rank graph; strict cleanup of both runs | 5-GPU periodic/targeted task (proposed) | New; different FFN subgroup sizes within one run. |

#### NPU single-host: 12 cases

| ID / target scenario | Model, deployment and modes | Target coverage | Scheduling proposal | Origin and change |
| --- | --- | --- | --- | --- |
| N01 / `baseline-graph` | DeepSeek-V2-Lite; 4 NPU; native DP4×TP1; graph | GSM8K, 7 questions | Targeted NPU execution; no continuous CI | Preserve coverage; migrate shared framework |
| N02 / `afd-eager-2a2f` | DeepSeek-V2-Lite; 4 NPU; A/F DP2×TP1; eager | GSM8K, 7 questions | Targeted NPU execution; no continuous CI | Preserve coverage; migrate shared framework |
| N03 / `afd-graph-2a2f` | DeepSeek-V2-Lite; 4 NPU; A/F DP2×TP1; graph | GSM8K, 7 questions; per-rank graph | Targeted NPU execution; no continuous CI | Preserve coverage; migrate shared framework |
| N04 / `afd-graph-dbo-2a2f` | DeepSeek-V2-Lite; 4 NPU; A/F DP2×TP1; graph+DBO | GSM8K, 24 questions / 12 concurrent requests; per-rank consecutive graph replay + two-stage layouts | Targeted NPU execution; no continuous CI | Preserve coverage; migrate shared framework |
| N05 / `afd-eager-2a1f` | DeepSeek-V2-Lite; 3 NPU; A DP2×TP1 / F DP1×TP1; eager | GSM8K, 7 questions | Targeted NPU execution; no continuous CI | Preserve coverage; migrate shared framework |
| N06 / `afd-graph-2a1f` | DeepSeek-V2-Lite; 3 NPU; A DP2×TP1 / F DP1×TP1; graph | GSM8K, 7 questions; per-rank graph | Targeted NPU execution; no continuous CI | Preserve coverage; migrate shared framework |
| N07 / `afd-graph-dbo-2a1f` | DeepSeek-V2-Lite; 3 NPU; A DP2×TP1 / F DP1×TP1; graph+DBO | GSM8K, 24 questions / 12 concurrent requests; per-rank consecutive graph replay + two-stage layouts | Targeted NPU execution; no continuous CI | Preserve coverage; migrate shared framework |
| N08 / `afd-dsv4-flash-async-cam-dp2tp4-ep8` | DeepSeek-V4-Flash; 16 NPU; A DP2×TP4 / F DP8×TP1; A/F eager; Attention configured with DBO, activation not qualified | 10 concurrent answers with request overlap; does not qualify FFN FULL | Targeted NPU execution; no continuous CI | Preserve coverage; migrate shared framework |
| N09 / `afd-dsv4-flash-sync-camp2p-2a2f` | DeepSeek-V4-Flash; A5, 4 NPU; A/F DP2×TP1; graph, DBO disabled | 10 concurrent nonempty completions + per-rank graph; no answer validation | Targeted NPU execution; no continuous CI | Retain smoke scope; correctness requires separate qualification |
| N10 / `afd-dsv4-flash-sync-camp2p-8a8f` | DeepSeek-V4-Flash; A3, 16 NPU; A DP2×TP4 / F DP8×TP1; eager | 10 concurrent nonempty completions; no answer validation | Targeted NPU execution; no continuous CI | Retain smoke scope; correctness requires separate qualification |
| N11 / `afd-eager-dbo-2a2f` | DeepSeek-V2-Lite; 4 NPU; A/F DP2×TP1; pure eager+DBO | GSM8K, 24 questions / 12 concurrent requests, 0.25; live two-nonempty-stage execution and A/F rank layouts in a fixed prefill-heavy window; no decode DBO claim; liveness and cleanup | Targeted NPU task (proposed) | Expose existing manual configuration through a stable pytest entry; do not lower thresholds to force activation. |
| N12 / `afd-dsv4-flash-async-cam-ffn-full-4a4f` | DeepSeek-V4-Flash; 8 NPU; A DP1×TP4 / F DP4×TP1; A eager; F eager reference → FULL target | Multiple lengths, repeated identical inputs, real HTTP concurrency; per-FFN capture for this deployment, replay increments/mode per request window; eager/FULL token comparison; strict shutdown of both runs | 8-NPU periodic/targeted task (proposed) | New self-contained #509 comparison; shutdown is not currently qualified. |

#### multi-pod: 9 cases

| ID / target scenario | Model, deployment and modes | Target coverage | Scheduling proposal | Origin and change |
| --- | --- | --- | --- | --- |
| P01 / `afd-graph-2a2f-2pod-role-split` | DeepSeek-V2-Lite; 4 GPU; A DP2×TP1 / F DP2×TP1; role-split pods; graph | GSM8K, coordination and cleanup; named cross-pod windows and real graph replay on every A/F rank; liveness/cleanup | Targeted multi-pod execution; environment required | Enhance existing case with cross-pod evidence transport; no duplicate configuration |
| P02 / `afd-graph-2a2f-2pod-interleaved` | DeepSeek-V2-Lite; 4 GPU; A DP2×TP1 / F DP2×TP1; interleaved layout; graph | GSM8K, coordination and cleanup; named cross-pod windows and real graph replay on every A/F rank; liveness/cleanup | Targeted multi-pod execution; environment required | Enhance existing case with cross-pod evidence transport; no duplicate configuration |
| P03 / `afd-graph-dbo-2a2f-2pod-role-split` | DeepSeek-V2-Lite; 4 GPU; A DP2×TP1 / F DP2×TP1; role-split pods; graph+DBO | GSM8K, coordination and cleanup; named cross-pod windows, matching two-stage layouts and consecutive graph replay on every A/F rank; liveness/cleanup | Targeted multi-pod execution; environment required | Enhance existing case with cross-pod evidence transport; no duplicate configuration |
| P04 / `afd-graph-dbo-2a2f-2pod-interleaved` | DeepSeek-V2-Lite; 4 GPU; A DP2×TP1 / F DP2×TP1; interleaved layout; graph+DBO | GSM8K, coordination and cleanup; named cross-pod windows, matching two-stage layouts and consecutive graph replay on every A/F rank; liveness/cleanup | Targeted multi-pod execution; environment required | Enhance existing case with cross-pod evidence transport; no duplicate configuration |
| P05 / `afd-deepep-m2n-eager-2a2f-2pod-role-split` | Qwen3 MoE; 4 GPU; A DP2×TP1 / F DP2×TP1; role-split pods; eager | GSM8K, default 100 questions, coordination and cleanup; graph/DBO proof not required | Targeted multi-pod execution; environment required | Preserve coverage; migrate shared framework |
| P06 / `afd-deepep-m2n-graph-2a2f-2pod-role-split` | Qwen3 MoE; 4 GPU; A DP2×TP1 / F DP2×TP1; role-split pods; graph | GSM8K, default 100 questions, coordination and cleanup; named cross-pod windows and real graph replay on every A/F rank; liveness/cleanup | Targeted multi-pod execution; environment required | Enhance existing case with cross-pod evidence transport; no duplicate configuration |
| P07 / `afd-deepep-m2n-graph-dbo-2a2f-2pod-role-split` | Qwen3 MoE; 4 GPU; A DP2×TP1 / F DP2×TP1; role-split pods; graph+DBO | GSM8K, default 100 questions, coordination and cleanup; named cross-pod windows, matching two-stage layouts and consecutive graph replay on every A/F rank; liveness/cleanup | Targeted multi-pod execution; environment required | Enhance existing case with cross-pod evidence transport; no duplicate configuration |
| P08 / `2pod-role-split` | DeepSeek-V4-Flash; 16 NPU; A DP2×TP4 / F DP8×TP1; role-split pods; A/F eager; Attention configured with DBO, activation not qualified | 10 concurrent answers, coordination and cleanup; no graph/DBO execution-path proof | Targeted multi-pod execution; environment required | Preserve coverage; migrate shared framework |
| P09 / `2pod-interleaved` | DeepSeek-V4-Flash; 16 NPU; A DP2×TP4 / F DP8×TP1; interleaved layout; A/F eager; Attention configured with DBO, activation not qualified | 10 concurrent answers, coordination and cleanup; no graph/DBO execution-path proof | Targeted multi-pod execution; environment required | Preserve coverage; migrate shared framework |

</details>

<a id="new-case-acceptance"></a>

### Acceptance definitions for new scenarios

These are fixed implementation/qualification goals, with no new device-pass records from this work. Independently extend and validate missing task, comparison or evidence capabilities under the main RFC's rules before assessing a configuration-only addition.

#### G39: compare native reference and 3A2F graph per request

- **Deployments:** one test sequentially runs a native DP4×TP1 graph reference, reusing G01's deployment profile on four devices, then a P2P 3A2F graph target on five. Use separate directories and finish reference execution and strict shutdown before target startup. The reference does not use AFD grouping, avoiding shared eager/graph connector errors as the oracle.
- **Fixed inputs/generation:** first 10 GSM8K test questions rendered with the existing 8-shot template; retain complete prompts. Run a serial window followed by a window with 10 real concurrent HTTP clients. Temperature 0, seed 0, generation limit 512; fix checkpoint, tokenizer, quantization and generation settings. Record input lengths from this fixed set; do not select questions in response to run results.
- **Oracle:** associate this test's native output by window, request index, complete input and generation parameters. Compare full returned text and termination reason exactly per request. References must be complete, nonempty and distinguishable within the corresponding window. Incomplete/indistinguishable references fail qualification rather than fall back to aggregate GSM8K score or partial matching. This establishes AFD/native consistency, without a new model-accuracy commitment.
- **Execution/cleanup:** target grouping is F0 receiving A0/A1 and F1 receiving A2. Each target window proves real requests and graph replay on all three A ranks and both F ranks, not cumulative evidence from other windows. FFN participation and per-request oracles together validate grouping/returns. Both deployments use strict shutdown. Missing shared task/comparison capabilities mean the complete G39 work is not a configuration-only addition.

#### N11: pure-eager live DBO in a specified prefill-heavy workload

- **Deployment/thresholds:** 4 NPU CAMP2P, A/F DP2×TP1, pure eager, retaining native decode/prefill thresholds 32/512.
- **Windows/workload:** `live-dbo` uses four distinct new 1024-token inputs, four real HTTP clients and a generation limit of 64. Declare fixed inputs in the model task; record actual tokenized lengths and avoid prefix-cache reuse substituting for new input. Follow with an independent `gsm8k` window using N04's 24 questions, 12 concurrent requests, 8-shot prompting, generation limit 512 and threshold 0.25.
- **Judgment:** each required A/F rank in `live-dbo` has real requests with two nonempty microbatches and a matching layout across roles, while execution remains eager. Capture/profile/warmup do not count. GSM8K validates results separately; its cumulative counters cannot replace prior-window evidence.
- **Scope:** qualify live DBO under this specific workload, not independent qualification of both prefill DBO and decode DBO. Concurrency 4 or 12 alone is not activation evidence. Missing required layouts fail; do not lower thresholds to pass.

#### N12: separate deployment capture from request-window replay

- **Fixed scope:** existing single-stream 4A4F profile, 8 NPU, capacity 128; A eager, F eager reference → FULL target; generate 8 tokens with seed 1024. Directories, reference identity and failure propagation follow §6.3 of the main document.
- **Request windows:** input lengths 16/128/129/257 have separate windows. Two consecutive runs of the same length-128 input form the repeat window. Independent HTTP requests of lengths 16 and 257 form the concurrent window, with actual overlap checked. Replay identical inputs/settings in both deployments and compare token IDs exactly per request.
- **Capture:** all expected FFN ranks establish their expected graphs in this FULL deployment, preserving the existing one-capture-per-rank requirement. Do not require recapture in each request window.
- **Replay/mode:** take independent before/after snapshots for every listed window. All required FFN ranks show real-request replay increments in effective FULL mode without unexpected fallback; Attention keeps the expected eager/MRV2 path. Earlier cumulative replay does not qualify a later window.
- **Shutdown:** eager and FULL independently complete strict shutdown and reclamation; either run failing fails the whole case. L01 forced-cleanup success is not this qualification. Qualify 4A8F, W8A8 and two-stream modes separately.

### Existing enhancements and scope limits

- **P01–P04, P06–P07:** six graph/DBO scenarios gain named cross-pod windows and per-rank evidence. Current [multi-pod runner](../../tests/e2e/multi_pod/runner.py) disables MRV2 evidence because marker files are not shared across pods. Implement real transport and coordination. P05/P08/P09 retain eager result acceptance, without automatic graph/DBO claims.
- **N09/N10:** preserve accurate smoke scope. N09 also requires graph evidence today; neither validates answers. Current configuration comments report unreliable A5 answers and A3 lacking answer-oracle validation, not a single confirmed defect on both. Resolve known problems and qualify correctness independently, upgrading existing cases rather than duplicating weak smoke configurations.
- **Two-stream FFN:** if still a support commitment, add a separately reproducible regression. L05 remains follow-up work and is not implied by single-stream N12. 4A8F/W8A8 also need independent target-device qualification and are outside N12 defaults.

<a id="other-entries"></a>

## C. Independent entries, device-specific tests and host checks

Do not omit these entries because they fall outside the 57 parametrized service scenarios. Do not count them as model E2E cases.

| ID / current entry | Current coverage and limits | Proposed action / destination | Counting |
| --- | --- | --- | --- |
| L01 / [ffn_graph.py](../../tests/e2e/models/deepseek_v4_flash/ffn_graph.py) | Single stream; A eager, F eager/FULL; 4A4F or optional 4A8F; multiple lengths, two-prompt batches, token comparison and per-rank evidence; no graceful-shutdown qualification | Reuse in N12 self-contained comparison. Move duplicated deployment code while preserving distinct proof. 4A8F still needs device qualification; decide old CLI retirement after callers migrate | Independent driver configurations; no current pytest/CI entry; outside current 57 |
| L02 / [dspark_eager.py](../../tests/e2e/models/deepseek_v4_flash/dspark_eager.py) | Existing-service length, batch, real concurrency, stop, stream abort followed by another request, and MTP activity; optional FFN/per-layer numerical proof; PD variants | Retain model-specific tasks/checks callable from scenarios. Keep CLI until deployment callers migrate. No numerical claim when that proof is not selected | Workload tool, without service startup/shutdown |
| L03 / [async_cam_roundtrip.py](../../tests/e2e/operators/async_cam_roundtrip.py) | TP1/2/4 × FP16/BF16 × quant0/1: 12 base configurations; boundaries, empty ranks and repeated windows | Retain independent operator/communication numerical acceptance, not 12 duplicate full-model cases | Device-specific operator/communication coverage, outside 57 |
| L04 / [manual NPU eager DBO configuration](../../tests/e2e/runner.py) | Existing `afd-eager-dbo-2a2f` configuration and native thresholds without pytest parameter | Expose and validate through N11; missing pytest callers alone do not establish dead code | Automation entry is new; deployment mode exists |
| L05 / [two-stream qualification notes](../npu/ASYNC_FFN_RUNTIME.md#validation) | Current real qualification depends on an external driver; L01 in this repository is single-stream | Add reproducibility according to support commitments; separate follow-up, not part of N12 | Future device coverage, outside target 58 |
| D01 / [DeepEP GPU tests](../../tests/gpu/test_deepep_m2n_gpu.py) | Real TP slicing, transport, expert results, eager/graph, two lanes, mixed modes and shutdown; insufficient hardware skips; not selected by current CI | Obtain zero-skip records for the [D01-TP2-4GPU manual task](#d01-manual-task) and execution ownership before removing G17/G37 | Lighter device regression, not a full model-service E2E or newly written test |
| H01 / [DSpark evidence checker](../../tests/e2e/models/deepseek_v4_flash/test_dspark_graph_checker.py) | 19 host checks, including missing ranks/layers and numerical-proof negative inputs | Preserve effective rejection of invalid evidence; not model/GPU/NPU pass evidence | 19 host checks |
| H02 / [deployment-script checks](../../tests/e2e/test_deploy_dsv4_flash_async_skill.py) | 16 host integration checks for CLI, environment, configuration and result handling | Preserve effective contracts through deployment-interface migration; not PD model-correctness acceptance | 16 host checks |
| H03 / [runner cancellation checks](../../tests/e2e/test_runner.py) | 2 real local-process tests covering blocked requests and SIGINT/SIGTERM cancellation | Preserve host integration coverage of ownership, cancellation and reclamation | 2 host checks |

H01 negative checks complement normal device execution. Passing normal E2E does not justify removing the checker's rejection of invalid proof.

<a id="code-consolidation"></a>

## D. Consolidating maintenance code is not scenario removal

| Current duplication | Target owner | Differences to preserve |
| --- | --- | --- |
| Five model-entry `_required_env` definitions | [environment.required_env](../../tests/e2e/environment.py) | Current bodies are identical; remove duplicate definitions |
| Shared asset preparation and command assembly in V2-Lite, Qwen3 and Qwen3.6 | Model scenario settings plus shared entry/deployment | Model, platform, device layout, arguments, GSM8K settings and override precedence |
| Dedicated-driver startup, waiting, cancellation and cleanup | Shared deployment and existing process tools | Limited NPU cleanup exceptions and whether graceful shutdown is qualified |
| Identical phase/window organization | Shared execution and host checks | Different MRV2, FFN FULL and DSpark probes and judgments |

Consolidated parameter tables or fewer functions/files do not necessarily reduce executed device combinations or GPU time. Track coverage and maintenance-code changes separately rather than treating deleted lines as benefits.

<a id="maintenance"></a>

## E. Decisions and ongoing maintenance

1. Confirm retention, enhancements and conditional removal per row, including actual substitute execution ownership. Recommendations are not validated conclusions.
2. Review trimming and replacement validation together. Unsupported hardware, unexecuted substitutes or skips cannot silently retire original coverage.
3. Current gates, samples, thresholds, per-rank/window requirements and cleanup-failure semantics remain valid. Deliberate changes to support or acceptance need explicit justification.
4. Determine resources/frequency for new hardware coverage separately. Framework reorganization does not implicitly authorize new resource commitments.
5. Check destinations against this appendix in every batch. The E2E README owns the ongoing list/run instructions and the module contract owns behavioral constraints. This appendix retains the design mapping without raw execution logs.

### Scenario-group ownership and triggers

Affected-change PR authors arrange validation and supply records, agreeing on scope with model/backend/connector maintainers using this table. Do not repeat identical validation already supplied by CI, run the full matrix on every change, or create pipelines merely because a group is listed here.

| Scenario group | Maintenance owner | Trigger and required scope | Execution category |
| --- | --- | --- | --- |
| GPU P2P: G01–G13, G27–G33, G39 | GPU connector, corresponding model, affected PR author | P2P grouping/transport, role TP, graph/DBO or integration changes; select affected modes/models. G02 diagnoses minimal eager; G39 covers unequal groups | Retain relevant selections among 15 existing unique CI node IDs; other affected changes run manually; G02 also diagnostic |
| DeepEP: G14–G18, G34–G38, D01 | DeepEP connector, corresponding model, affected PR author | TP slicing, dispatch/combine, payload, expert integration or dependency revision; trigger D01 and relevant retained model services | Affected-change manual runs; D01 records are removal conditions |
| DSV4 CUDA: G19–G26 | DSV4 GPU model and affected PR author | Hash input, weights/quantization, graph/DBO or runtime upgrades; select relevant native/AFD comparisons and all applicable windows | Affected-change manual; periodic resources settled separately |
| NPU V2-Lite: N01–N07, N11 | NPU connector/runner, model, affected PR author | CAMP2P, native ubatching, graph, cleanup or dependency upgrades; select affected eager/graph/DBO and topologies | Affected-change manual; N11 remains pending until qualified |
| DSV4 NPU: N08–N10, N12, L01/L02 | NPU model, async/sync connector, affected PR author | Model, FFN mode, layered/DSpark, shutdown or dependency upgrades; run affected profiles, preserving N09/N10 limited smoke | Affected-change manual; N12 qualification is independent of old-driver passes |
| Multi-pod: P01–P09 | Coordination/deployment, corresponding connector, affected PR author | Address, barrier, failure notification, placement or cleanup protocol changes; cover affected role-split/interleaved layouts and backends | Affected-change manual; six path enhancements need separate device qualification |
| Other independent device entries and host checks | Corresponding tool/probe maintainer and affected PR author | Relevant protocol, workload, checker or script-interface changes; validate C's actual purposes without counting them as model results | Affected checks; reuse existing host integration |

For shared deployment/cleanup changes spanning groups, select lifecycle validation by affected backend/connector. Do not extrapolate across backends. Missing five-device, eight-device or cross-pod allocations leave corresponding entries unvalidated with scope explained, not passed.

### Records and coverage reassessment

The durable README records execution category, triggers, ownership and qualification links by scenario group. Distinguish defined, qualified-for-a-version/hardware, and validated-for-this-change records. Include code/runtime versions, hardware, commands, canonical/effective settings, results and artifact links; raw logs remain task artifacts. Weakened debug settings, skips and other-platform results do not qualify a scenario.

If substitute workloads/checks weaken or stop running, or a trimmed combination has a specific regression, the affected PR revisits replacement relationships and restores service coverage when needed. Explain actual maintenance, execution or diagnostic savings; case/line deletion counts are not a substitute.

## Sources

- Current AFD entries, configuration and CI: [E2E README](../../tests/e2e/README.md), [ready](../../.buildkite/cuda/test-ready.yml), [merge](../../.buildkite/cuda/test-merge.yml), [weekly](../../.buildkite/cuda/test-weekly.yml).
- [P2P grouping and role ranks](../../afd_plugin/distributed/topology.py), [multi-peer inputs and result splitting](../../afd_plugin/connectors/gpu/p2p.py): the value of 2A1F and non-integral groups.
- [GPU ubatching compatibility](../../afd_plugin/compat/patches/gpu_mrv2_ubatching.py): G10 covers DP1+DBO; lack of CI selection does not make it redundant.
- [DeepSeek Attention routing/shared experts](../../afd_plugin/model_executor/models/deepseek_v2.py) and [Qwen3 integration](../../afd_plugin/model_executor/models/qwen3_moe.py): reasons to retain representatives for each model.
- [Current A5/A3 sync profiles](../../tests/e2e/models/deepseek_v4_flash/config.py) and [runner](../../tests/e2e/runner.py): N09/N10 answer and graph-evidence scope.
- [Pinned vLLM graph modes](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/compilation.py#L53): `FULL_DECODE_ONLY` uses `NONE` for mixed mode. This neither disables all `torch.compile` nor covers every pure-eager behavior.
