# AFD E2E Test Framework Design and Maintenance

[简体中文](e2e_test_framework.md) | English

Status: RFC draft. This document describes a proposed structure that has not been implemented. Example interfaces are not currently executable APIs. Existing commands and acceptance criteria remain defined by the [E2E README](../../tests/e2e/README.md) and the [E2E module contract](module/e2e_testing.md).

Investigation baseline: AFD `main@e778192d4c1cfd47e97be1881047532bfce7a7c0` (2026-11-05). This is the English translation of the proposal published at `2209c5ca5ada5056abfa69c59c825ba3e7340240`; it introduces no design changes. Reference revisions and source evidence appear below.

Reading guide: §3.1 shows module ownership; §5.2 shows execution order. The appendix provides the [current scenarios](e2e_case_matrix.en.md#current-matrix), [removal conditions](e2e_case_matrix.en.md#deletion-gates), [proposed changes](e2e_case_matrix.en.md#target-matrix), and the [folded complete target list](e2e_case_matrix.en.md#full-target-matrix). Additions and removals have not been implemented. The Chinese document retains the publication filename `e2e_test_framework.md`; this English counterpart uses `e2e_test_framework.en.md`.

For a new configuration expressible with existing capabilities, the maintenance workflow changes as follows:

| Work | Current workflow | Proposed workflow |
| --- | --- | --- |
| Define a combination | Model `test_*.py` selects IDs, devices and commands; `runner.configure_scenario` then sets layout and modes by ID | Model `cases.py` declares the configuration and exposes IDs; the existing parametrized entry reads the IDs |
| Select tasks and checks | Edit model branches, evaluation selection and evidence conditions in the runner as needed | The Case selects existing tasks and checks; shared-module changes must be empty |
| Schedule execution | Select CI node IDs and configure resources as needed | CI continues to select cases and allocate resources without changing acceptance criteria |

Only a new task, check, deployment protocol or connector capability justifies extending its corresponding module, with validation. A capability extension must not be presented as a configuration-only addition.

## 1. Goals and scope

The change should let maintainers understand a complete scenario in its model directory and use shared capabilities to add and maintain cases:

- A new combination supported by existing capabilities changes only model declarations, ID lists, and necessary CI selections and documentation. The incremental diff of shared modules must be empty, verified against the state before the addition.
- A new request task, check, deployment protocol or connector capability may extend the corresponding shared module and must include relevant validation. An independent framework defect is reviewed explicitly as a fix, not counted as a configuration-only addition.
- Preserve effective coverage for behavior that remains needed. Remove retired behavior together with its dedicated tests and helpers.
- CI and local execution use the same case definitions. CI selects cases, supplies resources and sets frequency.

E2E means starting real services, running real workloads, validating results and cleaning up resources created by that run. Operator tests, performance platforms and PD/proxy deployment tools do not automatically become unified-framework cases through this cleanup. Continue maintaining these tools for their actual purposes.

Scope includes existing GPU/NPU, baseline/AFD, A/F layouts, graph/DBO, model-specific acceptance and multi-pod entries, migrated in batches. This document does not expand model support commitments, lower current acceptance strength or introduce a scheduling platform.

## 2. Current problems

AFD already has shared runners, evaluation tools and process management. The main maintenance cost is scattered scenario information.

| Observed fact | Source | Direction |
| --- | --- | --- |
| Model entries list scenarios, divide devices, assemble commands and adjust evaluation parameters | [DeepSeek-V2-Lite entry](../../tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py) | Keep scenario meaning together in the model's scenario module |
| Scenario names determine topology, TP, graph, DBO, backend restrictions and model configuration | `configure_scenario` and `validate_topology` in [runner.py](../../tests/e2e/runner.py) | Shared code consumes explicit configuration and protocol requirements |
| The generic runner imports DSV4 configuration and evaluation functions | Imports, `build_vllm_command` and `main` in the same file | Model modules supply configuration and functions to shared execution |
| DSV4 GPU's two concurrent rounds and GSM8K form three stages organized directly in `main`, bypassing `run_scenario_evaluation` | `main` in [runner.py](../../tests/e2e/runner.py), baseline lines 227–249 | Models declare phases and cross-window requirements; the runner executes those declarations |
| Cleanup policy, timeouts and evaluation selection still depend on scenario names | `uses_npu_async_process_cleanup` and `run_scenario_evaluation` in the same file | Make lifecycle and acceptance ownership explicit |
| MRV2, FFN graph and DSpark each handle evidence separately | [MRV2](../../tests/e2e/mrv2_evidence.py), [FFN graph](../../tests/e2e/models/deepseek_v4_flash/ffn_graph.py), [DSpark](../../tests/e2e/models/deepseek_v4_flash/dspark_eager.py) | Share identical procedures while preserving probes and assertions with different meanings |

These are observed organization problems. They do not by themselves prove that an existing scenario is wrong or justify removing it.

The current single-host flow below is organized by `runner.py`:

```text
Model test_*.py: select scenario, prepare assets, assemble runner command
    → tests/conftest.py::run_runner(): launch subprocess
    → runner.main(): configure by name, validate, prepare probes, select A/F launch order
    → Start services and wait for readiness
    → Branch: DSV4 GPU → two concurrent rounds with window checks
                          → cross-round checks → GSM8K with window checks
              Other cases → evaluation window → run_scenario_evaluation()
    → Liveness check → select cleanup policy by scenario → report result
```

## 3. Target architecture and code ownership

### 3.1 Module collaboration

This diagram shows collaboration, not execution order. Each connection names a call or parameter input. Paths are relative to `tests/e2e/`, except `tests/conftest.py`. An asterisk marks a proposed new file.

```text
pytest process
models/<model>/test_*.py
    └── calls ──▶ tests/conftest.py::run_runner()
                       │
                       └── launches independent subprocess
                                  │
                                  ▼
runner subprocess
models/<model>/cases.py*
    │
    │ supplies case builder to the shared entry
    ▼
runner.py
    ├── reads ──▶ environment.py
    ├── calls ──▶ injected builder, obtains Case
    ├── calls ──▶ deployment.py*
    │               └── uses process_utils.py, cancellation.py
    ├── calls ──▶ tasks selected by Case
    │               └── e.g. accuracy/gsm8k.py, model-specific tasks
    └── calls ──▶ checks selected by Case
                    └── e.g. checks/mrv2.py*, model-specific checks

case.py* defines the scenario structures exchanged by these modules.
Worker-side probes run inside the tested services and expose evidence through RPC/logs.
```

The shared runner calls a model-supplied builder instead of importing a particular model's configuration module. Tasks and checks arrive through the Case; the shared runner does not select them by model name or case ID.

### 3.2 Module responsibilities

| Module | Status | Responsibilities and boundaries |
| --- | --- | --- |
| `case.py` | New | Data structures exchanged by scenarios, deployment and evaluation phases; no service startup or concrete model imports |
| `models/<model>/cases.py` | New | Safely importable stable IDs; complete Case construction only in the runner subprocess; the model's runtime entry |
| `models/<model>/test_*.py` | Adjust | Read IDs and parametrize during collection; call subprocess helpers during execution; retain existing pytest node IDs |
| `runner.py` | Simplify | Organize check preparation, deployment, phases, liveness and exit; propagate failures |
| `environment.py` | Extend | Read runtime inputs and resolve paths, devices, addresses, ports and executables; no scenario acceptance definitions |
| `deployment.py` | New | Validate topology, bind resources and build role commands; determine launch order from the connector protocol table, apply `cleanup_profile`, wait for readiness, check liveness and reclaim services |
| `accuracy/gsm8k.py` | Extend | Run GSM8K, parse results, validate exit status, sample count and accuracy; consume explicit evaluation settings |
| `checks/mrv2.py` | New | Extract current host-side probe configuration, window control, evidence retrieval and graph/DBO/Hash checks |
| `mrv2_evidence.py` | Retain | Worker-side MRV2 probes, separate from host control code |
| Model-specific configuration, tasks, checks and probes | Retain as needed | Model-specific logic stays in the model directory; models need not create identical file sets |
| `multi_pod/` | Retain and integrate incrementally | Layout, address discovery, barriers and failure notification; local services managed through deployment executors |
| `process_utils.py`, `cancellation.py` | Retain | Low-level process-group termination, reaping and cancellation; no model decisions |
| `tests/conftest.py` | Retain | Runner subprocess and signal forwarding, distinct from deployment's tested-service ownership |

Deployment owns service startup and release; no overlapping lifecycle module is needed. Multi-pod retains existing coordination. Each pod's deployment executor is the sole owner of its local processes and reclaims only its own resources. The current local executor may become a deployment adapter. Coordination must not run a second cleanup path; `cleanup_profile` is the single policy source.

## 4. Scenarios, environments and model configuration

### 4.1 What a Case expresses

A Case includes at least the following. Concrete type names will be settled through the pilot.

| Content | Meaning |
| --- | --- |
| Stable ID and coverage purpose | Behavior requiring the case; ID remains stable within the model/backend entry |
| Applicable model and backend | Fixed model requirements, quantization, tokenizer and other prerequisites |
| Deployment description | Baseline or AFD, per-role DP/TP layouts, connector and its configuration |
| Cleanup description | Deployment explicitly declares `cleanup_profile` and shutdown wait budget; default `strict`, with limited exceptions in §5.5 |
| Per-role execution modes | Actual eager, graph and DBO settings; A/F modes may differ |
| Ordered evaluation phases | Phase names, task parameters, result requirements and required execution evidence |
| Restrictions and acceptance scope | Resources, existing platform limits, and conclusions such as smoke-only, accuracy or execution-path acceptance |

In current AFD layouts, per-role device needs follow DP/TP and related settings. Validation preserves effective backend/connector rules. Do not maintain two ambiguous sources for role counts and total rank counts.

Scenario names are for selection and reporting. Shared code may branch on backend, role, deployment shape or connector protocol, rather than infer them from a case ID.

`deployment.py` owns role launch order through its connector protocol table. A Case selects the connector and has no scenario-specific launch-order override. Preserve current protocol orders during migration:

| Deployment / connector | Role launch order |
| --- | --- |
| Native baseline | baseline |
| GPU `P2pNcclAFDConnector`, `DeepEPM2NAFDConnector`; NPU `CAMP2pAFDConnector` | FFN → Attention |
| NPU `CAMAsyncAFDConnector` | Attention → FFN |

Order means process launch, not that the first role must finish full service readiness. On one host, start both roles before waiting for the public service. Multi-pod reuses the protocol decision and preserves the existing cross-pod barrier after the first role kind launches. Do not duplicate the order decision in each entry.

### 4.2 What the environment supplies

The environment supplies actual model paths, allocated devices, network addresses, ports, executables and output directories. Scenarios declare resource needs and acceptance settings; parsed environment inputs bind actual values.

Reuse applicable settings across NPU hardware and synchronous/asynchronous connectors while retaining necessary differences. Fixed model parameters stay in the model module; centralize settings shared by its scenarios.

Explicitly migrate local debugging-variable precedence, sample floors and concurrency requirements. CI must not silently weaken acceptance through environment variables. Run output must show effective configuration.

The Case holds canonical acceptance settings; run records hold effective settings and all overrides. Record task success separately from qualification for the fixed scenario. Runs with lower sample counts, thresholds, required concurrency, check strength or shutdown requirements are debugging runs and cannot qualify migration, coverage or replacement-based removal. Apply explicitly allowed rules to changes such as larger samples. Paths, devices, addresses and ports may change without affecting qualification when model/resource requirements still hold. A 24-question debugging run does not qualify a 128-question scenario's migration.

Compose startup arguments according to ownership below, not CLI append order or last-value-wins behavior. Role/common extra arguments and configuration files follow the same ownership rules. Incompatible conflicts fail before startup. Allowed overrides need explicit rules and recorded effective values.

| Parameter | Single semantic owner | Composition and override rule |
| --- | --- | --- |
| Topology, connector, per-role graph/DBO modes, `cleanup_profile` | Case deployment description | Deployment generates commands; environment and extra CLI cannot silently rewrite them |
| Fixed model settings, quantization, tokenizer | Model profile and explicit scenario variants | Environment binds a qualifying checkpoint; variants are declared in the scenario |
| Worker extension and probe startup arguments | Combined selected-check requirements and model extension requirements | Inject each probe once; incompatible classes/settings fail before startup, not overwrite one another |
| Actual devices, addresses, ports and executable | Parsed environment | Bind scenario requirements without changing layout or acceptance scope |
| Requests, generation parameters, sample counts, thresholds, required evidence | Case phases and tasks | Explicit override rules only; weakened debugging settings cannot qualify a fixed scenario |

### 4.3 Passing scenarios between pytest and runner

Keep the independent subprocess boundary:

1. The model module separately exposes a safely importable immutable ID list and `build_case(case_id, environment)`. Collection does not call the builder.
2. pytest collection reads IDs and parametrizes using the explicit backend only. It does not resolve checkpoint paths, detect quantization, download assets, initialize devices or start services.
3. `run_runner` launches the corresponding model scenario module, passing only the ID, environment and required runtime options.
4. The model module calls the shared CLI entry and injects `build_case`; the shared entry parses the environment and then calls it.
5. The completed Case is passed to `run_case` inside the subprocess.

This avoids serializing Python callables and a generic-runner registry importing every model. Actual checkpoint path resolution, model-configuration reads and quantization detection happen only in the runner subprocess. Assets are prepared during execution.

Collection safety covers the entire import chain. The ID module and its top-level imports must not indirectly perform the forbidden work. Moving only the `build_case()` call is insufficient. Review model-entry and shared `conftest.py` imports during migration.

Collection without runtime packages does not prove all import paths safe. Shared `conftest.py` currently imports `vllm_ascend.ops` when vLLM-Ascend is found. Review that installed-package branch and check collection in the corresponding environment for device-runtime loading, initialization and checkpoint reads. Do not suppress import or ABI failures.

This entry structure is proposed. Old commands, multi-pod entries and dedicated scripts have existing callers. Migrate callers and documentation together with their scenarios; moving functions alone does not justify deleting an entry.

## 5. Execution, checks and failures

### 5.1 Call contracts

Use a small set of explicit data structures and ordinary Python functions. Agree on inputs, outputs and resource ownership first; refine signatures through the pilot.

| Call | Contract |
| --- | --- |
| Model `build_case` | Receive ID and parsed environment, return complete Case; do not start services |
| Shared `run_case` | Receive Case and environment, organize execution, propagate failures; success cannot precede cleanup completion |
| Deployment context | Start and wait on entry, reclaim on exit; partial startup failure still cleans up created resources |
| Task function | Use public endpoints or explicitly supported evaluation interfaces, run real workloads and validate results |
| Execution-path check | Supply probe configuration, window control and evidence judgment; fail on missing required evidence |

Cases and templates hold configuration, not mutable counters, processes or probe state shared between cases. Create and release state per run.

Checks such as graph, DBO and Hash sharing a probe in one deployment prepare it once and sample once per phase. The shared execution flow owns reset, window start and window end. Probes produce evidence; checks interpret it without independently resetting counters or reopening windows. Incompatible startup requirements fail before services launch.

Each phase produces an independent snapshot that later samples cannot mutate. Associate it with the run, deployment, phase and role/rank. Multiple checks consume the same snapshot; cross-phase checks read completed-phase snapshots. Evidence from an earlier window, another deployment or capture/profile cannot substitute for the current real requests. No generic probe registry is required.

### 5.2 Execution order

These are runtime steps, separate from the module collaboration diagram:

```text
Parse environment, build and validate Case
    ↓
Prepare required probes, communication materials and role startup parameters
    ↓
Deployment starts services and waits for readiness
    ↓
Check liveness before evaluation
    ↓
For each phase:
    Begin this phase's observation window
      → Run request task and validate results
      → Collect and check this phase's evidence
      → Perform required cross-window checks
    ↓
Check liveness after evaluation
    ↓
Deployment closes services and verifies resource reclamation
    ↓
Release probe communication materials, temporary files and other run resources
    ↓
Report success
```

Failure or cancellation at any step enters cleanup. Preserve task/evidence failures and also report cleanup failures. Cleanup must not overwrite the original cause or turn cleanup failure into success.

### 5.3 Prepare probes before service startup

Current MRV2 evidence depends on worker extensions, role environments and marker files. A check cannot consist only of an assertion after requests:

- Prepare run-specific probe configuration and communication materials before startup according to selected checks.
- Inject the worker extension and role arguments when building commands.
- Reset and open named windows before requests; fetch and validate the matching evidence afterward.
- Reuse one probe configuration across checks, and release its communication materials after service exit.

Load probes only when required by acceptance, preserving current pure-eager and ordinary-baseline behavior. Keep the host-control/worker-observation protocol explicit. Do not change the production data plane for tests.

### 5.4 Result checks and execution-path checks

Tasks validate their own responses or accuracy. For example, GSM8K owns exit status, completeness, sample counts, NaN handling and accuracy thresholds; do not split these into generic plugins.

Independent execution-path checks share graph, DBO and Hash evidence across tasks:

- Preserve requirements for participating roles, ranks, real tokens, layouts and windows.
- Dummy, profile, capture, or an empty second microbatch cannot stand in for real request execution.
- Missing required checks, empty evidence, missing ranks or mismatched windows fail.
- Cross-phase checks may read earlier evidence and finish before the next phase begins.

Model-specific FFN graph and DSpark probes prove different things. Merge only code with identical semantics. Cases accurately declare scope; baseline graph configuration comparisons or existing smoke cases cannot automatically claim execution-path qualification.

### 5.5 Cleanup and cancellation

Preserve process ownership, signal forwarding, timeouts and reclamation boundaries. Escalating to SIGKILL remains a failure for ordinary gates.

Deployment explicitly selects `cleanup_profile`; neither runner nor deployment selects cleanup by scenario name:

| `cleanup_profile` | Applicability and success conditions |
| --- | --- |
| `strict` (default) | Ordinary gates and cases requiring graceful shutdown; SIGKILL escalation or unreclaimed resources fail. N12 for #509 uses this profile |
| `npu_async_cam_legacy` | Only migrate the existing FFN cleanup exception for NPU async CAM eager services. Preserve run/FFN-role markers, bounded scans, waiting and liveness checks. Existing bounded forced reclamation is allowed but does not prove graceful shutdown |

The model Case explicitly selects the existing exception; deployment validates its NPU/async-CAM applicability. Matching backend or connector alone must not enable it automatically. Other roles and cleanup failures are not exempt. Record the effective profile and shutdown result.

The old dedicated FFN graph entry does not qualify graceful shutdown. Keep that scope accurate until callers migrate. N12 must pass `strict`; the old forced-cleanup success does not qualify it.

## 6. Representative scenarios

### 6.1 DeepSeek-V2-Lite graph DP2

This illustrates the Case shape only. Classes, constants and parameters are proposed interfaces:

```python
Case(
    id="afd-v2-graph-dp2",
    model=DEEPSEEK_V2_LITE,
    deployment=AFDDeployment(
        backend="gpu",
        connector="P2pNcclAFDConnector",
        cleanup_profile="strict",
        attention=Role(dp=2, tp=1, execution="full_decode_graph"),
        ffn=Role(dp=2, tp=1, execution="full_decode_graph"),
    ),
    phases=(
        Phase(
            name="gsm8k",
            task=GSM8K(settings=DEEPSEEK_SMOKE_GSM8K),
            # Check also prepares/injects probes before startup and supplies windows.
            # Evidence judgment happens afterward.
            checks=(MRV2Graph(roles=("attention", "ffn")),),
        ),
    ),
)
```

Centralize model profiles and evaluation settings. Actual device IDs, paths and ports do not belong in this declaration. Here `DEEPSEEK_SMOKE_GSM8K` means this scenario's existing 7 samples, 8-shot prompting, 512 generated-token limit and 0.25 smoke threshold.

DBO comparison cases retain their existing 128 samples, 12 concurrent requests and accuracy requirements. Sharing the GSM8K task does not apply this smoke profile to them. The current contract defines full thresholds, floors and debug override rules.

### 6.2 DSV4 CUDA multi-phase scenarios

Preserve current order and acceptance:

| Phase | Task and result | Required evidence |
| --- | --- | --- |
| `concurrent-1` | First concurrent request round and answer validation | Applicable graph/DBO evidence for this window; A/F Hash input pairing for AFD |
| `concurrent-2` | Change input IDs while retaining required request shapes | Independent window evidence, Hash pairing and input freshness relative to round one |
| `gsm8k` | GSM8K results and accuracy | Applicable graph/DBO evidence for this window; do not impose concurrent-window bounded Hash sampling |

Check cross-round freshness after round two and before GSM8K. Round-one cumulative counts cannot stand in for round-two or GSM8K evidence.

This tests whether the design can express real multi-phase acceptance without putting request details back into the shared runner.

<a id="ffn-full-comparison"></a>

### 6.3 Async CAM FFN FULL comparison (issue #509)

[Issue #509](https://github.com/vllm-project/afd-plugin/issues/509) requires real-NPU FFN FULL regression coverage: output comparison, length and concurrency variation, capture/replay evidence and graceful shutdown. Deploy an eager reference, complete requests and release the services, then deploy FFN FULL with matching settings and replay the inputs for comparison.

One test completes both runs in its own output directory, without files or services left by a previous pytest case. A stable pytest node ID therefore represents combined acceptance; it need not correspond one-to-one to a single-deployment Case. The model pytest entry may invoke the runner twice in sequence using model-local helpers. The shared runner continues to handle a single deployment without an FFN FULL branch.

Eager and FULL use separate subdirectories. A reference must come from this test and record inputs, generation parameters, checkpoint and result-affecting configuration. FULL may consume it only after eager reaches successful termination and completes its required shutdown. Apart from predeclared FFN-mode and necessary probe differences, model, topology, capacity and request settings match; resource bindings such as ports/directories are recorded separately. Any failed subrun, incomplete reference, configuration mismatch or output mismatch fails the entire pytest node ID. The runner returns a single-run result and terminal status; model helpers compose them without a generic `CompositeCase`.

Capture and replay have different scope. This FULL deployment must establish expected graphs on all required FFN ranks. Each selected request window then proves replay increments from real requests, effective FULL mode and no unexpected fallback. Later windows can reuse graphs captured in this deployment, but cannot use earlier cumulative replay as their evidence. Both runs independently pass `strict` shutdown and reclamation. See the [new-case acceptance definitions](e2e_case_matrix.en.md#new-case-acceptance) for windows.

Existing [ffn_graph.py](../../tests/e2e/models/deepseek_v4_flash/ffn_graph.py) already supplies Attention eager / FFN eager or FULL, multiple input lengths, a two-prompt batch, per-rank capture/replay and FULL/eager token comparison. Reuse it, while preserving these gaps:

- A two-prompt batch is not concurrent HTTP clients, and sequential requests with different inputs do not specifically check repeated identical inputs. Add the missing workloads.
- Current sampling brackets the entire request group. Specify which length and concurrency windows independently prove FFN FULL execution.
- The script explicitly does not qualify graceful shutdown and allows forced cleanup records. Implement shutdown acceptance separately; existing `passed.json` alone cannot close the issue.
- Default 4A4F needs 8 NPUs; 4A8F needs 12 and still requires target-device qualification in the documentation. Neither belongs directly in the four-device PR gate.

Start with the existing single-stream 4A4F comparison. Qualify 4A8F, W8A8 and two-stream coverage separately according to actual support; single-stream results do not establish them.

## 7. Migration boundaries for existing capabilities

This table sets capability boundaries without freezing every configuration cross-product. The [scenario appendix](e2e_case_matrix.en.md#current-matrix) lists retention, enhancements and conditional removals; trimming a combination requires its replacement and execution conditions.

| Scenario or tool | Preserve during migration |
| --- | --- |
| DeepSeek-V2-Lite GPU/NPU | Effective entries, baselines, layouts, graph/DBO and existing evidence requirements for each backend |
| Qwen3 MoE | Existing baseline, eager, graph and graph+DBO acceptance |
| Qwen3.6 | Current valid scenarios; migration does not restore withdrawn GDN DBO support |
| DSV4 CUDA | Native/AFD comparisons, multiple request rounds, per-window execution evidence, Hash freshness and GSM8K |
| DSV4 NPU | Explicit scopes for async answer checking, sync smoke, FFN graph and layered/DSpark acceptance |
| Multi-pod | Existing layouts, coordination, results and per-pod reclamation; retain explicit evidence limits |
| Operator, deployment and performance tools | Independent capabilities still needed; assess framework inclusion by complete service-acceptance needs |

Current [multi_pod/runner.py](../../tests/e2e/multi_pod/runner.py) explicitly disables MRV2 evidence because single-host FFN marker files are invisible in other pods. Results validate the existing graph/DBO scenarios, but do not prove equivalent per-rank execution paths.

Cross-pod windows and evidence transport are new capabilities requiring design and device validation. Appendix P01–P04 and P06–P07 propose enhancing six graph/DBO scenarios. The other three eager scenarios retain their result-acceptance scopes. A Case requiring evidence must fail on deployments unable to provide it; deployment must not silently drop the check.

Remove dedicated runners only after their distinct workloads, assertions, platform limits and lifecycle acceptance have effective destinations. Shared-runner adoption or similar names do not establish replacement coverage.

## 8. Adding, changing and removing tests

| Maintenance work | Main change location | Required assessment |
| --- | --- | --- |
| New combination of supported capabilities | Only model declarations, ID lists, necessary CI selections and docs | Shared-module diff must be empty; explain independent value and reused capabilities |
| New model | Model configuration, scenario table and pytest entry | Fixed requirements, native/AFD comparisons and applicability of existing capabilities; do not put model differences into the shared runner |
| New task | Corresponding task and scenario declaration | Complete request flow, result oracle, timeout and failure propagation |
| New execution-path check | Corresponding check, necessary probe and declaration | Real-request observation and required roles, ranks and windows |
| New deployment or connector protocol | `deployment.py` protocol table, necessary environment support and scenario | Protocol owns launch order and cross-pod coordination; validate readiness, layout, failure reclamation and `cleanup_profile` |
| Retired behavior | Scenario, dedicated helpers, entry, CI and documentation | Whether behavior is removed and a substitute covers the same risk |

Avoid purposeless configuration Cartesian products. If existing parametrization, tasks, checks and protocols can express the case, shared modules must not change. This includes runner, scenario structures, environment, deployment, shared tasks/checks, probes, multi-pod protocols and process tools. A new case does not justify another case-ID branch.

When a new task, check, deployment protocol or connector capability is needed, identify the missing capability, corresponding module and validation results, and review the capability extension explicitly. Independent framework fixes similarly state their cause and validation. Neither is a configuration-only addition. Declare cleanup exceptions through `cleanup_profile` and launch order through the connector table, not rewritten policies in model entries.

Changing pass criteria or deleting coverage must explain how prior conclusions remain supported or why they are no longer needed. Missing environments, platform limits and temporary failures must not be hidden by skip/xfail.

Passing retained model cases and a device-specific test separately does not establish equivalent full-combination coverage. Accept the combination risk being retired. Revisit and, where needed, restore service coverage if substitutes weaken, stop running or miss a combination-specific regression. Retain G02's two-device pure-eager entry this round; G17/G37 remain conditional under the appendix.

Maintain case procedures in the E2E README and architecture/invariants in the module contract. Update these durable documents alongside implementation. This RFC appendix records the investigation baseline and change mapping; the ongoing scenario list belongs in the README.

## 9. CI, plugin boundaries and run artifacts

Prefer stable pytest entries, function names and parameter IDs. Ready/merge and weekly continue selecting explicit node IDs. CI resources match case requirements and do not change workloads or pass criteria.

Keep model-organized entries initially. New scenarios need no global case file or rebuilt CI. NPU CI, performance regression platforms and scheduling-platform unification are separate work.

For non-CI cases, affected-change PR authors arrange relevant validation under the [scenario-group maintenance rules](e2e_case_matrix.en.md#maintenance). Do not run the complete matrix on every change or add every scenario to CI. Distinguish defined, qualified, and validated-for-this-change records, including code/runtime versions, hardware, commands, effective settings, results and artifact links. Weakened debug runs do not advance qualification. Migrate durable lists, triggers and qualification links into the E2E README.

Framework work is limited to tests and documentation. Production `afd_plugin` must not depend on E2E code. Reorganization does not introduce vLLM/vLLM-Ascend production patches. Test probes continue through existing test extension interfaces.

Keep outputs, responses, evaluations and cleanup records in run directories or CI artifacts. Investigation logs, scratch scripts and raw output are not project documentation. The repository holds reusable instructions, behavioral contracts and necessary design.

## 10. Implementation order and acceptance

### 10.1 Batches

1. Use the [current scenario matrix and independent entries](e2e_case_matrix.en.md#current-matrix) as the baseline. Confirm dispositions and removal conditions, and identify callers and destinations for each batch.
2. Pilot migration with DeepSeek-V2-Lite baseline, eager DP2, graph DP2, graph+DBO DP2 and existing G07 graph TP2. Establish entry, deployment, task and evidence interfaces. Use #509's eager/FULL comparison to check model-local composition of deployments.
3. Check interfaces against DSV4 multi-phase acceptance and NPU-specific lifecycles, then migrate other effective scenarios and multi-pod callers.
4. Remove old branches, dedicated harnesses and unused helpers after effective replacements pass; update run/maintenance docs.

The pilot limits the first migration scope; existing required gates continue in full. No fixed PR count is prescribed. Each batch must be reviewable and verifiable; do not first unify every probe.

Each implementation PR lists destinations and exit conditions below. Report migration, additions/enhancements and removals separately:

| Batch | Destination | Required validation | Old-entry exit condition |
| --- | --- | --- | --- |
| V2-Lite migration pilot | Model Cases, shared execution/deployment, GSM8K/MRV2 | Collection/imports, command/config comparison, critical failure paths and real-device results for migrated scenarios | Preserve workloads, thresholds, evidence and cleanup; migrate callers; retain G02 |
| DSV4, other models and NPU | Model tasks/checks and deployment adapters | Relevant phases, model differences and NPU lifecycle; qualify N12 strict shutdown separately | Effective destinations for dedicated acceptance and CLI callers; unfinished additions are not reported as migrated |
| Multi-pod | Existing coordination plus per-pod deployment executors | Addresses, barriers, failure notification and cleanup for affected layouts; qualify six evidence enhancements separately | One process owner; replace current result acceptance; no cross-pod execution qualification until enhanced checks pass |
| Conditional removal | Retained model E2E plus D01 | Per-case removal conditions, valid records and residual-risk acceptance | Remove G17/G37 only when all conditions hold; otherwise retain |

Missing five-GPU, eight-NPU or cross-pod resources need not stop coverage-preserving migration. Additions remain unqualified. If N12 strict shutdown exposes a production defect, record and fix it separately. Do not weaken the profile to pass or silently expand framework migration into production changes.

### 10.2 What to validate

| Validation | Failure detected | Evidence boundary |
| --- | --- | --- |
| Collection safety | CPU collection without model assets/runtime dependencies plus installed-runtime import review and collection; select only applicable GPU/NPU IDs, without checkpoint reads, downloads or device-runtime loading | No Case construction, runner or service startup; absent-package results do not cover installed-package branches |
| Affected node IDs, commands and environments | Missing selections or changed layouts, arguments, environment/evaluation settings | Does not establish numerical, device-execution or shutdown correctness |
| Existing host process/signal integration | Startup failure, cancellation, timeout and child-reclamation regressions | Does not establish device behavior |
| Affected real-device E2E | Model results, real execution, complete rank/window evidence and service cleanup | Report actual backend, scenario and versions |
| Fixed addition exercise: G39 / V2-Lite graph 3A2F | Once reference/target execution, per-request comparison and evidence capabilities exist, adding the combination changes only declarations, ID lists and necessary CI/docs; shared-module diff is empty, and real five-GPU acceptance passes | Review and validate prerequisite extensions separately; the combined change is not configuration-only; G07 is a migration case |

Do not create mirror UT for the harness under `tests/unit`. Reuse host integration checks and, as needed, real subprocess/failure paths. New or changed production behavior follows UT module-behavior conventions separately.

Compare the G39 addition against a migrated-framework commit that already supplies the required capabilities, examining its full diff rather than runner alone. G39 compares a native reference with the 3A2F graph target per request; inputs and judgments are in the appendix. Missing task/check capabilities must first be reviewed and validated as extensions; do not report the mixed work as zero shared-module change. Five-GPU resources are separate from the four-device PR gate. Do not add another permanent case solely to demonstrate framework convenience.

The 57 valid service scenarios are not the raw collection count. Lite's independent DeepEP entry currently does not filter the NPU backend. Explain removal of such invalid IDs in collection comparison; it is not removal of valid coverage.

Validate critical failure paths with existing host integration or the smallest necessary real-subprocess paths, rather than mirror UT for data classes and internal helpers:

| Failure | Required observation |
| --- | --- |
| Required rank missing or check incomplete | Overall failure |
| Later phase receives earlier-phase evidence | Current phase cannot pass |
| Partial service startup failure | Reclaim created resources; preserve startup failure |
| Task and cleanup both fail | Preserve both causes; final failure |
| Either run of a comparison fails | Entire pytest node ID fails |

These host checks prove rejection of invalid outcomes and reclamation, not device graph, DBO or communication. Bounded best-effort evidence collection after task failure is an optional later improvement, not a pilot blocker. It cannot overwrite the original failure or delay cleanup.

Migration acceptance must confirm:

- Required scenarios remain selectable without omitted deployment, workload, threshold or check requirements.
- Result checks and execution evidence apply in the correct phases; missing required evidence fails.
- Failure/cancellation cleanup affects only owned resources, and cleanup failure cannot pass.
- Configuration-only additions have an empty shared-module diff; capability extensions touch their corresponding modules and include actual validation.
- Device results, host results, skips and unexecuted cases are separate; collection/static comparison is not device acceptance.

## 11. References and adaptations

Primary reference: vLLM-Ascend `18a1bd414b6609281c950206afa99aa8ac38227a`:

- [E2E case maintenance guide](https://github.com/vllm-project/vllm-ascend/blob/18a1bd414b6609281c950206afa99aa8ac38227a/tests/e2e/cases/E2E_TEST_CASE_GUIDE.md): separate cases, shared execution and CI selection; YAML-driven and custom pytest cases; new handlers or lifecycle changes are framework extensions.
- [Single-node configuration](https://github.com/vllm-project/vllm-ascend/blob/18a1bd414b6609281c950206afa99aa8ac38227a/tests/e2e/common/single_node/single_node_config.py) and [shared entry](https://github.com/vllm-project/vllm-ascend/blob/18a1bd414b6609281c950206afa99aa8ac38227a/tests/e2e/common/single_node/test_single_node.py): parametrized configuration objects, handlers for requests/assertions and AISBench for configured evaluation.
- [Service management](https://github.com/vllm-project/vllm-ascend/blob/18a1bd414b6609281c950206afa99aa8ac38227a/tests/e2e/conftest.py): context-managed lifecycle and PD/EPD multi-service support.

AFD adopts these responsibility boundaries with Python declarations rather than YAML. The concrete Case, deployment module and execution-path checks are AFD proposals, not upstream components with identical architecture. Upstream-specific branches, warning-only handling of unknown handlers and cleanup policies are not automatically AFD gate semantics.

Supplementary reference: vLLM `db914cad5bac261dbcd7cf05ded573c1d704f650`, [RemoteVLLMServer / RemoteOpenAIServer](https://github.com/vllm-project/vllm/blob/db914cad5bac261dbcd7cf05ded573c1d704f650/tests/utils.py). Reuse service-management boundaries while meeting AFD multi-role and existing cleanup protocols.

These fixed revisions describe research snapshots, not AFD's currently pinned dependency interfaces.

## 12. Details to settle in the pilot

The direction is Python declarations, shared service management, phase-specific acceptance and preservation of effective coverage. Settle these interfaces and batches without changing those goals:

- Concrete types for Cases, phase results and checks, and cross-window evidence passing.
- Migration batches for old CLI and model entries, including dedicated-runner and multi-pod callers.
- Available device validation per batch and conditions for retiring old implementations.
- Resources/frequency for new five-GPU, eight-NPU and other tasks, plus actual D01 allocation and implementation records; PR authors own execution as specified in the appendix.

These details determine implementation and batching. They do not justify an early global registry, generic plugin system or scheduling layer.
