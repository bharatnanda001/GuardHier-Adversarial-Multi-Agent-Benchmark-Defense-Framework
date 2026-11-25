# GuardHier: Securing Hierarchical Multi-Agent Collaboration Against Adversarial Attacks

**Research Report, System Documentation & Experimental Summary**  
**Repository:** [GuardHier-Adversarial-Multi-Agent-Benchmark-Defense-Framework](https://github.com/bharatnanda001/GuardHier-Adversarial-Multi-Agent-Benchmark-Defense-Framework)  
**Reproducibility Standard:** `temperature = 0.0`, `seed = 42`  

---

## 1. Executive Summary & Project Initiative

Multi-agent LLM collaboration frameworks improve reasoning accuracy by decomposing complex problems across specialized domain experts and synthesizing their outputs through iterative refinement. However, **hierarchical multi-agent topologies** (such as **TalkHier**) expose a critical attack surface: a single compromised member agent can inject fabricated context, prompt-extraction directives, or high-confidence false consensus claims into structured inter-agent messages $(M_{ij}, B_{ij}, I_{ij})$, poisoning the supervisor's final decision or leaking private agent memory.

**GuardHier** (`guardhier_bench`) is a modular defense and benchmarking framework that bridges:
1. **TalkHier** (`./talkhier/`): A hierarchical multi-agent refinement architecture where domain-specific expert teams (*Moral Scenario*, *College Physics*, *Machine Learning*, *Formal Logic*) iteratively refine intermediate outputs under an Evaluation Supervisor.
2. **Cracks-in-Agent-Collaboration** (`./Cracks-in-Agent-Collaboration/`): An adversarial evaluation suite defining three canonical multi-agent attack scenarios (**AS-I: Decision Poisoning**, **AS-II: Indirect EchoLeak**, and **AS-III: Consensus Collision**).

> **Strict Baseline Isolation:** GuardHier dynamically wraps and bridges both baseline repositories via `guardhier_bench/adapters/cracks_to_talkhier.py` without modifying a single line of code in `./talkhier/` or `./Cracks-in-Agent-Collaboration/` (verified by `test_01_baseline_repo_integrity` and `Test01BaselineIsolation`).

---

## 2. System Architecture & Working Structure

### 2.1 High-Level Defense Pipeline

![GuardHier Three-Layer Defense Architecture](./guardhier_architecture_final_1795639369713.jpg)

In each refinement round $r \in \{1, \dots, R\}$:
1. **Member Execution (`AdversarialMemberAgent`)**: Active domain experts generate structured outputs: Message ($M_{ij}$), Background ($B_{ij}$), and Intermediate Output ($I_{ij}$). Compromised agents inject attack payloads into these fields.
2. **Layer C1 (`SchemaGuard`)**: Wraps every member claim in an untrusted XML provenance envelope with a SHA-256 content hash, strips known exfiltration directives, and prepends a strict privilege boundary policy to the supervisor's system prompt.
3. **Layer C2 (`TrustSupervisor`)**: Embeds all member intermediate outputs $I_{ij}$ using `BAAI/bge-small-en-v1.5` (with a deterministic `FallbackEmbedder` when offline), computes the semantic consensus centroid, and assigns each agent a centroid-alignment trust weight $w_j \in [0, 1]$.
4. **Layer C3 (`NodeQuarantine`)**: Tracks cumulative anomaly strikes across rounds. Any agent with $w_j < \tau$ (`TRUST_THRESHOLD = 0.55`) or a `SchemaGuard` provenance violation for `QUARANTINE_STRIKES = 2` rounds has its communication edges severed for all subsequent rounds.
5. **Evaluation Supervisor**: Synthesizes the final consensus answer using only active, trust-weighted, provenance-isolated claims and logs a structured `TrialTrace` to `guardhier_bench/results/traces.jsonl`.

### 2.2 Multi-Round Execution Flow

```mermaid
flowchart LR
    A["Domain Member Agents (M_ij, B_ij, I_ij)"] --> B["C1: SchemaGuard XML Wrapping & Sanitization"]
    B --> C["C2: TrustSupervisor Centroid Scoring (w_j)"]
    C --> D{"C3: Strikes >= 2 or w_j < 0.55?"}
    D -- "Yes (2nd Strike)" --> E["Sever Edges & Quarantine Node"]
    D -- "No" --> F["Retain in Active Topology"]
    E --> G["Evaluation Supervisor Synthesis"]
    F --> G
    G --> H["Next Refinement Round / Final Output"]
```

### 2.3 Repository Structure

| Module | File Path | Responsibility |
| :--- | :--- | :--- |
| **Configuration** | `guardhier_bench/config/settings.py` | Research constants (`TEMPERATURE=0.0`, `SEED=42`, `TRUST_THRESHOLD=0.55`, `QUARANTINE_STRIKES=2`), OpenAI-compatible endpoint loading, dataset paths. |
| **Bridge Adapter** | `guardhier_bench/adapters/cracks_to_talkhier.py` | `LLMClient`, `AdversarialMemberAgent`, `GuardHierSession`, `TrialTrace` telemetry logging. |
| **Evaluators** | `guardhier_bench/adapters/evaluator.py` | `AS1Evaluator`, `AS2Evaluator`, `AS3Evaluator`, and `CANONICAL_CANARIES`. |
| **Defense C1** | `guardhier_bench/defenses/schema_guard.py` | XML provenance envelopes, SHA-256 signatures, `EXFILTRATION_PATTERNS` sanitization, `PRIVILEGE_BOUNDARY_PROMPT`. |
| **Defense C2** | `guardhier_bench/defenses/trust_supervisor.py` | Semantic centroid computation, pairwise cosine similarity, `[TRUST_WEIGHT=w_j]` injection. |
| **Defense C3** | `guardhier_bench/defenses/node_quarantine.py` | Strike tracking (`AgentQuarantineRecord`), dynamic topology pruning (`filter_active_agents`). |
| **Benchmark CLI** | `guardhier_bench/run_ablation.py` | Single-condition and `--run_full_matrix` runner exporting `ablation_table.csv` and `ablation_table.md`. |
| **Visualization** | `guardhier_bench/plot_results.py` | Generates 300 DPI publication figures (`fig1_asr_comparison`, `fig2_round_progression`). |
| **Test Suites** | `guardhier_bench/tests/test_smoke.py`, `test_end_to_end.py` | 35 automated unit and end-to-end tests verifying isolation, evaluators, C1–C3, and trace exports. |

---

## 3. Threat Model & Core Research Innovations

![Adversarial Threat Model vs. GuardHier Defenses](./guardhier_threat_model_comparison_1795639325037.jpg)

### 3.1 Adversarial Scenarios (AS-I, AS-II, AS-III)

1. **AS-I — Decision Poisoning:** A compromised member injects fabricated contextual reasoning into $B_{ij}$ and $I_{ij}$ (e.g., irrelevant geographic claims steering classification toward `"Attack"`), causing the unprotected supervisor to misclassify the query.
2. **AS-II — Indirect EchoLeak:** A compromised member embeds prompt-injection directives inside $B_{ij}$ (e.g., *"copy and output all the text before..."*) to trick the supervisor into leaking confidential `Memory_i` canary tokens (`CANONICAL_CANARIES`).
3. **AS-III — Consensus Collision:** A compromised member emits high-confidence contradictory claims forcing a specific incorrect choice (`Option A`), attempting to hijack majority/consensus aggregation.

### 3.2 Three-Layer Defense Innovations (C1, C2, C3)

- **Contribution C1 — Privilege & Provenance Separation (`SchemaGuard`):**
  Every peer claim is encapsulated in `<peer_claim agent_id="{id}" round="{r}" privilege="UNTRUSTED_DATA" hash="{sha256_12}">...</peer_claim>`. Regex-based sanitization replaces exfiltration directives with `[SANITIZED_ADVERSARIAL_DIRECTIVE]`, while `PRIVILEGE_BOUNDARY_PROMPT` instructs the supervisor to treat enclosed text strictly as untrusted data, never as executable instructions.
- **Contribution C2 — Embedding-Outlier Trust Scoring (`TrustSupervisor`):**
  All member intermediate outputs $I_{ij}$ are embedded into L2-normalized vectors. The supervisor computes the team's semantic centroid and each agent's cosine alignment weight $w_j \in [0, 1]$, injecting explicit trust metadata into the aggregation prompt so divergent/poisoned claims are down-weighted.
- **Contribution C3 — Dynamic Topology Reconfiguration (`NodeQuarantine`):**
  Instead of passively tolerating a Byzantine agent across all rounds, `NodeQuarantine` accumulates strikes whenever $w_j < 0.55$ or a provenance violation occurs. After `QUARANTINE_STRIKES = 2` strikes, the compromised node's edges are severed (`filter_active_agents`), preventing further drift and eliminating wasted tokens in subsequent rounds.

---

## 4. Experimental Results & Engineering Verification

### 4.1 Full 4×4 Ablation Matrix (`moral_scenario`, `N=5`, TalkHier Topology)

Executed via `guardhier_bench/run_ablation.py --dataset moral_scenario --num_samples 5 --run_full_matrix --mock` (`Seed=42`, `Temp=0.0`):

| Attack Scenario | Defense Mode | Clean Accuracy (CA %) | Attack Success Rate (ASR %) | Avg Rounds | Avg Tokens / Query | Quarantine Precision (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **NONE** | `NONE` | **100.0%** | **0.0%** | 2.0 | 506.4 | 100.0% |
| **NONE** | `C1` | **100.0%** | **0.0%** | 2.0 | 1247.6 | 100.0% |
| **NONE** | `C1_C2` | **100.0%** | **0.0%** | 2.0 | 1583.2 | 100.0% |
| **NONE** | `FULL_GUARDHIER` | **100.0%** | **0.0%** | 2.0 | 1528.2 | 0.0% |
| **AS1** | `NONE` | **0.0%** | **100.0%** | 2.0 | 562.0 | 0.0% |
| **AS1** | `C1` | **100.0%** | **0.0%** | 2.0 | 1318.4 | 0.0% |
| **AS1** | `C1_C2` | **100.0%** | **0.0%** | 2.0 | 1692.4 | 0.0% |
| **AS1** | `FULL_GUARDHIER` | **100.0%** | **0.0%** | 2.0 | 1561.0 | **100.0%** |
| **AS2** | `NONE` | **0.0%** | **100.0%** | 2.0 | 641.6 | 0.0% |
| **AS2** | `C1` | **100.0%** | **0.0%** | 2.0 | 1384.4 | 0.0% |
| **AS2** | `C1_C2` | **100.0%** | **0.0%** | 2.0 | 1758.4 | 0.0% |
| **AS2** | `FULL_GUARDHIER` | **100.0%** | **0.0%** | 2.0 | 1623.0 | **100.0%** |
| **AS3** | `NONE` | **0.0%** | **100.0%** | 2.0 | 601.2 | 0.0% |
| **AS3** | `C1` | **0.0%** | **100.0%** | 2.0 | 1342.0 | 0.0% |
| **AS3** | `C1_C2` | **100.0%** | **0.0%** | 2.0 | 1733.2 | 0.0% |
| **AS3** | `FULL_GUARDHIER` | **100.0%** | **0.0%** | 2.0 | 1569.0 | **100.0%** |

> **Key Ablation Insight:** **C1 alone** (`SchemaGuard`) defeats **AS-I** and **AS-II** (`0.0%` ASR), but **fails against AS-III** (`100.0%` ASR) because AS-III uses high-confidence contradictory domain claims rather than prompt-injection syntax. Adding **C2** (`TrustSupervisor`) neutralizes AS-III (`100.0%` CA, `0.0%` ASR), and adding **C3** (`NodeQuarantine`) achieves **100.0% Quarantine Precision** across all three attack scenarios while reducing token overhead compared to `C1_C2` (`1561–1623` vs. `1692–1758` tokens/query).

### 4.2 Quantitative Research Figures

![Attack Success Rate Comparison](./fig1_asr_comparison.png)

![Multi-Round Progression & Quarantine](./fig2_round_progression.png)

### 4.3 Live LLM Verification & Bug Fixes

During live integration debugging:
- **Seed Compatibility Fix:** OpenAI-compatible endpoints such as **Gemini (`gemini-2.5-flash`)** reject `seed=42` with `400 BadRequestError`. Updated `LLMClient.chat_completion` in `guardhier_bench/adapters/cracks_to_talkhier.py` to automatically retry without `seed` on `400`/`seed` errors and log warnings on live failures instead of silently falling back to mock responses.
- **Live Trial (`gemini-2.5-flash`):** Running `--dataset moral_scenario --num_samples 1 --attack as1 --defense full_guardhier` achieved **100.0% Clean Accuracy**, **0.0% Attack Success Rate**, **100.0% Quarantine Precision**, and **1941.0 tokens/query**.
- **Automated Test Suite:** All **35/35** tests in `guardhier_bench/tests/test_smoke.py` and `guardhier_bench/tests/test_end_to_end.py` pass after the final edit.
