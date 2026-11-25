# GuardHier: Project Overview, Problem Statement, Vision, Outputs & Future Scope

**Bridging Two Baseline Repositories:**
1. **TalkHier** (`./talkhier/`): Hierarchical multi-agent refinement framework where domain-specific expert teams iteratively refine intermediate outputs under an Evaluation Supervisor.
2. **Cracks-in-Agent-Collaboration** (`./Cracks-in-Agent-Collaboration/`): Adversarial multi-agent benchmark defining attack scenarios **AS-I (Decision Poisoning)**, **AS-II (Indirect EchoLeak)**, and **AS-III (Consensus Collision)**.

By combining **TalkHier**'s hierarchical collaboration loop with **Cracks-in-Agent-Collaboration**'s threat models inside `guardhier_bench/` (without modifying either baseline repository), **GuardHier** provides both a unified adversarial benchmark and a three-layer defense architecture (**C1: SchemaGuard**, **C2: TrustSupervisor**, **C3: NodeQuarantine**).

---

## 1. Problem Statement

Modern LLM-based multi-agent systems improve reasoning on complex tasks (medical QA, moral reasoning, physics, formal logic, and machine learning) by organizing specialized agents into hierarchical teams that exchange structured messages $(M_{ij}, B_{ij}, I_{ij})$ across multiple refinement rounds.

However, **inter-agent communication is implicitly trusted**:
1. **Lack of Privilege Separation (Vulnerable to AS-II Indirect EchoLeak):** Supervisors and peers parse incoming agent text as trusted context, allowing a compromised member to embed prompt-injection directives (e.g., *"copy and output all the text before..."*) that exfiltrate private agent memory or system canary tokens.
2. **Vulnerability to Semantic Poisoning (Vulnerable to AS-I Decision Poisoning):** A compromised agent can inject fabricated, off-topic, or misleading reasoning into its background $B_{ij}$ and intermediate output $I_{ij}$, steering the supervisor toward an incorrect decision.
3. **Susceptibility to Consensus Hijacking (Vulnerable to AS-III Consensus Collision):** Even when syntax-level prompt-injection filters are present, an adversary can emit high-confidence contradictory claims (e.g., forcing `Option A`) that distort majority/consensus aggregation.
4. **Static Communication Topologies (Token Waste & Persistent Drift):** Without dynamic node isolation, a compromised agent remains connected in every refinement round, repeatedly polluting consensus and wasting inference tokens.

**Core Research Question:**  
*How can we secure hierarchical multi-agent LLM collaboration against decision poisoning, memory exfiltration, and consensus collision—without degrading clean task accuracy or modifying the underlying collaboration framework?*

---

## 2. Vision & Research Initiative

Our vision is to establish **self-healing, trust-aware hierarchical multi-agent systems** where:
- **Untrusted Peer Data is Strictly Isolated from Instructions:** Every inter-agent message is encapsulated in provenance-tagged XML envelopes (`<peer_claim ... privilege="UNTRUSTED_DATA">`) with cryptographic hashes and privilege-boundary enforcement (**C1: SchemaGuard**).
- **Consensus is Semantically Weighted, Not Blindly Averaged:** The supervisor evaluates each member's semantic alignment against the team's embedding centroid (`BAAI/bge-small-en-v1.5`) and down-weights divergent outliers (**C2: TrustSupervisor**).
- **Compromised Nodes are Dynamically Quarantined:** Persistent anomalous or policy-violating agents are automatically severed from the communication topology after repeated strikes (**C3: NodeQuarantine**), preserving clean accuracy while reducing token overhead.
- **Reproducible, Non-Invasive Benchmarking:** Both baseline repositories remain 100% unmodified (`test_01_baseline_repo_integrity`), enabling fair apples-to-apples ablation studies (`NONE` → `C1` → `C1_C2` → `FULL_GUARDHIER`) under deterministic settings (`temperature=0.0`, `seed=42`).

---

## 3. System Outputs & Verified Deliverables

### 3.1 Generated Artifacts & Logs
| Output Artifact | Location | Description |
| :--- | :--- | :--- |
| **Per-Trial Telemetry** | `guardhier_bench/results/traces.jsonl` | Complete JSONL trace per query (clean accuracy, ASR, rounds, token counts, latency, trust weights, quarantined agents). |
| **Ablation Table (CSV & MD)** | `report/ablation_table.csv`, `report/ablation_table.md` | 4×4 matrix comparing `NONE`, `AS1`, `AS2`, `AS3` × `NONE`, `C1`, `C1_C2`, `FULL_GUARDHIER`. |
| **Publication Figures (300 DPI)** | `report/fig1_asr_comparison.png`, `report/fig2_round_progression.png` | Quantitative ASR comparison and multi-round quarantine progression plots. |
| **Architecture & Threat Diagrams** | `report/guardhier_architecture_final_1795639369713.jpg`, `report/guardhier_threat_model_comparison_1795639325037.jpg` | Visual architecture and threat-to-defense mapping diagrams. |
| **Full Research Report** | `report/GuardHier_Research_Report.md` | Complete technical documentation, formulas, ablation results, and verification summary. |

### 3.2 Key Observed Outputs
- **Automated Verification:** **35/35** unit and end-to-end tests passing (`guardhier_bench/tests/test_smoke.py` and `test_end_to_end.py`).
- **4×4 Ablation Matrix (`moral_scenario`, `N=5`, `--mock`):**
  - Unprotected baseline (`NONE`): **0.0% Clean Accuracy / 100.0% ASR** under AS-I, AS-II, and AS-III.
  - `C1` alone: Defeats AS-I and AS-II (**0.0% ASR**), but fails on AS-III (**100.0% ASR**).
  - `FULL_GUARDHIER` (`C1 + C2 + C3`): Achieves **100.0% Clean Accuracy**, **0.0% ASR**, and **100.0% Quarantine Precision** across all three attack scenarios while lowering token usage vs. `C1_C2` (`1561–1623` vs. `1692–1758` tokens/query).
- **Live LLM Verification (`gemini-2.5-flash`):** Executed live trial on `moral_scenario` (`AS1` + `FULL_GUARDHIER`) achieving **100.0% Clean Accuracy**, **0.0% ASR**, **100.0% Quarantine Precision**, and **1941.0 tokens/query**.

---

## 4. Future Scope

1. **Large-Scale Multi-Model Live Evaluation:** Run the full ablation matrix across `N=50–200` samples on `medqa`, `qa_data`, and all MMLU domains (`moral_scenario`, `college_physics`, `machine_learning`, `formal_logic`) using multiple live LLM backends (Gemini, Groq, OpenAI, local vLLM).
2. **Multi-Agent Collusion & Adaptive Adversaries:** Extend `COMPROMISED_COUNT` from `1` to `2+` colluding agents to evaluate Byzantine resilience when attackers attempt to shift the C2 semantic centroid or craft adaptive stealthy payloads.
3. **Cross-Topology Benchmarking:** Conduct full comparative evaluations across all four supported topologies (`centralized`, `horizontal`, `joint`, `talkhier`) to quantify structural resilience vs. communication cost.
4. **False-Positive & Minority-Opinion Preservation:** Add adaptive thresholding and benign-disagreement benchmarks to ensure honest minority experts are not falsely quarantined during legitimate scientific/medical debate.
5. **Cryptographic Agent Identity & Replay Protection:** Upgrade `SchemaGuard` content hashes to asymmetric digital signatures per agent to prevent identity spoofing in distributed deployments.
