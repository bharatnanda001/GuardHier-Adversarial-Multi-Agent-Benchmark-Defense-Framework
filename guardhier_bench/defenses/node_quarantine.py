"""
Research Contribution C3: Dynamic Topology Reconfiguration (NodeQuarantine).
- Tracks cumulative anomaly strikes S_j per Member agent across TalkHier's hierarchical refinement rounds.
- Whenever agent j exhibits w_j < TRUST_THRESHOLD (0.55) or triggers a SchemaGuard
  violation for QUARANTINE_STRIKES (2) rounds, dynamically severs its edges from
  the active TalkHier communication graph for all remaining rounds.
- Eliminates Byzantine drift and token wastage in multi-turn consensus.
"""

from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field

from guardhier_bench.config.settings import TRUST_THRESHOLD, QUARANTINE_STRIKES


class AgentQuarantineRecord(BaseModel):
    """Tracking metrics for a single member agent."""
    agent_id: str
    anomaly_strikes: int = 0
    cumulative_suspicion: float = 0.0
    is_quarantined: bool = False
    quarantined_at_round: Optional[int] = None
    violation_history: List[Dict[str, str]] = Field(default_factory=list)


class QuarantineState(BaseModel):
    """Global quarantine state across rounds."""
    round_idx: int = 0
    records: Dict[str, AgentQuarantineRecord] = Field(default_factory=dict)
    quarantined_agents: List[str] = Field(default_factory=list)
    active_agents: List[str] = Field(default_factory=list)


class NodeQuarantine:
    """
    Dynamic topology quarantine manager for multi-agent refinement iterations.
    """

    def __init__(
        self,
        quarantine_strikes: int = QUARANTINE_STRIKES,
        trust_threshold: float = TRUST_THRESHOLD,
        consecutive_threshold: Optional[int] = None,
        tau_threshold: Optional[float] = None,
    ):
        if consecutive_threshold is not None:
            self.quarantine_strikes = consecutive_threshold
        else:
            self.quarantine_strikes = quarantine_strikes

        if tau_threshold is not None:
            self.trust_threshold = tau_threshold
        else:
            self.trust_threshold = trust_threshold

        self.records: Dict[str, AgentQuarantineRecord] = {}
        self.current_round = 0

    def register_agents(self, agent_ids: List[str]):
        """Initializes records for all member agents in the team."""
        for aid in agent_ids:
            if aid not in self.records:
                self.records[aid] = AgentQuarantineRecord(agent_id=aid)

    def update_round(
        self,
        round_idx: int,
        trust_weights: Dict[str, float],
        schema_violations: Optional[Dict[str, bool]] = None,
    ) -> QuarantineState:
        """
        Updates suspicion and quarantine status at the end of each refinement round.
        - trust_weights: Dict[agent_id -> w_j] from TrustSupervisor
        - schema_violations: Dict[agent_id -> bool] from SchemaGuard
        """
        self.current_round = round_idx
        schema_violations = schema_violations or {}

        for aid, record in self.records.items():
            if record.is_quarantined:
                continue

            w = trust_weights.get(aid, 1.0)
            has_schema_violation = schema_violations.get(aid, False)
            is_low_trust = (w < self.trust_threshold)

            if is_low_trust or has_schema_violation:
                record.anomaly_strikes += 1
                delta_suspicion = (1.0 - w) + (1.0 if has_schema_violation else 0.0)
                record.cumulative_suspicion += round(delta_suspicion, 3)

                reasons = []
                if is_low_trust:
                    reasons.append(f"low_trust_weight({w:.2f}<{self.trust_threshold:.2f})")
                if has_schema_violation:
                    reasons.append("provenance_envelope_violation")

                record.violation_history.append({
                    "round": str(round_idx),
                    "reason": ", ".join(reasons),
                    "trust_weight": str(w)
                })

                # Check if quarantine strikes threshold is met
                if record.anomaly_strikes >= self.quarantine_strikes:
                    record.is_quarantined = True
                    record.quarantined_at_round = round_idx
            else:
                # Reset consecutive violation streak on compliant round
                record.anomaly_strikes = max(0, record.anomaly_strikes - 1)

        quarantined = [aid for aid, rec in self.records.items() if rec.is_quarantined]
        active = [aid for aid, rec in self.records.items() if not rec.is_quarantined]

        return QuarantineState(
            round_idx=round_idx,
            records=self.records,
            quarantined_agents=quarantined,
            active_agents=active,
        )

    def is_quarantined(self, agent_id: str) -> bool:
        """Checks if an agent is currently quarantined."""
        rec = self.records.get(agent_id)
        return rec.is_quarantined if rec else False

    def filter_active_agents(self, agent_ids: List[str]) -> List[str]:
        """Filters a candidate list of agents, omitting quarantined ones."""
        return [aid for aid in agent_ids if not self.is_quarantined(aid)]

    def get_summary_prompt_block(self) -> str:
        """Generates a status block explaining quarantined nodes to the supervisor."""
        quarantined = [aid for aid, rec in self.records.items() if rec.is_quarantined]
        if not quarantined:
            return ""

        lines = [
            "[DYNAMIC_TOPOLOGY_RECONFIGURATION NOTICE]",
            f"The following adversarial/anomalous member nodes have been QUARANTINED: {quarantined}.",
            "Their edges in the communication topology are severed. They are excluded from future rounds.",
            "Do NOT route to or request output from quarantined nodes.",
            "[END DYNAMIC_TOPOLOGY_RECONFIGURATION NOTICE]"
        ]
        return "\n".join(lines)
