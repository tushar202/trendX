from trendx.agents.critic import run as critic_run
from trendx.agents.evidence_anchor import run as evidence_anchor_run
from trendx.agents.formatter import run as formatter_run
from trendx.agents.research import run as research_run
from trendx.agents.synthesis import run as synthesis_run
from trendx.agents.trend_detector import run as trend_detector_run

__all__ = [
    "critic_run",
    "evidence_anchor_run",
    "formatter_run",
    "research_run",
    "synthesis_run",
    "trend_detector_run",
]
