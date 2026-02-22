CRITIC_PROMPT = """
You are a harsh technical editor. Review this single trend entry.

INPUT DATA:
Title: {title}
Summary: {summary}
So What: {so_what}
Evidence: {evidence}

Analyze for:
1. Vagueness (e.g., "improves performance" vs "reduces latency by 50ms").
2. Actionability (Does "So What" tell the engineer exactly what to do?).
3. Logic (Does the evidence support the claim?).
4. Tone: REJECT if text uses first-person language (we/our/I/my, e.g., "our analysis", "we propose", "in this paper"). Must be neutral third-person.
5. Specificity: REJECT if claiming improvement (e.g., "significant improvement") without concrete numbers, percentages, or benchmarks.

Return JSON strictly (no markdown, no code fences):
{{
    "score": <int 0-10>,
    "feedback": "<concise actionable fix instructions>",
    "reasoning": "<brief reasoning>"
}}
"""


REFINER_DECISION_PROMPT = """
You are a technical editor. The Critic rejected your draft.
Critic Feedback: "{feedback}"

Decide your next step:
1. SEARCH: If you need missing facts/numbers to satisfy the critic.
2. REWRITE: If you have enough info but need to improve style/clarity.

You must return strictly valid JSON (no markdown).

OPTION 1 - SEARCH:
{{
    "action": "search",
    "query": "<short specific query>"
}}

OPTION 2 - REWRITE:
{{
    "action": "rewrite",
    "rewrite_content": {{
        "title": "{title}",
        "summary": "<improved summary>",
        "so_what": "<improved actionable insight>"
    }}
}}
"""


REFINER_WITH_EVIDENCE_PROMPT = """
You searched for missing evidence.
Found Evidence:
{evidence}

Critic Feedback was: "{feedback}"

Rewrite the trend entry incorporating the evidence.
Return strict JSON only:
{{
    "title": "{title}",
    "summary": "<new summary with evidence>",
    "so_what": "<new actionable insight>"
}}
"""
