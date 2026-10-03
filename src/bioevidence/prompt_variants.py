"""Arm C: the A configuration with one prompt-level clarification.

C exists to test H3 — whether telling the model *when* ``mixed`` applies shifts
its verdict distribution on condition-dependent cases.  The intervention is
deliberately narrow and purely descriptive:

* only the system prompt and the ``finish`` tool description change;
* ``insufficient`` keeps its meaning (no relevant evidence found) and its
  contract coupling (``abstained=True``, citation optional);
* no verdict is added, nothing structural changes, and every non-prompt setting
  is shared with A.

C's prompt is *derived* from A's at call time rather than copied, so the shared
portion cannot drift and the recorded diff is exactly the inserted text.
"""

from __future__ import annotations

import copy
from typing import Any

from .llm_agent import TOOL_SCHEMAS, _system_prompt

#: Inserted into the system prompt's Rules section, immediately before the
#: insufficient rule so the intervention is a clean insertion.
ARM_C_MIXED_RULES = (
    "- Use verdict=mixed when the answer to the original question changes "
    "direction across conditions, subpopulations, or sub-questions in the "
    "evidence: the evidence supports the question under some conditions and "
    "contradicts it under others.\n"
    "- Do NOT use verdict=mixed merely because values differ in magnitude, "
    "because the evidence has limitations or uncertainty, or because several "
    "distinct findings are present. Those are not changes in the answer's "
    "direction.\n"
)

#: Appended to the ``finish`` tool description, carrying the same clarification.
ARM_C_FINISH_SUFFIX = (
    " Use verdict=mixed only when the answer to the original question changes "
    "direction across conditions, subpopulations, or sub-questions; a difference "
    "in magnitude, a limitation, or an uncertainty is not by itself a change in "
    "direction."
)

_ANCHOR = "- Use verdict=insufficient when there is genuinely no relevant evidence."


def arm_c_system_prompt() -> str:
    """A's system prompt with the mixed clarification inserted verbatim."""
    base = _system_prompt()
    if _ANCHOR not in base:
        raise RuntimeError(
            "anchor for the Arm C insertion is missing from the system prompt; "
            "refusing to guess where the clarification belongs"
        )
    return base.replace(_ANCHOR, ARM_C_MIXED_RULES + _ANCHOR, 1)


def arm_c_tool_schemas() -> list[dict[str, Any]]:
    """A deep copy of the tool schemas with only the finish description changed.

    Deep-copied so that handing C its own schemas cannot mutate the shared
    ``TOOL_SCHEMAS`` object that A reads.
    """
    schemas = copy.deepcopy(TOOL_SCHEMAS)
    for schema in schemas:
        if schema["function"]["name"] == "finish":
            schema["function"]["description"] += ARM_C_FINISH_SUFFIX
            return schemas
    raise RuntimeError("finish tool not found in TOOL_SCHEMAS")


def arm_c_intervention_diff() -> dict[str, Any]:
    """The exact, auditable difference between A and C — nothing else differs."""
    base = _system_prompt()
    variant = arm_c_system_prompt()
    return {
        "intervention": "mixed-usage clarification only",
        "changed": ["system_prompt", "finish_tool_description"],
        "unchanged": [
            "insufficient meaning and abstained coupling",
            "product contract and citation requirements",
            "verdict vocabulary",
            "workflow instructions and tool set",
            "all non-prompt run settings",
        ],
        "system_prompt_insertion": ARM_C_MIXED_RULES,
        "finish_description_appended": ARM_C_FINISH_SUFFIX,
        "system_prompt_chars": {"a": len(base), "c": len(variant),
                                "delta": len(variant) - len(base)},
        "system_prompt_delta_is_insertion_only": (
            variant.replace(ARM_C_MIXED_RULES, "", 1) == base
        ),
    }
