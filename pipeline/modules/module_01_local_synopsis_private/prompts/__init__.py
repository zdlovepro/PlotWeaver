from __future__ import annotations

from .builders import (
    boundary_extract_prompt, boundary_review_prompt,
    chapter_segmentation_prompt,
    chapter_segmentation_recheck_prompt, chapter_segmentation_repair_prompt,
    chapter_segmentation_review_prompt,
    chapter_summary_coverage_review_prompt,
    chapter_summary_prompt,
    chapter_summary_repair_prompt, chapter_summary_review_prompt,
    local_major_adjudication_prompt, local_repair_verification_prompt, local_risk_prompt,
    local_summary_prompt,
    local_summary_repair_prompt,
)
from .registry import PROMPT_REVISION, load_template, template_revision

LOCAL_SYSTEM_PROMPT = load_template("system")
SEGMENTATION_PROMPT_REVISION = template_revision("chapter_segmentation")
SEGMENTATION_REVIEW_PROMPT_REVISION = template_revision("chapter_segmentation_review")
SEGMENTATION_RECHECK_PROMPT_REVISION = template_revision("chapter_segmentation_recheck")
SEGMENTATION_REPAIR_PROMPT_REVISION = template_revision("chapter_segmentation_repair")
LOCAL_SUMMARY_PROMPT_REVISION = template_revision("local_summary")
LOCAL_RISK_PROMPT_REVISION = template_revision("local_risk_review")
LOCAL_MAJOR_ADJUDICATION_PROMPT_REVISION = template_revision("local_major_adjudication")
LOCAL_REPAIR_PROMPT_REVISION = template_revision("local_repair")
LOCAL_REPAIR_VERIFICATION_PROMPT_REVISION = template_revision("local_repair_verification")
BOUNDARY_PROMPT_REVISION = f"{template_revision('boundary_extract')}+{template_revision('boundary_review')}"
CHAPTER_SUMMARY_PROMPT_REVISION = "+".join((template_revision("chapter_summary"),
    template_revision("chapter_summary_review"),
    template_revision("chapter_summary_coverage_review"),
    template_revision("chapter_summary_repair")))

__all__ = [name for name in globals() if not name.startswith("_")]
