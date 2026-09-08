from __future__ import annotations

from ...contracts import ChapterDocument, ChapterSynopsis, LocalSynopsis, SynopsisIssue, SynopsisQuality


def assess_synopsis_quality(document: ChapterDocument, local_synopses: tuple[LocalSynopsis, ...],
                            chapter: ChapterSynopsis, *,
                            semantic_issues: tuple[SynopsisIssue, ...] = ()) -> SynopsisQuality:
    issues: list[SynopsisIssue] = list(semantic_issues)
    source_ids = tuple(unit.unit_id for unit in document.annotation_units)
    reviewed_ids = tuple(source_id for local in local_synopses for source_id in local.reviewed_source_unit_ids)
    coverage = len(set(reviewed_ids) & set(source_ids)) / max(1, len(source_ids))
    if reviewed_ids != source_ids:
        issues.append(SynopsisIssue("error", "SOURCE_WINDOW_COVERAGE", "局部窗口未按顺序恰好覆盖全部章节原文段落"))
    segment_count = sum(len(local.segments) for local in local_synopses)
    if segment_count == 0:
        issues.append(SynopsisIssue("error", "NO_LOCAL_PLOT", "整章没有可供章级压缩的局部剧情梗概"))
    try:
        for local in local_synopses: local.validate()
        chapter.validate(local_synopses)
    except ValueError as exc:
        issues.append(SynopsisIssue("error", "STRUCTURE_INVALID", str(exc)))
    passed = not any(issue.severity == "error" for issue in issues)
    return SynopsisQuality(passed, "verified" if passed else "failed", round(coverage, 6),
                           len(local_synopses), segment_count, tuple(issues))
