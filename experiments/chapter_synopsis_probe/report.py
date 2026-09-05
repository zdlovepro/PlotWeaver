from __future__ import annotations

from collections import Counter
from typing import Any


def build_summary(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(item.get("status", "unknown")) for item in decisions)
    return {
        "chapter_count": len(decisions),
        "status_counts": dict(sorted(status_counts.items())),
        "model_call_count": sum(int(item.get("model_call_count", 0)) for item in decisions),
        "chapters": [{
            "chapter_id": item.get("chapter_id", ""),
            "title": item.get("title", ""),
            "source_chars": item.get("source_chars", 0),
            "synopsis_chars": len(str(item.get("synopsis", ""))),
            "status": item.get("status", "unknown"),
            "model_call_count": item.get("model_call_count", 0),
        } for item in decisions],
    }


def _cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def build_markdown(manifest: dict[str, Any], decisions: list[dict[str, Any]]) -> str:
    summary = build_summary(decisions)
    rows = [
        "# 章节梗概提取能力实验报告",
        "",
        f"- 运行ID：`{manifest.get('run_id', '')}`",
        f"- 提取模型：`{manifest.get('extraction_model', '')}`",
        f"- 审校模型：`{manifest.get('review_model', '')}`",
        f"- 审校模式：`{manifest.get('audit_mode', '')}`",
        f"- 模型调用总数：`{summary['model_call_count']}`",
        "- 运行策略：每个请求最多调用一次，不重试、不回退、不修订。",
        "",
        "| 章节 | 标题 | 原文字符 | 梗概字符 | 调用数 | 状态 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in summary["chapters"]:
        rows.append(
            f"| {_cell(item['chapter_id'])} | {_cell(item['title'])} | "
            f"{item['source_chars']} | {item['synopsis_chars']} | "
            f"{item['model_call_count']} | `{item['status']}` |"
        )
    for decision in decisions:
        rows.extend(["", f"## {_cell(decision.get('chapter_id'))} {_cell(decision.get('title'))}", ""])
        if decision.get("synopsis"):
            rows.extend(["### 模型生成的章节梗概", "", str(decision["synopsis"]), ""])
        if decision.get("error"):
            rows.extend(["### 运行错误", "", str(decision["error"]), ""])
        reviews = decision.get("reviews", {})
        if isinstance(reviews, dict) and reviews:
            rows.extend(["### 独立审校", ""])
            for name in ("fidelity", "coverage", "granularity"):
                review = reviews.get(name)
                if not isinstance(review, dict):
                    continue
                rows.append(f"- `{name}`：`{review.get('status', 'unknown')}`")
                for issue in review.get("issues", []):
                    if isinstance(issue, dict):
                        category = issue.get("维度", issue.get("类别", ""))
                        source_ids = ", ".join(issue.get("相关原文段落ID", []))
                        rows.append(
                            f"  - {_cell(issue.get('问题ID'))} / {_cell(category)}："
                            f"{_cell(issue.get('说明'))}（{_cell(source_ids)}）"
                        )
                for error in review.get("protocol_errors", []):
                    rows.append(f"  - 协议错误：{_cell(error)}")
    rows.extend(["", "## 状态统计", ""])
    for status, count in summary["status_counts"].items():
        rows.append(f"- `{status}`：{count}")
    rows.append("")
    return "\n".join(rows)
