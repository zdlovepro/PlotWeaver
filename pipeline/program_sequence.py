"""Sequential orchestration for checked chapter-program generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .jsonio import read_json, write_json
from .paths import runs_dir, validate_author_id
from .scene_generation import DEFAULT_SCENE_MAX_REPAIRS, generate_from_program


def _index_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("generation_mode") != "controlled_expansion":
        raise ValueError("顺序生成只接受受控扩写章节规划索引")
    rows = payload.get("chapters")
    if not payload.get("passed") or not isinstance(rows, list) or not rows:
        raise ValueError("章节规划索引未通过质量门槛，不能开始生成")
    result = [item for item in rows if isinstance(item, dict)]
    if len(result) != len(rows):
        raise ValueError("章节规划索引包含无效记录")
    return result


def generate_program_sequence(
    author_id: str,
    index_path: Path,
    *,
    run_id: str = "program-sequence",
    max_repairs: int = DEFAULT_SCENE_MAX_REPAIRS,
) -> Path:
    """Generate planned chapters in order without passing a rejected state on.

    The program index contains no source prose.  The prose model receives its
    narrative facts solely through the paragraph subgraph; the prior chapter's
    verified ``state`` is carried only as orchestration output while committed
    graph patches provide the next request's factual state.
    """

    author_id = validate_author_id(author_id)
    if max_repairs < 0:
        raise ValueError("max_repairs must be non-negative")
    payload = read_json(index_path)
    if not isinstance(payload, dict):
        raise ValueError("章节规划索引必须是 JSON object")
    if str(payload.get("author_id", "")) != author_id:
        raise ValueError("章节规划索引的 author_id 与当前参数不一致")
    rows = _index_rows(payload)
    destination = runs_dir(author_id, run_id) / "program_sequence"
    destination.mkdir(parents=True, exist_ok=True)
    graph_path_text = str(payload.get("narrative_graph", "")).strip()
    graph_path = Path(graph_path_text) if graph_path_text else None
    if graph_path is None:
        raise ValueError("受控扩写顺序生成必须提供 narrative_graph.json 作为唯一事实底座")
    if not graph_path.is_file():
        raise FileNotFoundError(f"narrative graph does not exist: {graph_path}")
    runtime_graph_path = destination / "runtime_graph_state.json"

    results: list[dict[str, Any]] = []
    prior_state: dict[str, Any] | None = None
    blocked = False
    for ordinal, row in enumerate(rows, start=1):
        chapter_no = int(row.get("chapter_no", ordinal))
        program_path = Path(str(row.get("program", "")))
        if blocked:
            result = {
                "chapter_no": chapter_no,
                "chapter_id": str(row.get("chapter_id", "")),
                "status": "not_started",
                "reason": "前序章节未通过，禁止把未验证状态传入后续章节",
            }
            results.append(result)
            continue
        if not program_path.is_file():
            result = {
                "chapter_no": chapter_no,
                "chapter_id": str(row.get("chapter_id", "")),
                "status": "failed",
                "reason": f"章节程序不存在：{program_path}",
            }
            results.append(result)
            blocked = True
            continue

        input_state_path: Path | None = None
        if prior_state is not None:
            input_state_path = destination / f"chapter-{chapter_no:04d}.input_state.json"
            write_json(input_state_path, prior_state)
        scoped_run_id = f"{run_id}-chapter-{chapter_no:04d}"
        try:
            outcome = generate_from_program(
                author_id,
                program_path,
                story_state_path=input_state_path,
                graph_path=graph_path,
                runtime_graph_path=runtime_graph_path,
                run_id=scoped_run_id,
                max_repairs=max_repairs,
            )
            output_dir = Path(str(outcome.get("output_dir", "")))
            exit_path = Path(str(outcome.get("chapter_exit_state", output_dir / "chapter_exit_state.json")))
            exit_payload = read_json(exit_path)
            if not bool(outcome.get("overall_passed")) or not isinstance(exit_payload, dict) or not bool(exit_payload.get("overall_passed")):
                result = {
                    "chapter_no": chapter_no,
                    "chapter_id": str(row.get("chapter_id", "")),
                    "status": "rejected",
                    "output_dir": str(output_dir),
                    "reason": "章节候选稿未通过全部硬门槛，出口状态不可用于下一章",
                }
                blocked = True
            elif not bool(exit_payload.get("graph_patch_committed", False)):
                result = {
                    "chapter_no": chapter_no,
                    "chapter_id": str(row.get("chapter_id", "")),
                    "status": "rejected",
                    "output_dir": str(output_dir),
                    "reason": "章节未提交通过校验的图谱补丁，禁止把候选状态传入下一章",
                }
                blocked = True
            elif not isinstance(exit_payload.get("state"), dict):
                result = {
                    "chapter_no": chapter_no,
                    "chapter_id": str(row.get("chapter_id", "")),
                    "status": "failed",
                    "output_dir": str(output_dir),
                    "reason": "通过的章节缺少可传递的出口状态",
                }
                blocked = True
            else:
                prior_state = dict(exit_payload["state"])
                result = {
                    "chapter_no": chapter_no,
                    "chapter_id": str(row.get("chapter_id", "")),
                    "status": "passed",
                    "output_dir": str(output_dir),
                    "exit_state": str(exit_path),
                    "graph_patch": str(exit_payload.get("graph_patch", "")),
                    "input_state": str(input_state_path) if input_state_path else "",
                }
        except Exception as exc:
            result = {
                "chapter_no": chapter_no,
                "chapter_id": str(row.get("chapter_id", "")),
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            blocked = True
        results.append(result)
        write_json(destination / "sequence.partial.json", {
            "schema_version": "1.0",
            "author_id": author_id,
            "index_path": str(index_path),
            "completed": len(results),
            "blocked": blocked,
            "narrative_graph": str(graph_path) if graph_path else "",
            "runtime_graph_state": str(runtime_graph_path),
            "chapters": results,
        })

    report = destination / "sequence_summary.json"
    write_json(report, {
        "schema_version": "1.0",
        "author_id": author_id,
        "index_path": str(index_path),
        "chapter_count": len(rows),
        "passed_chapters": sum(item.get("status") == "passed" for item in results),
        "blocked": blocked,
        "narrative_graph": str(graph_path) if graph_path else "",
        "runtime_graph_state": str(runtime_graph_path),
        "chapters": results,
    })
    return report
