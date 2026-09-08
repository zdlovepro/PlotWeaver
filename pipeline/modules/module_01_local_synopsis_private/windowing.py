from __future__ import annotations

from dataclasses import dataclass

from ...contracts import ChapterDocument, SourceUnit


# 有限邻接上下文只用于消解指代和跨边界承接，不计入目标覆盖，也不允许引用。
DEFAULT_CONTEXT_UNITS = 1
DEFAULT_BATCH_BLOCKS = 4
# 这是请求装载软目标，不是语义分区上限。单个完整分区可以超过此值并独占批次。
DEFAULT_BATCH_SOURCE_CHARS = 6000
DEFAULT_BOUNDARY_UNITS = 3


@dataclass(frozen=True)
class NavigationBlock:
    """全章导航给出的连续原文所有权范围。

    label 只用于日志和人工检查，不能作为局部梗概的事实输入。
    """

    block_id: str
    start_unit_id: str
    end_unit_id: str
    label: str
    start_index: int
    end_index: int


@dataclass(frozen=True)
class ChapterNavigationPlan:
    """第一模块内部使用的全章语义分块计划。"""

    chapter_id: str
    blocks: tuple[NavigationBlock, ...]


@dataclass(frozen=True)
class NavigationViolation:
    code: str
    block_id: str
    actual: int
    limit: int
    message: str


@dataclass(frozen=True)
class SynopsisWindow:
    chapter_id: str
    index: int
    target_units: tuple[SourceUnit, ...]
    context_before: tuple[SourceUnit, ...] = ()
    context_after: tuple[SourceUnit, ...] = ()
    block_id: str = ""
    navigation_label: str = ""
    # 全章稳定别名，确保导航、局部提取和审校中的 P 编号含义一致。
    unit_aliases: tuple[tuple[str, str], ...] = ()

    @property
    def window_id(self) -> str:
        if self.block_id:
            return f"{self.chapter_id}:block-{self.block_id.lower()}"
        return f"{self.chapter_id}:window-{self.index + 1:03d}"

    @property
    def target_unit_ids(self) -> tuple[str, ...]:
        return tuple(item.unit_id for item in self.target_units)

    @property
    def source_chars(self) -> int:
        return sum(len(item.text) for item in self.target_units)


@dataclass(frozen=True)
class LocalRequestBatch:
    """相邻导航块的传输批次；不改变每个块的内容所有权。"""

    batch_id: str
    windows: tuple[SynopsisWindow, ...]

    @property
    def source_chars(self) -> int:
        return sum(window.source_chars for window in self.windows)


def chapter_paragraph_alias_maps(
    document: ChapterDocument,
) -> tuple[dict[str, str], dict[str, str]]:
    """为整章建立唯一且稳定的模型可读段落编号。"""

    width = max(4, len(str(len(document.annotation_units))))
    stable_to_short = {
        unit.unit_id: f"P{index + 1:0{width}d}"
        for index, unit in enumerate(document.annotation_units)
    }
    return stable_to_short, {short: stable for stable, short in stable_to_short.items()}


def validate_navigation_plan(
    document: ChapterDocument,
    plan: ChapterNavigationPlan,
) -> tuple[NavigationViolation, ...]:
    """只检查范围所有权；语义分区不设字符数、段落数或总区数限制。"""

    errors: list[NavigationViolation] = []

    def add(code: str, message: str, *, block_id: str = "", actual: int = 0, limit: int = 0) -> None:
        errors.append(NavigationViolation(code, block_id, actual, limit, message))
    units = document.annotation_units
    if plan.chapter_id != document.chapter_id:
        add("NAV_CHAPTER_MISMATCH", f"导航章节ID不匹配：{plan.chapter_id}")
    if not units:
        add("NAV_EMPTY", "章节没有可分块的原文段落")
        return tuple(errors)
    if not plan.blocks:
        add("NAV_EMPTY", "章节分块不得为空")
        return tuple(errors)

    known_ids = {item.unit_id for item in units}
    seen_block_ids: set[str] = set()
    expected_start = 0
    for index, block in enumerate(plan.blocks):
        path = f"章节分块[{index}]"
        if block.block_id in seen_block_ids:
            add("NAV_ORDER_ERROR", f"{path}.块ID重复：{block.block_id}", block_id=block.block_id)
        seen_block_ids.add(block.block_id)
        if block.start_unit_id not in known_ids:
            add("NAV_UNKNOWN_START", f"{path}.起始段落ID未知：{block.start_unit_id}", block_id=block.block_id)
        if block.end_unit_id not in known_ids:
            add("NAV_UNKNOWN_END", f"{path}.结束段落ID未知：{block.end_unit_id}", block_id=block.block_id)
        if block.start_index > block.end_index:
            add("NAV_ORDER_ERROR", f"{path}起始段落晚于结束段落", block_id=block.block_id)
        if block.start_index != expected_start:
            if block.start_index < expected_start:
                add("NAV_OVERLAP", f"{path}与前一分块重叠", block_id=block.block_id)
            else:
                add("NAV_GAP", f"{path}与前一分块之间存在未覆盖段落", block_id=block.block_id)
        expected_start = block.end_index + 1

    if plan.blocks[0].start_index != 0:
        add("NAV_GAP", "第一分块必须从本章第一段开始")
    if plan.blocks[-1].end_index != len(units) - 1:
        add("NAV_GAP", "最后分块必须覆盖本章最后一段")
    if expected_start != len(units):
        add("NAV_GAP", "章节分块未能恰好覆盖全部原文段落")
    unique: dict[tuple[str, str, str], NavigationViolation] = {}
    for item in errors:
        unique[(item.code, item.block_id, item.message)] = item
    return tuple(unique.values())


def split_oversized_navigation_blocks(
    document: ChapterDocument,
    plan: ChapterNavigationPlan,
    split_points_by_block: dict[str, tuple[str, ...]],
) -> ChapterNavigationPlan:
    """按终检给出的段落起点拆块；只新增边界，不移动或删除已有边界。"""

    if not split_points_by_block:
        return plan
    blocks_by_id = {block.block_id: block for block in plan.blocks}
    unknown_blocks = set(split_points_by_block) - set(blocks_by_id)
    if unknown_blocks:
        raise ValueError(f"终检引用未知分区：{sorted(unknown_blocks)}")

    _, short_to_stable = chapter_paragraph_alias_maps(document)
    index_by_unit_id = {
        unit.unit_id: index for index, unit in enumerate(document.annotation_units)
    }
    rebuilt_ranges: list[tuple[int, int, str]] = []
    for block in plan.blocks:
        short_ids = split_points_by_block.get(block.block_id, ())
        split_indices: list[int] = []
        for short_id in short_ids:
            stable_id = short_to_stable.get(short_id)
            split_index = index_by_unit_id.get(stable_id, -1)
            if not block.start_index < split_index <= block.end_index:
                raise ValueError(f"{block.block_id}拆分点不在分区内部：{short_id}")
            split_indices.append(split_index)
        if split_indices != sorted(set(split_indices)):
            raise ValueError(f"{block.block_id}拆分点必须唯一并按原文顺序排列")
        starts = [block.start_index, *split_indices]
        ends = [index - 1 for index in split_indices] + [block.end_index]
        rebuilt_ranges.extend(
            (start, end, block.label) for start, end in zip(starts, ends, strict=True)
        )

    units = document.annotation_units
    rebuilt = ChapterNavigationPlan(
        chapter_id=document.chapter_id,
        blocks=tuple(
            NavigationBlock(
                block_id=f"B{index:03d}",
                start_unit_id=units[start].unit_id,
                end_unit_id=units[end].unit_id,
                label=label,
                start_index=start,
                end_index=end,
            )
            for index, (start, end, label) in enumerate(rebuilt_ranges, 1)
        ),
    )
    errors = validate_navigation_plan(document, rebuilt)
    if errors:
        raise ValueError("；".join(item.message for item in errors))
    return rebuilt


def build_synopsis_windows_from_plan(
    document: ChapterDocument,
    plan: ChapterNavigationPlan,
    *,
    context_units: int = DEFAULT_CONTEXT_UNITS,
) -> tuple[SynopsisWindow, ...]:
    """把已通过确定性校验的语义分块计划转换为局部处理窗口。"""

    errors = validate_navigation_plan(document, plan)
    if errors:
        raise ValueError("；".join(item.message for item in errors))
    units = document.annotation_units
    aliases = tuple(chapter_paragraph_alias_maps(document)[0].items())
    windows: list[SynopsisWindow] = []
    for index, block in enumerate(plan.blocks):
        target = units[block.start_index:block.end_index + 1]
        before = units[max(0, block.start_index - context_units):block.start_index]
        after = units[
            block.end_index + 1:min(len(units), block.end_index + 1 + context_units)
        ]
        windows.append(SynopsisWindow(
            chapter_id=document.chapter_id,
            index=index,
            target_units=target,
            context_before=before,
            context_after=after,
            block_id=block.block_id,
            navigation_label=block.label,
            unit_aliases=aliases,
        ))
    return tuple(windows)


def build_boundary_window(
    document: ChapterDocument,
    *,
    kind: str,
    unit_count: int = DEFAULT_BOUNDARY_UNITS,
) -> SynopsisWindow:
    """构造章首或章末的独立边界切片，不改变导航分区。"""

    if kind not in {"opening", "ending"}:
        raise ValueError("boundary kind must be opening or ending")
    if unit_count < 1 or not document.annotation_units:
        raise ValueError("boundary window requires source units")
    units = document.annotation_units
    target = units[:unit_count] if kind == "opening" else units[-unit_count:]
    aliases = tuple(chapter_paragraph_alias_maps(document)[0].items())
    return SynopsisWindow(
        chapter_id=document.chapter_id,
        index=0 if kind == "opening" else len(units) - len(target),
        target_units=tuple(target),
        block_id=f"BOUNDARY-{kind.upper()}",
        unit_aliases=aliases,
    )


def build_local_request_batches(
    windows: tuple[SynopsisWindow, ...],
    *,
    max_blocks: int = DEFAULT_BATCH_BLOCKS,
    max_source_chars: int = DEFAULT_BATCH_SOURCE_CHARS,
) -> tuple[LocalRequestBatch, ...]:
    """按原顺序把完整导航块装入批次，不拆块也不重排。"""

    if max_blocks < 1 or max_source_chars < 1:
        raise ValueError("local request batch limits must be positive")
    batches: list[LocalRequestBatch] = []
    current: list[SynopsisWindow] = []
    chars = 0
    for expected_index, window in enumerate(windows):
        if window.index != expected_index:
            raise ValueError("synopsis windows must be consecutive and source ordered")
        overflow = current and (
            len(current) >= max_blocks or chars + window.source_chars > max_source_chars
        )
        if overflow:
            batches.append(LocalRequestBatch(f"LB{len(batches) + 1:03d}", tuple(current)))
            current = []
            chars = 0
        current.append(window)
        chars += window.source_chars
    if current:
        batches.append(LocalRequestBatch(f"LB{len(batches) + 1:03d}", tuple(current)))
    flattened = tuple(window.window_id for batch in batches for window in batch.windows)
    if flattened != tuple(window.window_id for window in windows):
        raise ValueError("local request batches changed window ownership or order")
    return tuple(batches)
