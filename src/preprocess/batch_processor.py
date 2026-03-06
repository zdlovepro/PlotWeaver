import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from ..models.novel import Novel
from .text_cleaner import clean_text, read_novel_file
from .chapter_splitter import split_chapters


def load_novel(file_path: str, novel_id: Optional[str] = None) -> Novel:
    """Load a novel from a .txt file, clean and split into chapters."""
    if novel_id is None:
        novel_id = os.path.splitext(os.path.basename(file_path))[0]
    raw_text = read_novel_file(file_path)
    cleaned = clean_text(raw_text)
    novel = Novel(
        id=novel_id,
        title=novel_id,
        file_path=file_path,
        raw_text=cleaned,
    )
    novel.chapters = split_chapters(novel)
    return novel


def process_novels_parallel(
    file_paths: list[str],
    max_workers: int = 4,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> list[Novel]:
    """Load and preprocess multiple novel files in parallel."""
    novels: list[Novel] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(load_novel, path): path for path in file_paths}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                novel = future.result()
                novels.append(novel)
                if progress_callback:
                    progress_callback(f"Loaded: {path}")
            except Exception as exc:
                if progress_callback:
                    progress_callback(f"Failed {path}: {exc}")
                raise
    return novels
