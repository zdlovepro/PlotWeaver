import re
import unicodedata


def clean_text(text: str) -> str:
    """Clean raw novel text: normalize encoding, remove noise characters."""
    # Normalize unicode
    text = unicodedata.normalize("NFC", text)

    # Remove BOM
    text = text.lstrip("\ufeff")

    # Replace Windows line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null bytes and other control characters (keep \n and \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Remove excessive whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]

    # Collapse more than 3 consecutive blank lines to 2
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def read_novel_file(file_path: str) -> str:
    """Read a novel file, trying UTF-8 first, then GBK fallback."""
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f"Cannot decode file: {file_path}")
