from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

IMAGE_OR_MEDIA_RE = re.compile(r"\[(?:图片|视频|文件|语音|表情|动画表情|转发|位置|红包|链接|回复消息)[^\]]*\]")
WHITESPACE_RE = re.compile(r"\s+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
QQ_RE = re.compile(r"(?<!\d)[1-9]\d{5,11}(?!\d)")
URL_RE = re.compile(r"https?://\S+|www\.\S+")


def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc


def get_sender_name(obj: Dict[str, Any]) -> str:
    sender = obj.get("sender") or {}
    return (
        sender.get("remark")
        or sender.get("name")
        or sender.get("nickname")
        or sender.get("uid")
        or "未知"
    )


def clean_text(text: Optional[str], anonymize: bool = True) -> str:
    if not text:
        return ""
    text = str(text).replace("\u200b", " ").replace("\ufeff", " ")
    text = IMAGE_OR_MEDIA_RE.sub(" ", text)
    text = URL_RE.sub("[链接]", text)
    if anonymize:
        text = PHONE_RE.sub("[手机号]", text)
        text = EMAIL_RE.sub("[邮箱]", text)
        # QQ-like numbers are common in exports. Avoid replacing years/timestamps by requiring 6+ digits.
        text = QQ_RE.sub("[数字账号]", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def is_good_text(text: str, min_chars: int = 2) -> bool:
    if len(text) < min_chars:
        return False
    if text in {"/", ".", "。", "？", "?", "！", "!", "…", "...", "哈哈", "hh"}:
        return False
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text):
        return False
    return True


def extract_message_rows(path: str | Path, anonymize: bool = True, min_chars: int = 1) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for obj in iter_jsonl(path):
        if obj.get("recalled"):
            continue
        msg_type = obj.get("type")
        if msg_type not in {"text", "reply"}:
            continue
        content = obj.get("content") or {}
        text = clean_text(content.get("text") or content.get("summary"), anonymize=anonymize)
        if not is_good_text(text, min_chars=min_chars):
            continue
        rows.append(
            {
                "time": obj.get("timestamp") or 0,
                "name": get_sender_name(obj),
                "text": text,
                "type": msg_type,
                "id": obj.get("id"),
            }
        )
    rows.sort(key=lambda x: (x["time"], str(x.get("id") or "")))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
