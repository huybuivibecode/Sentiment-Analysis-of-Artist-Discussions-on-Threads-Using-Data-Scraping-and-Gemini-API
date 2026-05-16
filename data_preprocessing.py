"""
Tien xu ly du lieu cho demo sentiment/social listening.

Nhung gi file nay dang lam:
- Doc/ghi CSV.
- Giu nguyen noi dung tieng Viet co dau.
- Lam sach comment Threads bang cach bo mot so dong nhieu nhu "Translate",
  "Pinned", "Author", moc thoi gian ngan, va cac dong chi chua chi so tuong tac.
- Trim khoang trang o cac cot text.
- Chuyen cot thoi gian sang datetime neu co.
- Xoa ban ghi trung lap theo mot so cot khoa.

Nhung gi file nay co y KHONG lam:
- Khong xoa dau tieng Viet.
- Khong tao text khong dau cho LLM/sentiment.
- Khong stemming/lemmatization.
- Khong gan nhan sentiment.
"""

import re
from pathlib import Path

import pandas as pd


def load_csv(csv_path, encoding="utf-8-sig"):
    return pd.read_csv(csv_path, encoding=encoding)


def save_csv(df, csv_path, encoding="utf-8-sig"):
    target_path = Path(csv_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False, encoding=encoding)
    return target_path


def strip_threads_noise(value):
    if pd.isna(value):
        return ""

    lines = [line.strip() for line in str(value).splitlines()]
    cleaned_lines = []

    for line in lines:
        if not line:
            continue
        if line == "Translate":
            continue
        if line in {"Pinned", "Author", "·"}:
            continue
        if re.fullmatch(r"[\d.,]+[KMB]?", line):
            continue
        if re.fullmatch(r"\d+[hdwmoy]", line.casefold()):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def preprocess_threads_comments(df):
    working_df = df.copy()

    if "comment_text" in working_df.columns:
        working_df["comment_text_raw"] = working_df["comment_text"]
        working_df["comment_text"] = working_df["comment_text"].fillna("").map(strip_threads_noise)

    if "author_handle" in working_df.columns:
        working_df["author_handle"] = working_df["author_handle"].fillna("").astype(str).str.strip()

    if "post_url" in working_df.columns:
        working_df["post_url"] = working_df["post_url"].fillna("").astype(str).str.strip()

    if "create_time" in working_df.columns:
        working_df["create_time"] = pd.to_datetime(working_df["create_time"], errors="coerce")

    subset = [c for c in ["post_url", "author_handle", "comment_text"] if c in working_df.columns]
    if subset:
        working_df = working_df.drop_duplicates(subset=subset).reset_index(drop=True)

    return working_df


def summarize_dataframe(df):
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": df.isna().sum().to_dict(),
    }
