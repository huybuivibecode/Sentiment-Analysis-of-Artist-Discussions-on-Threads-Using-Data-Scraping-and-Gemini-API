"""
Phan tich cam xuc bang Gemini.

Luu y:
- File nay doc comment tu CSV, gui theo batch cho Gemini, va luu ket qua ra CSV.
- Neu muon hardcode API key, dien vao GEMINI_API_KEY ben duoi.
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
from google import genai


DEFAULT_MODEL = "gemini-2.5-flash"
FALLBACK_MODELS = ["gemini-2.5-flash-lite", "gemini-2.0-flash"]
GEMINI_API_KEYS = [""] #them apikey vao day có thể thêm nhiều 
current_key_index = 0

def build_client():
    global current_key_index
    if not GEMINI_API_KEYS:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Chua co GEMINI_API_KEYS trong gemini_sentiment.py hoac bien moi truong.")
    else:
        api_key = GEMINI_API_KEYS[current_key_index].strip()
        
    return genai.Client(api_key=api_key)


def load_comments(csv_path, text_column, encoding="utf-8-sig"):
    df = pd.read_csv(csv_path, encoding=encoding)
    if text_column not in df.columns:
        raise ValueError(f"Khong tim thay cot text: {text_column}")
    return df


def build_records(df, text_column, id_column=None, limit=None):
    working_df = df.copy()
    working_df[text_column] = working_df[text_column].fillna("").astype(str).str.strip()
    working_df = working_df[working_df[text_column] != ""].reset_index(drop=True)

    if limit is not None:
        working_df = working_df.head(limit).copy()

    records = []
    for idx, row in working_df.iterrows():
        record_id = str(row[id_column]) if id_column and id_column in working_df.columns else str(idx)
        records.append(
            {
                "id": record_id,
                "text": row[text_column],
            }
        )

    return working_df, records


def chunk_records(records, batch_size):
    for index in range(0, len(records), batch_size):
        yield records[index:index + batch_size]


def build_prompt(batch_records, celebrity_name):
    lines = [
        f"Ban la he thong phan tich du luan mang xa hoi ve nguoi noi tieng: {celebrity_name}.",
        "",
        "NHIEM VU:",
        "1. Xac dinh sentiment tong quat cua comment.",
        "2. Xac dinh emotion cu the cua nguoi viet.",
        "3. Xac dinh topic chinh duoc nhac toi.",
        "4. Xac dinh comment co tinh tranh cai/drama hay khong.",
        "",
        "I. SENTIMENT LABELS",
        "",
        "- positive:",
        "  cam xuc tich cuc, ung ho, khen ngoi",
        "",
        "- neutral:",
        "  khong co cam xuc ro rang, thong tin, hoi dap",
        "",
        "- negative:",
        "  che bai, cong kich, phan doi, chi trich",
        "",
        "- mixed:",
        "  vua khen vua che, cam xuc trai chieu",
        "",
        "II. EMOTION LABELS",
        "",
        "- nguong_mo:",
        "  than tuong, yeu thich, tu hao, ung ho",
        "",
        "- that_vong:",
        "  hut hang, buon, mat niem tin",
        "",
        "- tuc_gian:",
        "  phan no, toxic, cong kich, chi trich manh",
        "",
        "- hai_huoc:",
        "  meme, troll, ca khia, pha tro, gay cuoi",
        "",
        "- dong_cam:",
        "  benh vuc, thuong cam, bao ve, chia se",
        "",
        "- trung_lap:",
        "  khong co emotion noi bat",
        "",
        "III. TOPIC LABELS",
        "",
        "- music:",
        "  bai hat, MV, album, am nhac, giong hat",
        "",
        "- drama:",
        "  scandal, phot, tranh cai cong dong",
        "",
        "- dating:",
        "  hen ho, tinh cam, couple",
        "",
        "- breakup:",
        "  chia tay, ket thuc moi quan he",
        "",
        "- appearance:",
        "  visual, ngoai hinh, body, guong mat",
        "",
        "- fashion:",
        "  outfit, trang phuc, phong cach thoi trang",
        "",
        "- personality:",
        "  tinh cach, thai do, cach ung xu",
        "",
        "- statement:",
        "  phat ngon, quan diem, tra loi phong van",
        "",
        "- livestream:",
        "  stream, livestream, noi dung truc tiep",
        "",
        "- social_post:",
        "  bai dang mang xa hoi, caption, story",
        "",
        "- fan_war:",
        "  tranh cai fandom, combat fan",
        "",
        "- comeback:",
        "  tro lai hoat dong, tai xuat",
        "",
        "- movie:",
        "  phim anh, vai dien, dien xuat",
        "",
        "- performance:",
        "  san khau, live show, vocal live, dancing",
        "",
        "- advertisement:",
        "  quang cao, dai dien thuong hieu, booking nhan hang",
        "",
        "- rumor:",
        "  tin don, nghi van, thong tin chua xac thuc",
        "",
        "- achievement:",
        "  giai thuong, thanh tich, ky luc",
        "",
        "- daily_life:",
        "  cuoc song hang ngay, hobby, sinh hoat",
        "",
        "- other:",
        "  noi dung khong thuoc cac nhom tren",
        "",
        "QUY TAC PHAN TICH:",
        "",
        "1. Phan tich theo ngu canh thuc te cua mang xa hoi Viet Nam.",
        "2. Hieu slang, viet tat, typo, teen code, emoji, meme.",
        "3. Hieu sarcasm, mia mai, ca khia.",
        "4. Danh gia cam xuc huong truc tiep toi nguoi duoc nhac den.",
        "5. Neu vua khen vua che -> sentiment=mixed.",
        "6. Neu comment mang tinh meme/troll -> uu tien emotion=hai_huoc.",
        "7. Neu comment benh vuc hoac thuong cam -> emotion=dong_cam.",
        "8. Neu chi thong tin, hoi dap, spam, tag ten -> emotion=trung_lap.",
        "9. Neu lien quan scandal, fanwar, tranh cai, cong kich -> controversy=true.",
        "10. Chi chon 1 topic chinh nhat.",
        "11. Neu co nhieu topic, chon topic chiem trong tam cua comment.",
        "12. Khong suy dien ngoai noi dung comment.",
        "13. Neu comment qua ngan va khong du context -> uu tien neutral + trung_lap.",
        "14. confidence phai nam trong khoang tu 0 den 1.",
        "15. rationale toi da 15 tu, khong xuong dong.",
        "",
        "OUTPUT FORMAT:",
        "",
        "Tra ve JSON object hop le theo schema duoc cung cap.",
        "Khong tra ve markdown.",
        "Khong giai thich them.",
        "",
        "DANH SACH COMMENT:",
    ]

    for record in batch_records:
        lines.append(f'ID: {record["id"]} | COMMENT: {record["text"]}')

    return "\n".join(lines)


def get_response_schema():
    return {
        "type": "object",
        "properties": {
            "analyses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "sentiment": {
                            "type": "string",
                            "enum": ["positive", "neutral", "negative", "mixed"],
                        },
                        "emotion": {
                            "type": "string",
                            "enum": [
                                "nguong_mo",
                                "that_vong",
                                "tuc_gian",
                                "hai_huoc",
                                "dong_cam",
                                "trung_lap",
                            ],
                        },
                        "topic": {
                            "type": "string",
                            "enum": [
                                "music",
                                "drama",
                                "dating",
                                "breakup",
                                "appearance",
                                "fashion",
                                "personality",
                                "statement",
                                "livestream",
                                "social_post",
                                "fan_war",
                                "comeback",
                                "movie",
                                "performance",
                                "advertisement",
                                "rumor",
                                "achievement",
                                "daily_life",
                                "other",
                            ],
                        },
                        "controversy": {"type": "boolean"},
                        "confidence": {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "id",
                        "sentiment",
                        "emotion",
                        "topic",
                        "controversy",
                        "confidence",
                        "rationale",
                    ],
                },
            }
        },
        "required": ["analyses"],
    }


def is_retryable_gemini_error(exc: Exception) -> bool:
    error_text = str(exc).upper()
    return any(token in error_text for token in ["503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "429"])


def analyze_batch(client, batch_records, celebrity_name, model=DEFAULT_MODEL):
    prompt = build_prompt(batch_records, celebrity_name=celebrity_name)
    models_to_try = [model] + [candidate for candidate in FALLBACK_MODELS if candidate != model]
    last_error = None

    for model_name in models_to_try:
        for retry_index in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": get_response_schema(),
                        "temperature": 0.2,
                    },
                )
                return json.loads(response.text)["analyses"]
            except Exception as exc:
                last_error = exc
                if not is_retryable_gemini_error(exc):
                    raise

                if retry_index < 2:
                    sleep_seconds = 2 * (retry_index + 1)
                    time.sleep(sleep_seconds)
                    continue

    raise RuntimeError(f"Gemini tam thoi qua tai hoac khong san sang: {last_error}")


def analyze_comments(
    df,
    text_column,
    celebrity_name,
    id_column=None,
    model=DEFAULT_MODEL,
    batch_size=10,
    limit=None,
    checkpoint_file="sentiment_checkpoint.csv"
):
    working_df, records = build_records(df, text_column=text_column, id_column=id_column, limit=limit)
    if not records:
        return working_df.assign(
            sentiment="",
            emotion="",
            topic="",
            controversy=False,
            confidence=None,
            rationale="",
        )

    # 1. Đọc checkpoint nếu có
    analysis_rows = []
    processed_ids = set()
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            chk_df = pd.read_csv(checkpoint_file, encoding='utf-8-sig')
            # Đảm bảo id là chuỗi để so sánh chính xác
            chk_df['id'] = chk_df['id'].astype(str)
            analysis_rows = chk_df.to_dict('records')
            processed_ids = set(chk_df['id'].tolist())
            print(f"[*] Đã tải {len(processed_ids)} kết quả từ file checkpoint.")
        except Exception as e:
            print(f"[!] Lỗi đọc checkpoint: {e}")

    # 2. Lọc ra các record chưa xử lý
    records_to_process = [r for r in records if r["id"] not in processed_ids]
    print(f"[*] Cần xử lý tiếp: {len(records_to_process)} records.")

    if records_to_process:
        client = build_client()
        batch_iterator = list(chunk_records(records_to_process, batch_size=batch_size))
        i = 0
        while i < len(batch_iterator):
            batch_records = batch_iterator[i]
            try:
                batch_result = analyze_batch(
                    client,
                    batch_records,
                    celebrity_name=celebrity_name,
                    model=model,
                )
                analysis_rows.extend(batch_result)
                
                # 3. Lưu xuống checkpoint ngay sau mỗi batch
                if checkpoint_file:
                    chk_df_temp = pd.DataFrame(analysis_rows)
                    chk_df_temp.to_csv(checkpoint_file, index=False, encoding='utf-8-sig')
                    print(f"  -> Đã lưu tạm {len(analysis_rows)} kết quả...")

                # Nghỉ 2 giây giữa các batch để tránh bị Rate Limit
                time.sleep(2)
                i += 1  # Chỉ tăng khi batch chạy thành công
            except Exception as e:
                global current_key_index
                if current_key_index < len(GEMINI_API_KEYS) - 1:
                    current_key_index += 1
                    print(f"\n[!] Lỗi: {e}")
                    print(f"[*] Đang tự động chuyển sang API Key thứ {current_key_index + 1}/{len(GEMINI_API_KEYS)}...")
                    client = build_client()
                    time.sleep(2)
                    continue  # Quay lại vòng lặp while, giữ nguyên i để thử lại batch này
                else:
                    print(f"\n[!] Dừng giữa chừng do lỗi: {e}")
                    print(f"[*] Đã dùng hết {len(GEMINI_API_KEYS)} API Keys. Đã lưu an toàn {len(analysis_rows)} kết quả vào {checkpoint_file}.")
                    print("[*] Hãy cập nhật API keys mới và chạy lại.")
                    break

    # 4. Gộp kết quả
    analysis_df = pd.DataFrame(
        analysis_rows,
        columns=["id", "sentiment", "emotion", "topic", "controversy", "confidence", "rationale"],
    )

    if id_column and id_column in working_df.columns:
        working_df["_record_id"] = working_df[id_column].astype(str)
    else:
        working_df["_record_id"] = working_df.index.astype(str)

    merged_df = working_df.merge(
        analysis_df,
        left_on="_record_id",
        right_on="id",
        how="left",
    ).drop(columns=["_record_id", "id"])

    return merged_df


def save_analysis_csv(df, output_path, encoding="utf-8-sig"):
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False, encoding=encoding)
    return target_path
