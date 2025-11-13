import os
import pandas as pd
import json
from utils.logger import get_logger
from utils.config import config

logger = get_logger("json_creator")

REQUIRED_COLUMNS = [
    "Number",
    "Short description",
    "Description",
    "Comments and Work notes",
    "Resolution notes",
]


def create_json_from_file(upload_path: str) -> str:

    os.makedirs(config.DATA_DIR, exist_ok=True)
    logger.info("Starting JSON creation from file: %s", upload_path)

    ext = upload_path.split(".")[-1].lower()

    if ext in ("xlsx", "xls"):
        df = pd.read_excel(upload_path, dtype=str)
        logger.info("Read Excel file, rows: %d", len(df))
    elif ext == "csv":
        try:
            df = pd.read_csv(upload_path, dtype=str, encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("UTF-8 decoding failed. Retrying with latin1 encoding...")
            df = pd.read_csv(upload_path, dtype=str, encoding="latin1")
        logger.info("Read CSV file, rows: %d", len(df))
    else:
        logger.error("Unsupported file type: %s", ext)
        raise ValueError("Unsupported file type. Please upload CSV or Excel.")

    df.columns = df.columns.str.strip()

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.warning("Missing columns in upload: %s", missing_cols)
    available_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]

    df = df[available_cols].copy()

    before = len(df)
    df = df.dropna(subset=available_cols)
    after = len(df)
    logger.info("Dropped %d rows with missing required fields. Remaining: %d", before - after, after)

    records = df.to_dict(orient="records") # json compatible form
    tmp_path = config.TEMP_JSON + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, config.TEMP_JSON)

    logger.info("Temporary JSON file created at: %s", config.TEMP_JSON)
    return config.TEMP_JSON
