import os
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from utils.logger import get_logger
from cluster_manager import recluster_and_update_indices
from utils.config import config

logger = get_logger("faiss_updater")

def _get_text_for_embedding(record: dict) -> str:
    if record.get("Incident description"):
        return str(record.get("Incident description") or "")
    sd = record.get("Short description") or ""
    desc = record.get("Description") or ""
    return f"{sd} {desc}".strip()


def update_faiss_with_new_data(new_json_path: str):
    
    model = SentenceTransformer(config.MODEL_NAME, device="cpu", trust_remote_code=True)

    # Load existing data
    if os.path.exists(config.DATA_FILE):
        with open(config.DATA_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
        logger.info("Loaded existing data: %d records", len(existing))
    else:
        existing = []
        logger.info("No existing data file found; starting fresh.")

    # Load new data
    with open(new_json_path, "r", encoding="utf-8") as f:
        new_data = json.load(f)
    logger.info("Loaded new data: %d records", len(new_data))

    combined_df = pd.DataFrame(existing + new_data)
    before = len(combined_df)
    if "Number" in combined_df.columns:
        combined_df.drop_duplicates(subset=["Number"], keep="last", inplace=True)
    after = len(combined_df)
    logger.info("Deduplicated dataset: before=%d after=%d removed=%d", before, after, before - after)

    os.makedirs(os.path.dirname(config.DATA_FILE), exist_ok=True)
    combined_df.to_json(config.DATA_FILE, orient="records", indent=2, force_ascii=False)
    logger.info("Saved combined incidents to %s", config.DATA_FILE)

    existing_numbers = {rec.get("Number") for rec in existing}
    new_records = [rec for rec in new_data if rec.get("Number") not in existing_numbers]
    logger.info("Found %d brand new incidents to embed", len(new_records))

    if os.path.exists(config.EMBEDDINGS_FILE):
        existing_embeddings = np.load(config.EMBEDDINGS_FILE)
        logger.info("Loaded existing embeddings: shape=%s", existing_embeddings.shape)
    else:
        existing_embeddings = np.zeros((0, 384), dtype=np.float32)  # 384 is MiniLM-L6-v2 output dim
        logger.info("No existing embeddings file found; starting with empty array.")

    if new_records:
        texts = [_get_text_for_embedding(rec) for rec in new_records]
        logger.info("Encoding %d new records...", len(texts))
        new_embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        dim = new_embeddings.shape[1]

        all_embeddings = np.vstack([existing_embeddings, new_embeddings]) #stack vertically
        np.save(config.EMBEDDINGS_FILE, all_embeddings)
        logger.info("Saved updated embeddings: total=%d vectors", len(all_embeddings))

        if os.path.exists(config.INDEX_FILE):
            index = faiss.read_index(config.INDEX_FILE)
            logger.info("Loaded existing FAISS index with %d vectors", index.ntotal)
        else:
            index = faiss.IndexFlatL2(dim)
            logger.info("Created new FAISS index.")

        index.add(np.array(new_embeddings, dtype=np.float32))
        faiss.write_index(index, config.INDEX_FILE)
        logger.info(
            "Appended %d new vectors to FAISS index (total now: %d, dim: %d)",
            len(new_embeddings),
            index.ntotal,
            dim,
        )
    else:
        logger.info("No new records found. Skipping embedding and FAISS update.")
        all_embeddings = existing_embeddings

    logger.info("Starting reclustering process with %d total embeddings...", len(all_embeddings))
    recluster_and_update_indices(all_embeddings, combined_df)
    logger.info("Reclustering completed successfully.")

    return len(new_records), len(combined_df)
