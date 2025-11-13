import os
import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from utils.logger import get_logger
from cluster_manager import load_cluster_model
from utils.config import config

logger = get_logger("retriever")

DATA_FILE = os.path.join("data", "clustered_incidents.json")
INDEX_FILE = os.path.join("data", "embeddings.faiss")
MODEL_NAME = "all-MiniLM-L6-v2"


class IncidentRetriever:
    def __init__(self):
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
        if not os.path.exists(INDEX_FILE):
            raise FileNotFoundError(f"FAISS index not found: {INDEX_FILE}")

        with open(DATA_FILE, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.df = pd.DataFrame(self.data)

        logger.info("Loaded DataFrame with shape %s and columns: %s", self.df.shape, list(self.df.columns))

        self.model = SentenceTransformer(MODEL_NAME, device="cpu", trust_remote_code=True)
        self.index = faiss.read_index(INDEX_FILE)

        logger.info("Retriever initialized: loaded %d records and FAISS index", len(self.df))

    def search(self, query: str, top_k: int = 5):
        logger.info("Starting search for query: %s", query)

        query_vec = self.model.encode([query], convert_to_numpy=True)

        cluster_model = load_cluster_model()
        cluster_id = int(cluster_model.predict(query_vec)[0]) #predict() method is designed to handle batches of inputs
        logger.info("Predicted cluster ID: %d", cluster_id)

        cluster_index_path = os.path.join(config.CLUSTER_FAISS_DIR, f"cluster_{cluster_id}.faiss")
        logger.info("Expected FAISS index path for this cluster: %s", cluster_index_path)

        if not os.path.exists(cluster_index_path):
            logger.warning("No FAISS index found for cluster %d", cluster_id)
            return pd.DataFrame(), []

        cluster_index = faiss.read_index(cluster_index_path)
        logger.info("Loaded FAISS cluster index successfully (ntotal=%d)", cluster_index.ntotal)

        distances, indices = cluster_index.search(query_vec, top_k)
        logger.info("Search results from cluster FAISS: indices=%s, distances=%s", indices, distances)

        if "cluster_id" not in self.df.columns:
            logger.error(
                "'cluster_id' column missing in DataFrame! Columns present: %s",
                list(self.df.columns)
            )
            raise KeyError("cluster_id")

        # Get records for that cluster
        cluster_df = self.df[self.df["cluster_id"] == cluster_id].reset_index(drop=True)
        logger.info(
            "Filtered DataFrame for cluster %d: %d rows found",
            cluster_id, len(cluster_df)
        )

        if cluster_df.empty:
            logger.warning("No incidents found for cluster %d in data file.", cluster_id)
            return pd.DataFrame(), []

        # Collect valid results
        valid = []
        valid_distances = []
        for idx_pos, idx in enumerate(indices[0]):
            if 0 <= idx < len(cluster_df):
                valid.append(idx)
                valid_distances.append(float(distances[0][idx_pos]))

        # if cluster too small, use global search ??
        if len(valid) < top_k:
            logger.info("Cluster too small; falling back to global search.")
            global_distances, global_indices = self.index.search(query_vec, top_k)
            valid = [int(i) for i in global_indices[0] if 0 <= i < len(self.df)]
            valid_distances = [float(d) for d in global_distances[0]]

            results = self.df.iloc[valid].reset_index(drop=True)
            logger.info("Global search returned %d results", len(results))
            return results, valid_distances

        results = cluster_df.iloc[valid].reset_index(drop=True)
        logger.info("Final results retrieved: %d records", len(results))
        return results, valid_distances
