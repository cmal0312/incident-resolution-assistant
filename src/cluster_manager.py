import os
import json
import pickle
import numpy as np
import pandas as pd
import faiss
from sklearn.cluster import MiniBatchKMeans
from logger import get_logger
from utils.config import config

logger = get_logger("cluster_manager")

def _determine_num_clusters(n_samples: int) -> int:
    if n_samples < 1000:
        return 10
    elif n_samples < 5000:
        return 25
    elif n_samples < 20000:
        return 50
    else:
        return min(config.MAX_CLUSTERS, max(config.MIN_CLUSTERS, n_samples // 500))


def recluster_and_update_indices(embeddings: np.ndarray, combined_df: pd.DataFrame):
    n_samples = len(embeddings)
    if n_samples == 0:
        logger.warning("No embeddings available for clustering. Skipping.")
        return

    num_clusters = min(_determine_num_clusters(n_samples), n_samples)
    logger.info("Starting reclustering on %d embeddings using %d clusters...", n_samples, num_clusters)

    os.makedirs(config.CLUSTER_FAISS_DIR, exist_ok=True)

    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=config.BATCH_SIZE,
        random_state=42,
        verbose=0
    )
    cluster_ids = kmeans.fit_predict(embeddings)
    logger.info("Clustering completed: %d clusters formed.", num_clusters)

    with open(config.CLUSTER_MODEL_FILE, "wb") as f:
        pickle.dump(kmeans, f)
    logger.info("Saved MiniBatchKMeans model to %s", config.CLUSTER_MODEL_FILE)

    combined_df = combined_df.copy()
    combined_df["cluster_id"] = cluster_ids
    combined_df.to_json(config.CLUSTER_ASSIGNMENTS_FILE, orient="records", indent=2, force_ascii=False)
    logger.info("Saved cluster assignments to %s", config.CLUSTER_ASSIGNMENTS_FILE)

    dim = embeddings.shape[1]
    total_saved = 0
    for cluster in range(num_clusters):
        cluster_indices = np.where(cluster_ids == cluster)[0]
        if len(cluster_indices) == 0:
            continue

        cluster_embeddings = embeddings[cluster_indices].astype(np.float32)
        cluster_index = faiss.IndexFlatL2(dim)
        cluster_index.add(cluster_embeddings)

        cluster_path = os.path.join(config.CLUSTER_FAISS_DIR, f"cluster_{cluster}.faiss")
        faiss.write_index(cluster_index, cluster_path)
        total_saved += len(cluster_indices)
        logger.info("Cluster %d: saved %d vectors to %s", cluster, len(cluster_indices), cluster_path)

    logger.info("Reclustering completed — %d clusters updated, %d total vectors processed.",
                num_clusters, total_saved)


def load_cluster_model():
    if not os.path.exists(config.CLUSTER_MODEL_FILE):
        raise FileNotFoundError("Cluster model not found. Run reclustering first.")
    with open(config.CLUSTER_MODEL_FILE, "rb") as f:
        return pickle.load(f)
