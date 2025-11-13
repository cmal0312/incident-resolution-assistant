import os

class config:
    # === file paths ===
    DATA_DIR = "data"
    DATA_FILE = os.path.join("data", "cleaned_incidents.json")
    CLUSTER_MODEL_FILE = os.path.join("data", "cluster_model.pkl")
    CLUSTER_ASSIGNMENTS_FILE = os.path.join("data", "clustered_incidents.json")
    CLUSTER_FAISS_DIR = os.path.join("data", "clusters")
    INDEX_FILE = os.path.join("data", "embeddings.faiss")
    EMBEDDINGS_FILE = os.path.join("data", "embeddings.npy")
    TEMP_JSON = os.path.join("data", "temp_incidents.json")
    LOG_FILE = os.path.join("data", "process.log")

    # === clustering parameters ===
    BATCH_SIZE = 1000
    MAX_CLUSTERS = 200
    MIN_CLUSTERS = 10

    # === other parameters ===
    MODEL_NAME = "all-MiniLM-L6-v2"


