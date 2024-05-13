from loguru import logger
from datasets import load_dataset, Dataset
import lancedb
from env import settings
import os
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from tqdm import tqdm
from itertools import batched
import hashlib
import json


DATASET = "ms_marco"
SPLIT = "train"
CONFIG = "v1.1"
SLICE = 40
LANCEDB_PATH = "./lance"
LANCEDB_TABLE = "ms_marco"

QUERY_FILE = "queries.jsonl"

# Set the Hugging Face token from settings into the environment
os.environ["HF_TOKEN"] = settings.HF_TOKEN
func = get_registry().get("openai").create(name="text-embedding-3-small")


class EmbeddedPassage(LanceModel):
    vector: Vector(func.ndims()) = func.VectorField()
    chunk_id: str
    text: str = func.SourceField()


def configure_lancedb():
    db = lancedb.connect(LANCEDB_PATH)

    try:
        db.drop_table(LANCEDB_TABLE)
        logger.info("Existing 'marco-ms' table dropped.")
    except Exception as e:
        logger.error(f"Failed to drop 'marco-ms' table: {e}")

    db.create_table(LANCEDB_TABLE, schema=EmbeddedPassage)


def generate_data_and_labels(dataset: Dataset):
    passages = set()
    data = []
    labels = []
    for row in tqdm(dataset):
        selected_passages = []
        for idx, passage in enumerate(row["passages"]["passage_text"]):
            if passage not in passages:
                chunk_id = hashlib.md5(passage.encode()).hexdigest()
                passage_data_obj = {"text": passage, "chunk_id": chunk_id}
                data.append(passage_data_obj)
                passages.add(passage)

                if row["passages"]["is_selected"][idx]:
                    selected_passages.append(passage_data_obj)

        if selected_passages:
            labels.append(
                {
                    "query": row["query"],
                    "selected_passages": selected_passages,
                    "answer": row["answers"],
                    "query_id": row["query_id"],
                    "query_type": row["query_type"],
                }
            )

    logger.info(
        f"Extracted {len(passages)} unique passages and {len(labels)} test queries"
    )
    return data, labels


def insert_into_lancedb(data):
    db = lancedb.connect(LANCEDB_PATH)
    table = db.open_table(LANCEDB_TABLE)
    batches = batched(data, 20)
    total_inserted = 0
    for batch in tqdm(batches):
        table.add(list(batch))
        total_inserted += len(batch)
    logger.info(f"Total of {total_inserted} records inserted into the database.")


def save_labels(labels):
    with open(QUERY_FILE, "w") as file:
        for label in labels:
            file.write(json.dumps(label) + "\n")
    logger.info(f"Labels saved to {QUERY_FILE}")


def setup():
    dataset_object = load_dataset(DATASET, CONFIG, split=SPLIT, streaming=True).take(
        SLICE
    )

    logger.info("Starting to download database")
    configure_lancedb()
    data, labels = generate_data_and_labels(dataset_object)
    insert_into_lancedb(data=data)
    save_labels(labels)


if __name__ == "__main__":
    setup()
