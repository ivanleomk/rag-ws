from lib.db import (
    get_table,
    insert_data_into_async_table,
    get_table_async,
)
from lib.data import (
    generate_category_test_labels,
    get_dataset,
    generate_data_and_labels,
    generate_test_labels,
    save_labels,
    download_arxiv_dataset,
    format_arxiv_dataset,
)
from asyncio import run
from lib.synthethic import generate_category_questions
from lib.models import EmbeddedPassage, ArxivPaper
import lancedb
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
LANCE_DIR_PATH = os.path.join(BASE_PATH, "../lance")
DATA_DIR = os.path.join(BASE_PATH, "../data")


async def setup_table():
    async_db = await lancedb.connect_async(LANCE_DIR_PATH)
    db = lancedb.connect(LANCE_DIR_PATH)

    table = get_table(db, "ms_marco", EmbeddedPassage)
    async_table = await get_table_async(async_db, "ms_marco")

    dataset = get_dataset(1000)
    data, labels = generate_data_and_labels(dataset)

    await insert_data_into_async_table(async_table, data, batch_size=500)

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    single_label_path = os.path.join(DATA_DIR, "queries_single_label.jsonl")
    if not os.path.exists(single_label_path):
        single_label = generate_test_labels(labels, True)
        save_labels(single_label, single_label_path)

    multi_label_path = os.path.join(DATA_DIR, "queries_multi_label.jsonl")
    if not os.path.exists(multi_label_path):
        multi_label = generate_test_labels(labels, False)
        save_labels(multi_label, multi_label_path)

    table.create_fts_index("text", replace=True)


async def setup_metadata_example():
    db = lancedb.connect(LANCE_DIR_PATH)
    async_db = await lancedb.connect_async(LANCE_DIR_PATH)

    table = get_table(db, "arxiv_papers", ArxivPaper)
    async_table = await get_table_async(async_db, "arxiv_papers")

    dataset = download_arxiv_dataset(100)
    data = format_arxiv_dataset(dataset)

    metadata_path = os.path.join(DATA_DIR, "arxiv_metadata.jsonl")
    if not os.path.exists(metadata_path):
        save_labels(data, metadata_path)

    await insert_data_into_async_table(async_table, data, batch_size=50)

    table.create_fts_index("text", replace=True)

    category_questions_path = os.path.join(DATA_DIR, "category_questions.jsonl")
    if not os.path.exists(category_questions_path):
        category_labels = await generate_category_questions(data, 10)
        category_labels = generate_category_test_labels(category_labels)
        save_labels(category_labels, category_questions_path)


if __name__ == "__main__":
    run(setup_table())
    run(setup_metadata_example())
