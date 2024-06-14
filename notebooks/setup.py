from lib.db import get_table, insert_data_into_table
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


def setup_table():
    db = lancedb.connect("../lance")
    table = get_table(db, "ms_marco", EmbeddedPassage)
    dataset = get_dataset(1000)
    data, labels = generate_data_and_labels(dataset)
    insert_data_into_table(table, data)

    single_label = generate_test_labels(labels, True)
    multi_label = generate_test_labels(labels, False)

    if not os.path.exists("./data"):
        os.makedirs("./data")
    save_labels(single_label, "../data/queries_single_label.jsonl")
    save_labels(multi_label, "../data/queries_multi_label.jsonl")

    table.create_fts_index("text")


async def setup_metadata_example():
    db = lancedb.connect("../lance")
    dataset = download_arxiv_dataset(100)
    data = format_arxiv_dataset(dataset)
    save_labels(data, "../data/arxiv_metadata.jsonl")

    table = get_table(db, "arxiv_papers", ArxivPaper)
    insert_data_into_table(table, data)

    table.create_fts_index("text", replace=True)

    category_labels = await generate_category_questions(data, 10)
    category_labels = generate_category_test_labels(category_labels)
    save_labels(category_labels, "./data/category_questions.jsonl")


if __name__ == "__main__":
    setup_table()
    # run(setup_metadata_example())
