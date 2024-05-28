import lancedb
import json


def load_jsonl_file(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data


def get_ms_marco_table():
    db = lancedb.connect("../lance")
    return db.open_table("ms_marco")


def get_k_relevant_chunk_ids(table, query, number):
    return [
        item["chunk_id"]
        for item in table.search(query, query_type="fts").limit(number).to_list()
    ]


def get_test_queries():
    data = load_jsonl_file("../queries.jsonl")
    return [
        {
            "query": item["query"],
            "selected_chunk_id": item["selected_passages"][0]["chunk_id"],
        }
        for item in data
    ]
