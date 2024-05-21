import json
import openai
from itertools import batched
from tqdm.asyncio import tqdm_asyncio as asyncio

client = openai.AsyncOpenAI()


def load_jsonl_file(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data


if __name__ == "__main__":
    data = load_jsonl_file("./queries.jsonl")
    print(data)
