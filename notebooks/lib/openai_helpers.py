from itertools import batched
from openai import Client
from tqdm import tqdm
from lib.models import QueryItem

client = Client()


def generate_embeddings(data: list[QueryItem], batch_size):
    batches = batched(data, batch_size)

    def generate_embeddings_batch(batch):
        res = client.embeddings.create(
            model="text-embedding-3-small", input=[item.query for item in list(batch)]
        )
        return res.data

    batched_embeddings = [generate_embeddings_batch(list(batch)) for batch in batches]

    res = []
    for embeddings in tqdm(
        batched_embeddings, desc=f"Generating Embeddings for {len(data)} queries"
    ):
        for embedding in embeddings:
            res.append(embedding.embedding)

    return res
