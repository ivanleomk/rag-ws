from lancedb.table import Table
from tqdm import tqdm
from lib.openai_helpers import generate_embeddings
from lancedb.rerankers import LinearCombinationReranker, CohereReranker


def full_text_search(table: Table, queries, top_k):
    data = []
    for query in tqdm(queries):
        items = table.search(query["query"], query_type="fts").limit(top_k).to_list()
        data.append([item["chunk_id"] for item in items])
    return data


def semantic_search(table: Table, queries, top_k):
    data = []
    embedded_queries = generate_embeddings(queries, 20)
    for embedding in tqdm(embedded_queries):
        items = table.search(embedding, query_type="vector").limit(top_k).to_list()
        data.append([item["chunk_id"] for item in items])
    return data


def hybrid_search(table: Table, queries, top_k):
    data = []
    for query in tqdm(queries):
        items = table.search(query["query"], query_type="hybrid").limit(top_k).to_list()
        data.append([item["chunk_id"] for item in items])
    return data


def linear_combination_search(
    table: Table, queries, top_k: int, vector_search_weight: float
):
    reranker = LinearCombinationReranker(weight=vector_search_weight)
    data = []
    for query in tqdm(
        queries, desc=f"Linear Combination (Weight {vector_search_weight})"
    ):
        items = (
            table.search(query["query"], query_type="hybrid")
            .rerank(reranker=reranker)
            .limit(top_k)
            .to_list()
        )
        data.append([item["chunk_id"] for item in items])
    return data


def cohere_rerank_search(table: Table, queries, top_k: int, model_name: str):
    data = []
    cohere_reranker = CohereReranker(model_name=model_name)
    for query in tqdm(queries, desc=f"Cohere Reranker ({model_name})"):
        items = (
            table.search(query["query"], query_type="hybrid")
            .rerank(reranker=cohere_reranker)
            .limit(top_k)
            .to_list()
        )
        data.append([item["chunk_id"] for item in items])
    return data
