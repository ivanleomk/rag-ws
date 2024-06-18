from lancedb.table import Table
from tqdm import tqdm
from lib.openai_helpers import generate_embeddings
from lancedb.rerankers import LinearCombinationReranker, CohereReranker
import instructor
import openai
from pydantic import BaseModel, Field
from typing import Literal
from tqdm.asyncio import tqdm_asyncio as asyncio
from lib.string_helpers import strip_punctuation
from lib.models import QueryItem


def fts_search(table: Table, queries: list[QueryItem], top_k: int):
    data = []
    for query in tqdm(queries, desc="Executing Full Text Search now..."):
        items = (
            table.search(strip_punctuation(query.query), query_type="fts")
            .limit(top_k)
            .to_list()
        )
        data.append(items)
    return data


def vector_search(
    table: Table, queries: list[QueryItem], top_k: int, batch_size: int = 20
):
    embedded_queries = generate_embeddings(queries, batch_size)
    return [
        table.search(query_embedding, query_type="vector").limit(top_k).to_list()
        for query_embedding in tqdm(
            embedded_queries, desc="Executing Vector Search now..."
        )
    ]


def hybrid_search(
    table: Table, queries: list[QueryItem], top_k: int, batch_size: int = 20
):
    return [
        table.search(strip_punctuation(query.query), query_type="hybrid")
        .limit(top_k)
        .to_list()
        for query in tqdm(queries, desc="Executing Hybrid Search now...")
    ]


def linear_combination_search(
    table: Table, queries, top_k: int, vector_search_weight: float
):
    reranker = LinearCombinationReranker(weight=vector_search_weight)
    return [
        table.search(strip_punctuation(query.query), query_type="hybrid")
        .rerank(reranker=reranker)
        .limit(top_k)
        .to_list()
        for query in tqdm(
            queries, desc=f"Linear Combination (Weight {vector_search_weight})"
        )
    ]


def cohere_rerank_search(
    table: Table, queries, top_k: int, model_name: str, query_type="fts"
):
    cohere_reranker = CohereReranker(model_name=model_name)
    return [
        table.search(strip_punctuation(query.query), query_type=query_type)
        .rerank(reranker=cohere_reranker)
        .limit(top_k)
        .to_list()
        for query in tqdm(queries, desc=f"Cohere Reranker ({model_name})")
    ]


async def metadata_search(table: Table, queries: list[QueryItem], top_k):
    data = []
    categories = await classify_queries(queries)
    for query, category in tqdm(zip(queries, categories)):
        items = (
            table.search(strip_punctuation(query.query), query_type="fts")
            .where(f"category = '{category.category}'", prefilter=True)
            .limit(top_k)
            .to_list()
        )
        data.append(items)
    return data


async def classify_queries(queries: list[str]):
    client = instructor.from_openai(openai.AsyncOpenAI())

    category_description = """
    This represents a categorization of the user's query

    - stat.ML : Covers machine learning papers (supervised, unsupervised, semi-supervised learning, graphical models, reinforcement learning, bandits, high dimensional inference, etc.) with a statistical or theoretical grounding
    - cs.AI : Covers all areas of AI except Vision, Robotics, Machine Learning, Multiagent Systems, and Computation and Language (Natural Language Processing), which have separate subject areas. In particular, includes Expert Systems, Theorem Proving (although this may overlap with Logic in Computer Science), Knowledge Representation, Planning, and Uncertainty in AI. Roughly includes material in ACM Subject Classes I.2.0, I.2.1, I.2.3, I.2.4, I.2.8, and I.2.11.
    - cs.IR : Covers indexing, dictionaries, retrieval, content and analysis. Roughly includes material in ACM Subject Classes H.3.0, H.3.1, H.3.2, H.3.3, and H.3.4.
    """.strip()

    class Category(BaseModel):
        category: Literal["cs.AI", "cs.IR", "stat.ML"] = Field(
            ..., description=category_description
        )

    async def classify_query(query: str):
        return await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert topic classifier, your job is to classify the following title into the categories provided in the response object. Make sure to classify it into one of the individual categories provided",
                },
                {"role": "user", "content": f"The title is {query}"},
            ],
            model="gpt-4o",
            response_model=Category,
        )

    coros = [classify_query(query) for query in queries]
    results = await asyncio.gather(*coros)
    return results
