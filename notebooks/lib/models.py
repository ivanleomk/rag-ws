from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from typing import Literal
from pydantic import Field

func = get_registry().get("openai").create(name="text-embedding-3-small")


class EmbeddedPassage(LanceModel):
    vector: Vector(dim=func.ndims()) = func.VectorField()  # type: ignore
    chunk_id: str
    text: str = func.SourceField()


class EmbeddedPassageWithQA(LanceModel):
    vector: Vector(func.ndims()) = func.VectorField()
    chunk_id: str
    text: str = func.SourceField()
    source_text: str


class EmbeddedPassageWithMetadata(LanceModel):
    vector: Vector(func.ndims()) = func.VectorField()
    chunk_id: str
    text: str = func.SourceField()
    keywords: str
    search_queries: str


class ArxivPaper(LanceModel):
    title: str
    authors: str
    category: str
    abstract: str
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField(default=None)
    chunk_id: str


class QueryItem(LanceModel):
    """
    This is a Pydantic class representing a single MS-Marco query and it's corresponding chunk(s) that we should be retrieving when working with the MS-Marco dataset.
    """

    query: str
    selected_chunk_ids: list[str]


class Capability(LanceModel):
    """
    This is a model representing an evaluation of the required capabilities to execute a task.

    A capability here is a brief 2-3 word phrase that describes a tool or function that is needed to fulfil a user request
    """

    capabilities: list[
        Literal[
            "Product Information Retrieval",
            "Flight Information Retrieval",
            "Restaurant Recomendations",
            "Search Email",
            "Retrieve Calendar",
            "Latest News",
            "Historical Price Information",
        ]
    ] = Field(
        ...,
        description="This is a list of capabilities that must be provided in order to execute/respond to the user's query",
    )


class QueryTagger(LanceModel):
    capabilities: list[str]
    topic_model: int
    query: str
