from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

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
