from lib.models import QueryTagger, Capability, QueryItem
from tenacity import retry, stop_after_attempt, wait_random_exponential
from asyncio import Semaphore
import instructor
from openai import AsyncOpenAI
from typing import Callable
from tqdm.asyncio import tqdm_asyncio as asyncio

client = instructor.from_openai(AsyncOpenAI())


@retry(
    wait=wait_random_exponential(multiplier=1, min=10, max=90),
    stop=stop_after_attempt(3),
)
async def tag_query(
    query: str, sem: Semaphore, topic_model: Callable[[str], int]
) -> QueryTagger:
    async with sem:
        resp = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=Capability,
            messages=[
                {
                    "role": "system",
                    "content": "You are an advanced tagging system that excels at extracting what capabilites from the provided list \
                    need to be used in order to answer a user's query.",
                },
                {"role": "user", "content": f"The query is '{query}'"},
            ],
            max_retries=3,
        )
        return QueryTagger(
            **{
                "capabilities": resp.capabilities,
                "topic_model": topic_model(query),
                "query": query,
            }
        )


async def tag_queries(
    queries: list[QueryItem],
    max_concurrent_calls: int,
    topic_model: Callable[[str], int],
):
    sem = Semaphore(max_concurrent_calls)
    coros = [tag_query(item.query, sem, topic_model) for item in queries]
    res = await asyncio.gather(*coros)
    return res
