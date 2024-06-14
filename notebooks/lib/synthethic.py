import instructor
import openai
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm_asyncio as asyncio
from asyncio import Semaphore
from tenacity import retry, stop_after_attempt, wait_random_exponential
from lib.models import ArxivPaper

client = instructor.from_openai(openai.AsyncOpenAI())


class QuestionAnswerResponse(BaseModel):
    """
    This is a model that represents a sample question and answer that is derived from the given text chunk. It is useful in helping
    """

    chain_of_thought: str = Field(
        ...,
        description="The reasoning process leading to the answer and question being generated.",
    )
    question: str = Field(
        ..., description="The generated question from the text chunk."
    )
    answer: str = Field(..., description="The answer to the generated question.")


class Metadata(BaseModel):
    """
    This is a model which represents some metadata that we want to generate from a given text. 
    
    Make sure to expand on the text by extracting out any accronyms, context or phrases that users might search for later on \
    when trying to retrieve this specific chunk and model the metadata in a way that allows us to retrieve the most relevant chunks when searching for the query
    """

    keywords: list[str] = Field(
        ...,
        description="This is a field which represents keywords that a user might use to search for this text",
    )
    hypothetical_phrases: list[str] = Field(
        ...,
        description="This is a field which represents hypothetical phrases that a user might use to search for this text",
    )


async def generate_question_batch(
    text_chunk_batch, max_concurrent_calls: int, model_name="gpt-4o"
):
    sem = Semaphore(max_concurrent_calls)

    @retry(
        wait=wait_random_exponential(multiplier=1, min=10, max=90),
        stop=stop_after_attempt(3),
    )
    async def generate_question(text: str):
        async with sem:
            question = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a world class search engine. You are about to be passed a text chunk and your job is to generate a hypothetical question and answer pair that a user might ask to search for in order to retrieve the text chunk. Make sure to use information that is unique to the text chunk itself and also explain any sort of information/accronym that you use in the answer.",
                    },
                    {"role": "user", "content": f"Here is the text chunk : {text}"},
                ],
                response_model=QuestionAnswerResponse,
                max_retries=3,
            )
            return (question, text)

    coros = [generate_question(item) for item in text_chunk_batch]
    res = await asyncio.gather(*coros)
    return [{"response": item, "source": text} for item, text in res]


async def generate_category_questions(
    data: list[ArxivPaper], max_concurrent_calls: int, model_name="gpt-3.5-turbo"
):
    sem = Semaphore(max_concurrent_calls)

    @retry(
        wait=wait_random_exponential(multiplier=1, min=10, max=90),
        stop=stop_after_attempt(3),
    )
    async def generate_question(text: ArxivPaper):
        async with sem:
            question = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a world class question generator. You are about to be passed a text chunk. Your job is to generate a question and answer that will enable a user to find similar chunks. Make sure not to include the title of the text chunk within the question",
                    },
                    {"role": "user", "content": f"Here is the text chunk : {text}"},
                ],
                response_model=QuestionAnswerResponse,
                max_retries=3,
            )
            return (question, text)

    coros = [generate_question(item) for item in data]
    res = await asyncio.gather(*coros)
    return [{"response": item, "source": text} for item, text in res]


async def generate_metadata_batch(
    text_chunk_batch, max_concurrent_calls: int, model_name: str = "gpt-3.5-turbo"
):
    sem = Semaphore(max_concurrent_calls)

    @retry(
        wait=wait_random_exponential(multiplier=1, min=10, max=90),
        stop=stop_after_attempt(3),
    )
    async def enhance_query(text_chunk: str):
        async with sem:
            return (
                await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    response_model=Metadata,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a world class query indexing system. You are about to be passed a text chunk and you'll need to generate some metadata that will allow you to retrieve this specific chunk when the user makes a relevant query",
                        },
                        {"role": "user", "content": f"The text chunk is {text_chunk}"},
                    ],
                ),
                text_chunk,
            )

    coros = [enhance_query(item) for item in text_chunk_batch]
    res = await asyncio.gather(*coros)
    return [{"response": item, "source": text} for item, text in res]
