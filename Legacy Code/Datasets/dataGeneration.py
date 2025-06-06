from mirascope.core import openai
from openai import OpenAI
from pydantic import BaseModel

custom_client = OpenAI(api_key="dummy-key", base_url="http://localhost:11434/v1/")
llm_model = "llama3.2:latest"


@openai.call(llm_model, client=custom_client)
def recommend_book(genre: str) -> str:
    return f"Recommend a {genre} book"


recommendation = recommend_book("fantasy")
print(recommendation)
# Output: Here are some popular and highly-recommended fantasy books...


class Book(BaseModel):
    title: str
    author: str


@openai.call(llm_model, response_model=Book, client=custom_client)
def extract_book(text: str) -> str:
    return f"Extract {text}"


book = extract_book("The Name of the Wind by Patrick Rothfuss")
print("-" * 100, "\n", isinstance(book, Book))
print(book)
