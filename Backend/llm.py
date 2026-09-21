from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import (
    OPENAI_MODEL,
    EMBEDDING_MODEL
)


llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0
)


embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL
)