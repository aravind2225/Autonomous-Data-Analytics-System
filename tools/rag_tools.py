from langchain.tools import tool
from vectorstore.vectordb import VectorStoreManager

VECTOR_DB = None


def initialize_vectorstore(df):

    global VECTOR_DB

    manager = VectorStoreManager()

    VECTOR_DB = manager.create_vectorstore(df)


@tool

def semantic_search_tool(query: str):
    """Perform semantic similarity search over dataset."""

    results = VECTOR_DB.similarity_search(query, k=5)

    return [doc.page_content for doc in results]