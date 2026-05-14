from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Initialize global variable
VECTOR_DB = None


def initialize_vectorstore(df, text_column: str = "text"):
    """Convert a Pandas DataFrame into LangChain Documents and load them into a vector store."""
    global VECTOR_DB

    # 1. Convert rows of the DataFrame into Document objects
    documents = [
        Document(
            page_content=str(row[text_column]),
            metadata={k: v for k, v in row.items() if k != text_column},
        )
        for _, row in df.iterrows()
    ]

    # 2. Define your embedding function (requires openai API key setup)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. Initialize and populate the vector store directly from documents
    VECTOR_DB = InMemoryVectorStore.from_documents(
        documents=documents, embedding=embeddings
    )


@tool
def semantic_search_tool(query: str) -> list[str]:
    """Perform semantic similarity search over dataset."""
    if VECTOR_DB is None:
        raise ValueError(
            "Vector store has not been initialized. Run initialize_vectorstore(df) first."
        )

    # Search the vector store and pull out top 5 matches
    results = VECTOR_DB.similarity_search(query, k=5)

    return [doc.page_content for doc in results]
