from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


class VectorStoreManager:

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def dataframe_to_documents(self, df):

        docs = []

        for _, row in df.iterrows():
            row_text = " | ".join(map(str, row.values))
            docs.append(row_text)

        return docs

    def create_vectorstore(self, df):

        docs = self.dataframe_to_documents(df)

        chunks = self.text_splitter.create_documents(docs)

        vectordb = Chroma.from_documents(
            chunks,
            embedding=embedding_model
        )

        return vectordb