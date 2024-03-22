
from dotenv import load_dotenv, find_dotenv
import weaviate
from weaviate.embedded import EmbeddedOptions
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate, Milvus, Pinecone, Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from enum import Enum
from typing import List
import os
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStoreDbType(Enum):
    CHROMA = "Chroma"
    MILVUS = "Milvus"
    WEAVIATE = "Weaviate"
    PINECONE = "Pinecone"

class VectorStoreDb:
    """
    A class for creating and interacting with vector stores for storing and retrieving text embeddings.
    """
       
    def __init__(self):
        # Load OpenAI API key from .env file
        load_dotenv(find_dotenv())
        
    def create_vectorstore(self, chunks:List[Document]=[], vector_store: VectorStoreDbType=VectorStoreDbType.CHROMA, embedding:Embeddings=OpenAIEmbeddings()) -> VectorStore:
        """
        Creates a vector store based on the specified type and populates it with text embeddings.

        Args:
            chunks (list[Document]): A list of Documents representing the text data (type depends on the chosen vector store).
            vector_store (VectorStoreType, optional): The type of vector store to create. Defaults to VectorStoreType.MILVUS.
            embedding (langchain_community.embeddings.BaseEmbedding, optional): The embedding model to use for generating text embeddings. Defaults to OpenAIEmbeddings.

        Returns:
            object: An instance of the chosen vector store class, or None if an error occurs.
        """
           
        try:
            creation_methods = {
                VectorStoreDbType.CHROMA: self._create_chroma_db_vectorstore,
                VectorStoreDbType.WEAVIATE: self._create_weaviate_vectorstore,
                VectorStoreDbType.MILVUS: self._create_milvus_vectorstore,
                VectorStoreDbType.PINECONE: self._create_pinecone_vectorstore
            }
            if vector_store in creation_methods:
                return creation_methods[vector_store](chunks, embedding)
            else:
                logging.error("Invalid vector store type provided.")
                return None
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return None

    def _create_milvus_vectorstore(self, chunks:List[Document], embedding: Embeddings) -> VectorStore:
        return Milvus.from_documents(chunks, embedding)
    
    def _create_weaviate_vectorstore(self, chunks:List[Document], embedding: Embeddings) -> VectorStore:
        client = weaviate.Client(embedded_options=EmbeddedOptions())
        return Weaviate.from_documents(client=client, documents=chunks, embedding=embedding, by_text=False)
    
    def _create_pinecone_vectorstore(self, chunks:List[Document], embedding: Embeddings) -> VectorStore:
        index_name = os.getenv("PINECONE_INDEX_NAME")
        return Pinecone.from_documents(documents=chunks, embedding=embedding, index_name=index_name)
    
    def _create_chroma_db_vectorstore(self, chunks:List[Document], embedding: Embeddings) -> VectorStore:
        return Chroma(embedding_function=embedding)