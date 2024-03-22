from dotenv import load_dotenv,find_dotenv
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAI
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_openai import ChatOpenAI
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from enum import Enum
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from typing import List
import logging
from langchain_core.retrievers import BaseRetriever
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class RetrieverType(Enum):
    VECTOR_STORE_BACKED = "VectorStore Backed"
    MULTIQUERY = "Multiquery"
    CONTEXTUAL_COMPRESSION = "Contextual Compression"
    PARENT_DOCUMENT = "Parent Document"
    ADVANCED_RETRIEVER = "Advanced Retriever"
    

class RetrieverFactory:
    """
    Provides a factory for creating retriever objects based on specified types and configurations.
    """
    
    def __init__(self):
        load_dotenv(find_dotenv())
        self.creation_methods = {
            RetrieverType.VECTOR_STORE_BACKED: self._create_vectorstore_backend_retriver,
            RetrieverType.MULTIQUERY: self._create_multiquery_retriver,
            RetrieverType.CONTEXTUAL_COMPRESSION: self._create_compression_retriver,
            RetrieverType.PARENT_DOCUMENT: self._create_parent_document_retriver,
            RetrieverType.ADVANCED_RETRIEVER: self._create_advanced_retriver
        }

        
    def create_retriever(self, vector_store:VectorStore, retrieve_type: RetrieverType = RetrieverType.VECTOR_STORE_BACKED, chunks: List[Document] = []) -> BaseRetriever:
        """
        Creates a retriever object based on the specified type and configuration.

        Args:
            vector_store: The vector store used by the retriever.
            retrieve_type (RetrieverType): The type of retriever to create.
            chunks (list, optional): List of text chunks. Required for creating an advanced retriever. Defaults to [].

        Returns:
            object: The created retriever object.

        Raises:
            ValueError: If the specified retriever type is not supported.
        """
        try:
            creation_method = self.creation_methods.get(retrieve_type)
            if creation_method:
                return creation_method(vector_store, chunks)
            else:
                logging.error("Invalid retriever type provided.")
                return None
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return None

    def _create_vectorstore_backend_retriver(self, vector_store:VectorStore, chunks:List[Document]) -> BaseRetriever:
        return  vector_store.as_retriever()
    
    def _create_multiquery_retriver(self, vector_store: VectorStore, chunks:List[Document]) -> BaseRetriever:
        llm = ChatOpenAI(temperature=0)
        retriever = MultiQueryRetriever.from_llm(
            retriever = self._create_vectorstore_backend_retriver(vector_store), 
            llm=llm
        )
        return retriever 
    
    def _create_compression_retriver(self, vector_store: VectorStore, chunks:List[Document]) -> BaseRetriever:
        llm = OpenAI(temperature=0)
        compressor = LLMChainExtractor.from_llm(llm)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=self._create_vectorstore_backend_retriver(vector_store)
        )
        return retriever
    
    def _create_parent_document_retriver(self, vector_store: VectorStore, chunks:List[Document]) -> BaseRetriever:
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        store = InMemoryStore()
        retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        )
        return retriever
    
    def _create_advanced_retriver(self, vector_store:VectorStore, chunks:List[Document]) -> BaseRetriever:
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k=10
        vs_retriever = vector_store.as_retriever(search_kwargs={"k":10})
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever,vs_retriever],
                                            weight=[0.5,0.5])
        return ensemble_retriever