import logging
from enum import Enum
from dotenv import load_dotenv, find_dotenv
from rag.data_extractor import DataExtractor
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List
load_dotenv(find_dotenv())
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ChunkingType(Enum):
    SEMANTIC = "Semantic"
    NAIVE = "Naive"
    RECURSIVE = "Recursive"


class Chunking:
    """
    A class for chunking text data using different strategies.
    """

    def __init__(self, chunking_strategy:ChunkingType=ChunkingType.SEMANTIC) -> None:
        if chunking_strategy not in ChunkingType:
            raise ValueError("Invalid chunking strategy. Please use a valid ChunkingType enum value.")
        self.chunking_strategy = chunking_strategy
        self.data_extract_tool = DataExtractor()

    def chunk_data(self, file_path: str) -> List[Document]:
        """
        Extracts text data from a file and chunks it based on the chosen strategy.

        Args:
            file_path (str): The path to the file containing the text data.

        Returns:
            list[Document]: A list of Document objects representing the text chunks, or None if an error occurs.
        """


        try:
            text = self.data_extract_tool.extract_data(file_path)
            chunking_method = self._get_chunking_method()
            return chunking_method(text)
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return None

    def _get_chunking_method(self) -> callable:
        chunking_methods = {
            ChunkingType.NAIVE: self._naive_chunking,
            ChunkingType.SEMANTIC: self._semantic_chunking,
            ChunkingType.RECURSIVE: self._recursive_chunking
        }
        return chunking_methods.get(self.chunking_strategy)

    def _naive_chunking(self, text:str, chunk_size:int=500, chunk_overlap:int=50) -> List[Document]:
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.create_documents([text])

    def _recursive_chunking(self, text:str, chunk_size:int=500, chunk_overlap:int=50) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.create_documents([text])

    def _semantic_chunking(self, text:str) -> List[Document]:
        text_splitter = SemanticChunker(OpenAIEmbeddings())
        return text_splitter.create_documents([text])