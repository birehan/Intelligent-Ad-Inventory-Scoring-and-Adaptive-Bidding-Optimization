

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from dotenv import load_dotenv,find_dotenv
from rag.databases import VectorStoreDbType, VectorStoreDb
from rag.chunking import ChunkingType, Chunking
from rag.data_extractor import DataExtractor
from rag.retrivers import RetrieverFactory, RetrieverType
from rag.embedding import EmbeddingFactory, EmbeddingType
from rag.memory import MemoryType, MemoryFactory
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class Rag:
    def __init__(self, 
                 template_filepath="../prompts/system_message.txt",
                 context_filepath:str="",
                 vector_store:VectorStoreDbType=VectorStoreDbType.CHROMA,
                 retrieve_type:RetrieverType=RetrieverType.VECTOR_STORE_BACKED,
                 chunking_strategy:ChunkingType=ChunkingType.RECURSIVE,
                 embedding_model: EmbeddingType=EmbeddingType.OPENAI_EMBEDDING,
                 memory_type: MemoryType=MemoryType.CONVERSATION_BUFFER_WINDOW,
                 ):
        
        load_dotenv(find_dotenv())
        chunks = []
        self.chunking_tool = Chunking(chunking_strategy=chunking_strategy)
        if context_filepath:
            chunks = self.chunking_tool.chunk_data(context_filepath)
        
        self.chat_memory = MemoryFactory().create_memory(memory_type)
        self.embedding = EmbeddingFactory().create_embedding(embedding_model)
        self.template = DataExtractor.extract_data(template_filepath)
        self.vectorstore_factory = VectorStoreDb()
        self.vectorstore = self.vectorstore_factory.create_vectorstore(chunks=chunks, vector_store=vector_store, embedding=self.embedding)
        self.retriver_factory = RetrieverFactory()
        self.retriever = self.retriver_factory.create_retriever(vector_store= self.vectorstore, retrieve_type=retrieve_type, chunks=chunks)
        
        self.data_sources = {}
        self.executor = self.get_rag_chain()


    def add_datasource(self, file_path):
        try:
            chunks = self.chunking_tool.chunk_data(file_path)

            if chunks:
                ids = self.retriever.add_documents(chunks)
                self.data_sources[file_path] = ids
                logging.info("Data source added to the system successfully")
            else:
                logging.info("do data to add")
        
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return None 
        
    def remove_datasource(self, file_path):
        try:
            if file_path in self.chunks: 
                ids = self.chunks[file_path]
                self.vectorstore.delete(ids)
                del self.data_sources[file_path]
                logging.info("Data source removed from the system successfully")
            else:
                logging.info("no data with the given path to remove")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return None 
        
    def get_rag_chain(self):
        try:
            llm = ChatOpenAI(temperature=0.1, model = 'gpt-3.5-turbo')
            
            system_message_prompt = SystemMessagePromptTemplate.from_template(self.template)
            human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")

            conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.retriever,
            memory=self.chat_memory,
            combine_docs_chain_kwargs={
                "prompt": ChatPromptTemplate.from_messages(
                    [
                        system_message_prompt,
                        human_message_prompt,
                    ]
                ),
            },
            )

            logging.info("langchain with rag pipeline created successfully.")
            return conversation_chain

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return None 