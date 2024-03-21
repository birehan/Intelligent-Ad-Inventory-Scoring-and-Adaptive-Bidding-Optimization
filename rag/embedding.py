import enum
from langchain_community.embeddings import SentenceTransformerEmbeddings, OpenAIEmbeddings, CohereEmbeddings
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv,find_dotenv
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingType(enum.Enum):
    OPENAI_EMBEDDING = "OpenAI Embedding"
    SENTENCE_TRANSFORMER_EMBEDDING = "Sentence Transformers"
    COHERE_EMBEDDING = "Cohere Embeddings"

class EmbeddingFactory:
    """
    Provides a factory for creating embedding objects based on specified types and optional default models.
    """

    def __init__(self):
        load_dotenv(find_dotenv())
        self.embedding_classes = {
            EmbeddingType.SENTENCE_TRANSFORMER_EMBEDDING: (SentenceTransformerEmbeddings, "all-MiniLM-L6-v2"),  
            EmbeddingType.OPENAI_EMBEDDING: (OpenAIEmbeddings, "text-embedding-ada-002"), 
            EmbeddingType.COHERE_EMBEDDING: (CohereEmbeddings, "embed-english-light-v3.0"), 

        }

    def create_embedding(self, embedding_type: EmbeddingType=EmbeddingType.OPENAI_EMBEDDING, model: str = None) -> Embeddings:
        """
        Creates an embedding object of the specified type and optionally uses the provided model.

        Args:
            embedding_type (EmbeddingType): The type of embedding to create.
            model (str, optional): The specific model to use. Defaults to the type's default model.

        Returns:
            object: The created embedding object.

        Raises:
            ValueError: If the specified embedding type is not supported.
        """

        try:
            if embedding_type not in self.embedding_classes:
                raise ValueError(f"Unsupported embedding type: {embedding_type}")

            embedding_class, default_model = self.embedding_classes[embedding_type]
            if embedding_type == EmbeddingType.SENTENCE_TRANSFORMER_EMBEDDING:
                return embedding_class(model_name=model or default_model)
            else:
                return embedding_class(model=model or default_model) 
        except Exception as e:
            logging.error(f"Error creating embedding: {e}")
            return None