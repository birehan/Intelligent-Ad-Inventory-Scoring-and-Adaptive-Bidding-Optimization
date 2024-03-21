import enum
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    ConversationKGMemory,
)
from langchain.memory.chat_memory import BaseChatMemory
from langchain_openai import OpenAI
from dotenv import load_dotenv,find_dotenv
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class MemoryType(enum.Enum):
    CONVERSATION_BUFFER = "Conversation Buffer"
    CONVERSATION_BUFFER_WINDOW = "Conversation Buffer Window"
    CONVERSATION_SUMMARY = "Conversation Summary"
    CONVERSATION_SUMMARY_BUFFER = "Conversation Summary Buffer"
    CONVERSATION_KNOWLEDGE_GRAPH = "Conversation Knowledge Graph"


class MemoryFactory:
    """
    Provides a factory for creating memory objects based on specified types.

    This factory maintains a mapping of supported memory types to their corresponding implementation classes.
    """
    load_dotenv(find_dotenv())

    def __init__(self):
        self.memory_classes = {
            MemoryType.CONVERSATION_BUFFER: ConversationBufferMemory,
            MemoryType.CONVERSATION_BUFFER_WINDOW: ConversationBufferWindowMemory,
            MemoryType.CONVERSATION_SUMMARY: ConversationSummaryMemory,
            MemoryType.CONVERSATION_SUMMARY_BUFFER: ConversationSummaryBufferMemory,
            MemoryType.CONVERSATION_KNOWLEDGE_GRAPH: ConversationKGMemory,
        }

    def create_memory(self, memory_type: MemoryType=MemoryType.CONVERSATION_BUFFER_WINDOW) -> BaseChatMemory:
        """
        Creates a memory object of the specified type.

        Args:
            memory_type (MemoryType): The type of memory to create.

        Returns:
            object: The created memory object.

        Raises:
            ValueError: If the specified memory type is not supported.
        """
        try:
            if memory_type not in self.memory_classes:
                raise ValueError(f"Unsupported memory type: {memory_type}")

            if memory_type in [MemoryType.CONVERSATION_SUMMARY, MemoryType.CONVERSATION_SUMMARY_BUFFER, MemoryType.CONVERSATION_KNOWLEDGE_GRAPH]:
                return self.memory_classes[memory_type](llm=OpenAI(temperature=0), memory_key="chat_history", return_messages=True)
            
            return self.memory_classes[memory_type](memory_key="chat_history", return_messages=True) 
        except Exception as e:
            logging.error(f"Error creating memory: {e}")
            return None