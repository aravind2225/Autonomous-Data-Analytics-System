from langchain_classic.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory


class ChatMemoryManager:

    def __init__(self):

        self.chat_history = InMemoryChatMessageHistory()

        self.memory = ConversationBufferMemory(
            chat_memory=self.chat_history,
            return_messages=True,
            memory_key="chat_history"
        )

    def get_memory(self):
        return self.memory

    def get_messages(self):
        return self.chat_history.messages