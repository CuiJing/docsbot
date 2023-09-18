import os
import sys
from qdrant_client import QdrantClient

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant, Chroma
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain import OpenAI



from docsbot.config import CONFIG

COLLECTION_NAME = "docsbot_default"

class BaseDB:
    def add_documents(self, documents):
        pass

    def delete(self):
        pass


class BaseDB_Qdrant(BaseDB):
    def __init__(self,
                 collection_name,
                 embeddings=OpenAIEmbeddings()
                 ):
        if not hasattr(CONFIG.env, 'QDRANT_SERVER_URL'):
            raise Exception(f"QDRANT_SERVER_URL not set in {CONFIG.config_file}")
        self.url = CONFIG.env.QDRANT_SERVER_URL
        self.client = QdrantClient(url=self.url)
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.vector_store = Qdrant(self.client,
                                   collection_name,
                                   embeddings
                                   )

    def __str__(self):
        return f"[Qdrant]-{self.collection_name}@{self.url}"

    def add_documents(self, documents):
        self.vector_store.from_documents(documents,
                                         self.embeddings,
                                         url=self.url,
                                         collection_name=self.collection_name
                                         )

    def delete(self):
        self.client.delete_collection(collection_name=self.collection_name)


class BaseDB_Chroma(BaseDB):
    def __init__(self,
                 collection_name,
                 embeddings=OpenAIEmbeddings()):
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.vectors_dir = os.path.join(CONFIG.vectors_dir, 'Chroma')
        self.vector_store = Chroma(collection_name=self.collection_name,
                                   persist_directory=self.vectors_dir,
                                   embedding_function=self.embeddings
                                   )

    def __str__(self):
        return f"[Chroma]-{self.collection_name}@{self.vectors_dir}"

    def add_documents(self, documents):
        self.vector_store.from_documents(documents,
                                         embedding=self.embeddings,
                                         collection_name=self.collection_name,
                                         persist_directory=self.vectors_dir)
        self.vector_store.persist()


    def delete(self):
        self.vector_store.delete_collection()


class Base:
    def __init__(self, base_id, vector_store_type=None):
        self.base_id = base_id

        if vector_store_type:
            self.vector_store_type = vector_store_type
        elif not hasattr(CONFIG.env, 'VECTOR_STORE_TYPE'):
            self.vector_store_type = 'Chroma'
        elif CONFIG.env.VECTOR_STORE_TYPE == 'Chroma':
            self.vector_store_type = 'Chroma'
        elif CONFIG.env.VECTOR_STORE_TYPE == 'Qdrant':
            self.vector_store_type = 'Qdrant'
        else:
            raise Exception('VECTOR_STORE_TYPE must be either "Chroma" or "Qdrant"')


        if self.vector_store_type == 'Qdrant':
            self.base_db = BaseDB_Qdrant(collection_name=base_id)
        elif self.vector_store_type == 'Chroma':
            self.base_db = BaseDB_Chroma(collection_name=base_id)

        print(f"Using vector store:  {self.base_db} ")
        retriever = VectorStoreRetriever(vectorstore=self.base_db.vector_store,
                                         llm=OpenAI(temperature=0, max_tokens=300),
                                         search_kwargs=dict(k=2)  # TOP 2 results
                                         )
        memory_key = f"chat_history_{base_id}"
        memory_key = "chat_history"
        self.memory = ConversationBufferMemory(memory_key=memory_key,
                                          return_messages=True,
                                          input_key="question",
                                          output_key="answer",
                                          )

        # self.qa = RetrievalQA.from_llm(llm=OpenAI(temperature=0, max_tokens=300),
        #                                retriever=retriever,
        #                                return_source_documents=True,
        #                                memory=self.memory
        #                                )


        # 使用 ConversationalRetrievalChain 可以合并历史对话
        # combine_docs_chain 和 question_generator 使用默认的
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=OpenAI(temperature=0, max_tokens=300),
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True
        )

    def add(self, location):
        file_extensions = ('.docx', '.doc', '.pdf', '.pptx')
        documents = []
        try:
            # 每个文件会作为一个 document
            for ext in file_extensions:
                loader = DirectoryLoader(location, glob=f"**/*{ext}")
                documents += loader.load()
        except Exception as e:
            print(f"Load error from dir: {location} , error: {e}")
            raise e

        if not documents:
            return []
        else:
            print(f"Loaded {len(documents)} document(s)!")
        # 初始化加载器
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        # 切割加载的 document
        split_trunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} document(s) to {len(split_trunks)} trunk(s)!")

        self.base_db.add_documents(split_trunks)
        print(f"Embedding {len(split_trunks)} trunk(s) to vector store {self.base_db}!")
        return [i.metadata["source"] for i in documents]

    def delete(self):
        self.base_db.delete()

    def query(self, question):
        return self.qa({"question": question})

