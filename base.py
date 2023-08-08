import os

from qdrant_client import QdrantClient

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant, Chroma
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain import OpenAI

from config import CONFIG

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

        print(f"Using vector store:  {self.vector_store_type} ")
        if self.vector_store_type == 'Qdrant':
            self.base_db = BaseDB_Qdrant(collection_name=f"chatbase_{base_id}")
        elif self.vector_store_type == 'Chroma':
            self.base_db = BaseDB_Chroma(collection_name=f"chatbase_{base_id}")

        retriever = VectorStoreRetriever(vectorstore=self.base_db.vector_store,
                                         llm=OpenAI(temperature=0, max_tokens=300)
                                         )
        self.qa = RetrievalQA.from_llm(llm=OpenAI(temperature=0, max_tokens=300),
                                       retriever=retriever,
                                       return_source_documents=True
                                       )

    def _get_subdirectories(self, directory):
        subdirectories = [directory]
        for dirpath, dirnames, filenames in os.walk(directory):
            subdirectories.extend([os.path.join(dirpath, dir) for dir in dirnames])
        return subdirectories

    def add(self, location):
        documents = []
        for dir in self._get_subdirectories(location):
            print(f"loading from dir: {dir}")
            try:
                # 每个文件会作为一个 document
                pptx_loader = DirectoryLoader(dir, glob='*.pptx')
                documents += pptx_loader.load()

                docx_loader = DirectoryLoader(dir, glob='*.docx')
                documents += docx_loader.load()

                doc_loader = DirectoryLoader(dir, glob='*.doc')
                documents += docx_loader.load()

                pdf_loader = DirectoryLoader(dir, glob='*.pdf')
                documents += pdf_loader.load()
            except Exception as e:
                print(f"load error from dir: {dir} , error: {e}")
                raise e

        # 初始化加载器
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        # 切割加载的 document
        split_docs = text_splitter.split_documents(documents)

        self.base_db.add_documents(split_docs)
        return [i.metadata["source"] for i in documents]

    def delete(self):
        self.base_db.delete()

    def query(self, question):
        return self.qa({"query": question})
