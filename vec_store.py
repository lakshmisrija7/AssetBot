from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')



import nest_asyncio
nest_asyncio.apply()

from langchain.document_loaders.sitemap import SitemapLoader

loader = SitemapLoader(
    "https://www.assetsense.com/sitemap.xml"
)
docs2 = loader.load()

text_splitter=CharacterTextSplitter(separator='\n',
                                    chunk_size=500,
                                    chunk_overlap=50)

text_chunks=text_splitter.split_documents(docs2)
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vecstore=FAISS.from_documents(text_chunks, embeddings)
DB_FAISS_PATH = '/content/drive/MyDrive/Colab Notebooks/demovs3'
vecstore.save_local(DB_FAISS_PATH)
