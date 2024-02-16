from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
# import pinecone
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# URLs=[
#     'https://www.assetsense.com/',
#     "https://www.assetsense.com/power-generation/persona/overview.html",
#     "https://www.assetsense.com/power-generation/persona/executive-(vp,gm,cxo).html",
#     "https://www.assetsense.com/power-generation/persona/production-manager.html",
#     "https://www.assetsense.com/power-generation/persona/it-and-cio.html",
#     "https://www.assetsense.com/power-generation/asset-performance/asset-performance-overview.html",
#     "https://www.assetsense.com/power-generation/asset-performance/asset-performance.html",
#     "https://www.assetsense.com/power-generation/asset-performance/operator-rounds.html",
#     "https://www.assetsense.com/power-generation/asset-performance/predictive-maintenance.html",
#     "https://www.assetsense.com/power-generation/asset-performance/asset-maintenance.html",
#     "https://www.assetsense.com/platform/digital-transformation.html",
#     "https://www.assetsense.com/platform/industrial-iot.html",
#     "https://www.assetsense.com/platform/cloud-computing.html",
#     "https://www.assetsense.com/platform/data-analytics.html",
#     "https://www.assetsense.com/platform/mobile-apps.html",
#     "https://www.assetsense.com/platform/integrations.html",
#     "https://www.assetsense.com/platform/security.html"


# ]

# loaders=UnstructuredURLLoader(urls=URLs)
# data=loaders.load()

# data
# text_splitter=CharacterTextSplitter(separator='\n',
#                                     chunk_size=500,
#                                     chunk_overlap=50)

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