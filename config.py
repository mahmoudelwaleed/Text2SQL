import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
import qdrant_client
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase

# --- Environment Setup & Global Variables ---
load_dotenv()

QDRANT_HOST = os.getenv("qdrant_host")
QDRANT_API_KEY = os.getenv("qdrant_api_key")
QDRANT_COLLECTION_NAME = os.getenv("qdrant_collection_name", "your_default_collection_name")

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "false")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME2 = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME2", AZURE_OPENAI_CHAT_DEPLOYMENT_NAME)

if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME]):
    raise ValueError("Azure OpenAI credentials are not fully configured in environment variables.")

embedding_model = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
)
llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    temperature=0,
    streaming=False,
)
sql_generation_llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME2,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    temperature=0,
    streaming=False,
)
natural_language_llm = llm

qdrant_client_instance = None
vector_store = None
if QDRANT_HOST and QDRANT_API_KEY and QDRANT_COLLECTION_NAME != "your_default_collection_name":
    try:
        qdrant_client_instance = qdrant_client.QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
        vector_store = Qdrant(client=qdrant_client_instance, collection_name=QDRANT_COLLECTION_NAME, embeddings=embedding_model)
    except Exception:
        vector_store = None

DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
db = None
if DB_CONNECTION_STRING:
    try:
        sql_alchemy_engine = create_engine(DB_CONNECTION_STRING)
        db = SQLDatabase(engine=sql_alchemy_engine)
    except Exception:
        db = None
