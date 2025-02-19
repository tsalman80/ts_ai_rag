import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

UPLOAD_FOLDER = "uploads"
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


BASE_DIR = Path(__file__).resolve().parent.joinpath("data")
TEMP_DIR = BASE_DIR.joinpath("tmp").as_posix()
LOCAL_VECTOR_STORE_DIR = BASE_DIR.joinpath("vector_store").as_posix()


# from langchain.globals import set_verbose, set_debug

# set_verbose(True)
# set_debug(True)
