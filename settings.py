from dotenv import load_dotenv
import os

load_dotenv()


class Settings:

    gemini_api_key: str = os.getenv("GEMINI_API_KEY")
    model_name: str = os.getenv("MODEL_NAME")
    qdrant_key: str = os.getenv("QDRANT_API_KEY")
    qdrant_url: str = os.getenv("QDRANT_URL")


sett = Settings()
