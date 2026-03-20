from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Needed for OpenAI embeddings
    OPENAI_API_KEY: SecretStr | None = None

    # Local dev default is inside project workspace.
    # In Docker, set FAISS_DIR=/app/faiss/offers (volume-mounted path).
    FAISS_DIR: str = "data/faiss/offers"
    FAISS_INDEX_FILENAME: str = "offers.index"
    FAISS_IDMAP_FILENAME: str = "offers_id_map.json"
    FAISS_METADATA_FILENAME: str = "offers_meta.json"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
