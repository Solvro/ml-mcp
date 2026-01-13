from pydantic_settings import BaseSettings
from pydantic import SecretStr

class Settings(BaseSettings):
    OPENAI_API_KEY: SecretStr

    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: SecretStr

    OFFERS_TABLE_NAME: str = 'offers'

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
