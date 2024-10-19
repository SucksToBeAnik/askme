from pydantic_settings import SettingsConfigDict, BaseSettings
from pydantic import SecretStr


class Settings(BaseSettings):
    LANGCHAIN_TRACING_V2: bool
    LANGCHAIN_ENDPOINT: str
    LANGCHAIN_API_KEY: str
    GROQ_API_KEY: SecretStr
    HF_API_KEY: SecretStr

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


env_store = Settings()  # type: ignore
