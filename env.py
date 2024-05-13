from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HF_TOKEN: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
