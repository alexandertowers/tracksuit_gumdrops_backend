from pydantic_settings import BaseSettings, validator

class Settings(BaseSettings):
    openai_api_key: str
    
    @classmethod
    @validator("openai_api_key")
    def validate_api_key(cls, v):
        if not v:
            raise ValueError("API key is required")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()