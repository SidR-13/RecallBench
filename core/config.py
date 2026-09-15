from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    # Defaults on so a fresh checkout can never bill the Claude API by
    # accident -- same convention as BuildBoard/PromptGate.
    ai_mock: bool = True
    claude_model: str = "claude-haiku-4-5-20251001"

    class Config:
        env_file = ".env"


settings = Settings()
