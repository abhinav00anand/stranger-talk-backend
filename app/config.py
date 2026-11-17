"""Application configuration and environment settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings loaded from environment or .env file."""

    APP_NAME: str = "ChatFreely Backend"
    APP_ENV: str = "development"
    DEBUG: bool = False
    PORT: int = 8000

    # Matchmaker & WebRTC STUN Configuration
    STUN_SERVERS: list[str] = [
        "stun:stun.l.google.com:19302",
        "stun:stun1.l.google.com:19302",
    ]

    # AI Fallback Companion
    AI_FALLBACK_TIMEOUT_SECONDS: float = 5.0
    AI_PROVIDER: str = "mock"  # "mock" | "gemini" | "groq"
    GEMINI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-3.8-flash"
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Moderation & Ban Policies
    MODERATION_ENABLED: bool = True
    MODERATION_PROVIDER: str = "mock"  # "mock" | "gemini" | "openai"
    MODERATION_FAIL_OPEN: bool = False
    TOXICITY_THRESHOLD: float = 0.8
    BAN_DURATION_SECONDS: int = 900

    # WebSocket transport and abuse controls
    WS_MAX_MESSAGE_BYTES: int = 262_144
    WS_RATE_LIMIT_MESSAGES: int = 60
    WS_RATE_LIMIT_WINDOW_SECONDS: float = 10.0

    # Optional exact browser-origin allowlist.
    ALLOWED_ORIGINS: list[str] = []
    ALLOWED_GIF_DOMAINS: list[str] = [
        "giphy.com",
        "*.giphy.com",
        "media.giphy.com",
        "i.giphy.com",
        "tenor.com",
        "*.tenor.com",
        "media.tenor.com",
        "c.tenor.com",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
