"""Application configuration and environment settings."""

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings loaded from environment or .env file."""

    APP_NAME: str = "ChatFreely Backend"
    APP_ENV: str = "development"
    DEBUG: bool = False
    PORT: int = 8000
    FORWARDED_ALLOW_IPS: str = "*"

    # Matchmaker & WebRTC STUN Configuration
    STUN_SERVERS: list[str] = [
        "stun:stun.l.google.com:19302",
        "stun:stun1.l.google.com:19302",
    ]

    # Moderation & Ban Policies
    MODERATION_ENABLED: bool = True
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

    @model_validator(mode="after")
    def moderation_must_remain_enabled_in_production(self):
        """Prevent an accidental environment toggle from disabling safety checks."""
        if self.APP_ENV.lower() in {"production", "prod"} and not self.MODERATION_ENABLED:
            raise ValueError("MODERATION_ENABLED cannot be false in production")
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
