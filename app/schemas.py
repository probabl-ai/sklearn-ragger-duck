"""Schemas for the chat app."""
from pydantic import BaseModel, validator


class WSMessage(BaseModel):
    """Websocket Message schema."""

    sender: str
    message: str
    type: str

    @validator("sender")
    def sender_must_be_bot_or_you(cls, v):
        if v not in ["bot", "you"]:
            raise ValueError("sender must be bot or you")
        return v

    @validator("type")
    def validate_message_type(cls, v):
        if v not in [
            "question",
            "start",
            "restart",
            "stream",
            "end",
            "error",
            "info",
            "system",
            "done",
        ]:
            raise ValueError("type must be start, stream or end")
        return v
