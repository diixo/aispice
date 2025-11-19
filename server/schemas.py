from pydantic import BaseModel
from typing import List
from urllib.parse import urlparse


class StrRequestModel(BaseModel):
    str_request: str

class DialogueParams(BaseModel):
    dialogue_type: str
    message_str: str


class ContentItemModel(BaseModel):
    url:  str
    heading: str
    description: str
    icon_url: str
    hostname: str
    hostname_slug: str
    distance: str
    img_url: str
    date: str
    tags: List[str]

class DialogueModel(BaseModel):
    assistant: List[str]
    user: List[str]

class Message(BaseModel):
    role: str
    utterance: str

