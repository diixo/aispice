from pydantic import BaseModel
from typing import List
from urllib.parse import urlparse



class ParsingUrlModel(BaseModel):
    url_str: str
    tags_str: str
    categs_str: str

class StrRequestModel(BaseModel):
    str_request: str

class DialogueParams(BaseModel):
    dialogue_type: str
    message_str: str

class DomainItemModel(BaseModel):
    domain_id: int
    name: str
    url:  str
    url_icon: str
    description: str
    text: str
    tags: str
    slug: str
    ptype: str
    urls: List[str]

class ParsingDomainModel(BaseModel):
    parsing_domain: str
    parsing_type: int
    global_id: int
    parsing_id: int
    channel_id: int
    slug: str

class IndexPaginationModel(BaseModel):
    page_id: int
    page_size: int

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

class DomainIdModel(BaseModel):
    domain_id: int

class Message(BaseModel):
    role: str
    utterance: str

