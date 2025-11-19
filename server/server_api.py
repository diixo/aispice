import uvicorn
import sys
import os
import json
import requests
from collections import deque
from fastapi import FastAPI
from server.searching_server import SearchingServer
from server.schemas import StrRequestModel, ContentItemModel, DialogueParams, Message
from server.dialogue import Dialogue_gpt2, Conversation
from typing import List, Optional
import logging


app = FastAPI()

searching_server_global = SearchingServer()

conversations = {
    "developer": Conversation(system_prompt="You are developer Assistant."),
    "manager": Conversation(system_prompt="You are project-manager Assistant."),
    "auditor": Conversation(system_prompt="You are auditor Assistant."),
}

dialogue_dev = Dialogue_gpt2()

dialogue_dev.handle_user_message(conversations["developer"])
dialogue_dev.handle_user_message(conversations["manager"])
dialogue_dev.handle_user_message(conversations["auditor"])

######################################################
def searching_server():
    global searching_server_global
    return searching_server_global


@app.post("/ai-search", response_model=List[ContentItemModel])
async def ai_search(input_request: StrRequestModel):
    return searching_server().search(input_request.str_request)


@app.post("/page-to-index")
async def page_to_index(input_request: StrRequestModel):
    searching_server().page_to_index(input_request.str_request)
    return { "status": "200" }


@app.post("/text-to-index")
async def text_to_index(input_request: StrRequestModel):
    searching_server().text_to_index(input_request.str_request)
    return { "status": "200" }


@app.post("/new-message")
async def new_message(dialogue: DialogueParams):
    if dialogue.dialogue_type in {"developer", "manager", "auditor",}:
        print("new message:", dialogue.message_str)
        dialogue_dev.handle_user_message(
            conversations[dialogue.dialogue_type],
            dialogue.message_str
            )
    return { "status": "200" }


@app.post("/get-last-answer", response_model=Optional[Message])
async def get_last_answer(dialogue: DialogueParams):
    if dialogue.dialogue_type in {"developer", "manager", "auditor",}:
        return dialogue_dev.get_last_answer(
            conversations[dialogue.dialogue_type]
            )
    else:
        return None


@app.post("/get-dialogue", response_model=List[Message])
async def get_dialogue(dialogue: DialogueParams):
    if dialogue.dialogue_type in {"developer", "manager", "auditor",}:
        return dialogue_dev.get_messages(
            conversations[dialogue.dialogue_type]
            )
    else:
        return []


@app.post("/clear-dialogue")
async def clear_dialogue(dialogue: DialogueParams):
    if dialogue.dialogue_type in {"developer", "manager", "auditor",}:
        #print("::clear_dialogue:", dialogue.dialogue_type)
        dialogue_dev.clear(
            conversations[dialogue.dialogue_type]
            )
    return { "status": "200" }


if __name__ == "__main__":
    try:
        uvicorn.run(
            "server_api:app",
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8001)),
            log_level="info")
    except Exception as e:
        logging.error("An error occurred", exc_info=True)
        print(f"Exception with force stop: {e}", file=sys.stderr)
        #parsing_server_global.stop()
