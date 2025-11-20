import json
from urllib.parse import urlparse
import requests
from pathlib import Path
import aiohttp


class ViixApi:

    def __init__(self):
        pass


    def ai_search(self, search_request: str):
        url = "http://127.0.0.1:8001/ai-search"

        try:
            #response = requests.post(url, headers={'Content-type': 'application/json'}, data=json.dumps(params))
            response = requests.post(url, json={ "str_request": search_request })
            response.raise_for_status()
            ###
            return response.json()
        except requests.RequestException as e:
            #return {"error": f"RequestException: {e}"}
            return None


    def page_to_index(self, page_request: str):
        url = "http://127.0.0.1:8001/page-to-index"

        try:
            #response = requests.post(url, headers={'Content-type': 'application/json'}, data=json.dumps(params))
            response = requests.post(url, json={ "str_request": page_request })
            response.raise_for_status()
            ###
            return response.json()
        except requests.RequestException as e:
            return {"error": f"RequestException: {e}"}


    def text_to_index(self, txt_request: str):
        url = "http://127.0.0.1:8001/text-to-index"

        try:
            #response = requests.post(url, headers={'Content-type': 'application/json'}, data=json.dumps(params))
            response = requests.post(url, json={ "str_request": txt_request })
            response.raise_for_status()
            ###
            return response.json()
        except requests.RequestException as e:
            return {"error": f"RequestException: {e}"}


    def new_message(self, dialogue_type: str, message: str):
        url = "http://127.0.0.1:8001/new-message"

        try:
            response = requests.post(url, json={ "dialogue_type": dialogue_type, "message_str": message })
            response.raise_for_status()
            ###
            return response.json()
        except requests.RequestException as e:
            return {"error": f"RequestException: {e}"}


    def get_last_answer(self, dialogue_type: str):
        url = "http://127.0.0.1:8001/get-last-answer"

        try:
            response = requests.post(url, json={ "dialogue_type": dialogue_type, "message_str": ""  })
            response.raise_for_status()
            ###
            return response.json()
        except requests.RequestException as e:
            return {"error": f"RequestException: {e}"}


    def get_dialogue(self, dialogue_type: str):
        url = "http://127.0.0.1:8001/get-dialogue"

        try:
            response = requests.post(url, json={ "dialogue_type": dialogue_type, "message_str": "" })
            response.raise_for_status()
            ###
            return response.json()
        except requests.RequestException as e:
            return {"error": f"RequestException: {e}"}


    def clear_dialogue(self, dialogue_type: str):
        url = "http://127.0.0.1:8001/clear-dialogue"

        try:
            response = requests.post(url, json={ "dialogue_type": dialogue_type, "message_str": "" })
            response.raise_for_status()
            ###
            return response.json()
        except requests.RequestException as e:
            return {"error": f"RequestException: {e}"}

######################################################################

async def get_request(session, url, params):
    async with session.get(url, params=params) as response:
        return await response.content.read(), response.status


async def make_get_request(url, params=None):
    if not params:
        params = {}
    async with aiohttp.ClientSession() as session:
        return await get_request(session, url, params=params)


async def post_request(session, url, data):
    async with session.post(url, json=data) as response:
        return await response.content.read(), response.status


async def make_post_request(url, data=None):
    if not data:
        data = {}
    async with aiohttp.ClientSession() as session:
        return await post_request(session, url, data=data)


async def put_request(session, url, data):
    async with session.put(url, data=data) as response:
        return await response.content.read(), response.status


async def make_put_request(url, data=None):
    if not data:
        data = {}
    async with aiohttp.ClientSession() as session:
        return await put_request(session, url, data=data)


async def delete_request(session, url):
    async with session.delete(url) as response:
        return await response.content.read(), response.status


async def make_delete_request(url):
    async with aiohttp.ClientSession() as session:
        return await delete_request(session, url)

######################################################################
viix_api = ViixApi()


def get_api():
    global viix_api
    return viix_api
