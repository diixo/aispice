
from django.shortcuts import render, redirect
from django.urls import reverse
from django.contrib.auth import authenticate, logout, login as auth_login
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from . import viix_api
from itertools import zip_longest
from django.http import JsonResponse
import json
import time
from .viix_api import make_get_request, make_post_request
from asgiref.sync import async_to_sync


def main(request):
    return redirect(to="app_main:confluence")
    return render(request, "app_main/index.html", context={
        "title": "Viix AI-search for AI-tools",
        "description": "AI-search for AI-tools. Combined with AI, it can revolutionize the workplace. Viix brings comprehensive, accurate, and search-based AI"})


def chat_view(request):
    api = viix_api.get_api()
    if request.method == "POST":
        if request.POST.get("send_action_btn"):
            query = request.POST.get("query", "").strip()
            api.new_message("developer", query)
            answer = api.get_last_answer("developer")
            time.sleep(1.0)
            return JsonResponse({"response": answer["utterance"]})
        if request.POST.get("create_new_chat"):
            api.clear_dialogue("developer")
            dialogue = api.get_dialogue("developer")
            return JsonResponse({"status": "ok", "messages": dialogue})

    dialogue = api.get_dialogue("developer")

    return render(request, "app_main/chat-dev.html", context = {
        "messages": dialogue,
        })


def confluence(request):
    headingBackend = "collapse"
    headingEval = "collapse"
    headingPick = "collapse"
    headingPull = "collapse"
    headingBuild = "collapse"

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "save":
            confluence_host = request.POST.get("confluence_host_field", "")
            pat = request.POST.get("pat_field", "")
            headingBackend = ""
            print(action, confluence_host, pat)

            raw_body, status = async_to_sync(make_post_request)(
                url="http://127.0.0.1:8001/confluence/pat/save", data=
                {
                    "confluence_host": confluence_host,
                    "token": pat,
                },
            )
            body = json.loads(raw_body)
            print(status, body)

        elif action == "test":
            confluence_host = request.POST.get("confluence_host_field", "")
            pat = request.POST.get("pat_field", "")
            headingBackend = ""
            print(action, confluence_host, pat)

            raw_body, status = async_to_sync(make_post_request)(
                url="http://127.0.0.1:8001/confluence/pat/test", data=
                {
                    "confluence_host": confluence_host,
                    "token": pat,
                    "root_page_id": None,
                },
            )
            body = json.loads(raw_body)
            print(status, body)

        elif action == "evaluate":
            headingEval = ""
            print(action)

        else:
            print("unknown action...")

    return render(request, "app_main/confluence.html", context={
        "title": "Confluence settings",
        "description": "Confluence",
        "headingBackend": headingBackend,
        "headingEval": headingEval,
        "headingPick": headingPick,
        "headingPull": headingPull,
        "headingBuild": headingBuild,
        })


def ai_search(request):
    api = viix_api.get_api()
    result = None
    results_amount = ""
    query = ""

    if request.method == "POST":
        if request.POST.get("search_action_btn"):
            query = request.POST.get("query", "")

            if query:
                result = api.ai_search(query)
                results_amount = str(len(result)) if result is not None else "0"
                print(f"ai_search query:{query}, result.sz={results_amount}")

    return render(request, "app_main/ai-search.html", context={
        "title": "Viix Search engine powered by AI",
        "description": "Search engine powered by AI. Tired of ads clogging up your search results? Get Answers. Not ads.",
        "content_list": result,
        "results_amount": results_amount,
        "search_request": query,
        "classify_categories": []
        })


def login_view(request):
    if request.user.is_authenticated:
        return redirect(to="app_main:main")

    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            auth_login(request, user)
            return redirect(to="app_main:main")
        else:
            messages.error(request, "Wrong username or password")
    return render(request, "app_main/signin.html")


@login_required
def logout_view(request):
    logout(request)
    return redirect(to="app_main:main")
    #return redirect(to="app_main:signin")


def add_text(request):
    api = viix_api.get_api()
    if request.method == "POST":
        button_value = request.POST.get("submit_txt_btn")

        if button_value:
            txt = request.POST.get("input_txt_field", "")
            print(f"add_text:{txt}")
            api.text_to_index(txt)
            return redirect(to="app_main:main")
            #return redirect(to="app_main:ai-search")

    return render(request, "app_main/add-text.html", context={
        "title": "viix add_text: AI-search",
        "description": "add_text: viix AI-search for AI-tools"})


def add_page(request):
    api = viix_api.get_api()
    if request.method == "POST":
        button_value = request.POST.get("submit_url_btn")

        if button_value:
            txt = request.POST.get("input_url_field", "")
            print(f"add_page:{txt}")
            api.page_to_index(txt)
            return redirect(to="app_main:main")
            #return redirect(to="app_main:ai-search")

    return redirect(to="app_main:main")
    return render(request, "app_main/add-page.html", context={
        "title": "viix add_page: AI-search",
        "description": "add_page: viix AI-search for AI-tools"})
