
from django.shortcuts import render, redirect
from django.urls import reverse
from django.contrib.auth import authenticate, logout, login as auth_login
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from . import viix_api
from itertools import zip_longest
from django.http import JsonResponse
import time


def main(request):
    return redirect(to="app_main:ai-search")
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


def chat_manager(request):
    return render(request, "app_main/chat-dev.html", context={
        "title": "Chat",
        "description": "Description chat"})


def chat_auditor(request):
    return render(request, "app_main/chat-dev.html", context={
        "title": "Chat",
        "description": "Description chat"})


def demo(request):
    return render(request, "app_main/demo.html", context={
        "title": "Demo font: Helvetica",
        "description": "Helvetica"})

def demo_0(request):
    return render(request, "app_main/demo-0.html", context={
        "title": "Demo fonts: Reith-Sans",
        "description": "Reith-Sans"})

def demo_1(request):
    return render(request, "app_main/demo-1.html", context={
        "title": "Demo fonts: Source-Sans-3",
        "description": "Source-Sans-3"})

def demo_2(request):
    return render(request, "app_main/demo-2.html", context={
        "title": "Demo fonts: Public-Sans",
        "description": "Public Sans"})

def demo_3(request):
    return render(request, "app_main/demo-3.html", context={
        "title": "Demo fonts: PT-Sans",
        "description": "PT Sans"})

def demo_4(request):
    return render(request, "app_main/demo-4.html", context={
        "title": "Demo fonts: Fira-Sans",
        "description": "Fira-Sans"})

def demo_5(request):
    return render(request, "app_main/demo-5.html", context={
        "title": "Demo fonts: Assistant",
        "description": "Assistant"})

def demo_6(request):
    return render(request, "app_main/demo-6.html", context={
        "title": "Demo fonts: Cabin",
        "description": "Cabin"})

def demo_7(request):
    return render(request, "app_main/demo-7.html", context={
        "title": "Demo fonts: Golos Text",
        "description": "Golos Text"})

def demo_8(request):
    return render(request, "app_main/demo-8.html", context={
        "title": "Demo fonts: PT-Sans-Caption",
        "description": "PT Sans Caption"})

def demo_9(request):
    return render(request, "app_main/demo-9.html", context={
        "title": "Demo fonts: Work-Sans",
        "description": "Work Sans"})

def demo_10(request):
    return render(request, "app_main/demo-10.html", context={
        "title": "Demo fonts: Nata-Sans",
        "description": "Nata Sans"})

def demo_11(request):
    return render(request, "app_main/demo-11.html", context={
        "title": "Demo fonts: Vendor-Sans",
        "description": "Vendor Sans"})

def demo_12(request):
    return render(request, "app_main/demo-12.html", context={
        "title": "Demo fonts: Tirra",
        "description": "Tirra"})

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
