
from django.urls import path
from . import views

app_name = "app_main"

urlpatterns = [
    path("", views.main, name="main"),
    path("ai-search",   views.ai_search,    name="ai-search"),
    path("signin",      views.login_view,   name="signin"),
    path("logout",      views.logout_view,  name="logout"),
    path("add-text",    views.add_text,     name="add-text"),
    path("add-page",    views.add_page,     name="add-page"),
    path("chat-dev",    views.chat_view,    name="chat-dev"),
    path("chat-manager",views.chat_manager, name="chat-manager"),
    path("chat-auditor",views.chat_auditor, name="chat-auditor"),
]
