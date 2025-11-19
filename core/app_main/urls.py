
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
    path("demo-fonts", views.demo, name="demo-fonts"),
    path("demo-0", views.demo_0, name="demo-0"),
    path("demo-1", views.demo_1, name="demo-1"),
    path("demo-2", views.demo_2, name="demo-2"),
    path("demo-3", views.demo_3, name="demo-3"),
    path("demo-4", views.demo_4, name="demo-4"),
    path("demo-5", views.demo_5, name="demo-5"),
    path("demo-6", views.demo_6, name="demo-6"),
    path("demo-7", views.demo_7, name="demo-7"),
    path("demo-8", views.demo_8, name="demo-8"),
    path("demo-9", views.demo_9, name="demo-9"),
    path("demo-10", views.demo_10, name="demo-10"),
    path("demo-11", views.demo_11, name="demo-11"),
    path("demo-12", views.demo_12, name="demo-12"),
]
