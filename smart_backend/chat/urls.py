from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat_message, name='chat_message'),
    path('chat/<str:session_id>/history/', views.get_conversation_history, name='conversation_history'),
]