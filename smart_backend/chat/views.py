from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .logic.dialog_flow import DialogFlowManager  
from .models import Conversation, Message
import uuid

@api_view(['POST'])
def chat_message(request):
    user_input = request.data.get('message', '').strip()
    current_state = request.data.get('current_state', {})
    session_id = request.data.get('session_id')

    # Allow empty message only if current_state exists (session initialization)
    if not user_input and not current_state:
        return Response({'error': 'Message is required'}, status=status.HTTP_400_BAD_REQUEST)

    # Get or create conversation
    if not session_id:
        session_id = str(uuid.uuid4())
        conversation = Conversation.objects.create(session_id=session_id)
    else:
        conversation, created = Conversation.objects.get_or_create(session_id=session_id)

    # Save user message only if it's not empty
    if user_input:
        Message.objects.create(conversation=conversation, text=user_input, is_user=True)

    # Process message
    flow_manager = DialogFlowManager()
    bot_response = flow_manager.process_message(user_input or "", current_state)

    # Save bot response
    Message.objects.create(conversation=conversation, text=bot_response['text'], is_user=False)

    response_data = {
        'session_id': session_id,
        'bot_response': bot_response,
        'current_state': bot_response.get('state', current_state)
    }

    return Response(response_data)

@api_view(['GET'])
def get_conversation_history(request, session_id):
    try:
        conversation = Conversation.objects.get(session_id=session_id)
        messages = conversation.messages.all()
        
        history = [
            {
                'text': msg.text,
                'is_user': msg.is_user,
                'timestamp': msg.timestamp
            }
            for msg in messages
        ]
        
        return Response({'messages': history})
    except Conversation.DoesNotExist:
        return Response({'messages': []})
