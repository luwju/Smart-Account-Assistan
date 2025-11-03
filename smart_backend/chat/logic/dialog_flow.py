# dialog_flow.py
import re
import json
import pickle
import numpy as np
from difflib import SequenceMatcher
from datetime import datetime
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class BankingChatbotTrainer:
    def __init__(self):
        self.model = None
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        self.classes = []
        self.intent_examples = defaultdict(list)
        
    def add_example(self, intent, example):
        """Add training example"""
        self.intent_examples[intent].append(example.lower())
    
    def prepare_training_data(self):
        """Prepare training data from intent examples"""
        texts = []
        labels = []
        
        for intent, examples in self.intent_examples.items():
            for example in examples:
                texts.append(example)
                labels.append(intent)
        
        return texts, labels
    
    def train_model(self, model_type='naive_bayes'):
        """Train the classification model"""
        texts, labels = self.prepare_training_data()
        
        if len(set(labels)) < 2:
            print("⚠ Need at least 2 different intents for training")
            return None
        
        self.classes = list(set(labels))
        
        if model_type == 'naive_bayes':
            self.model = Pipeline([
                ('tfidf', self.vectorizer),
                ('clf', MultinomialNB())
            ])
        elif model_type == 'logistic':
            self.model = Pipeline([
                ('tfidf', self.vectorizer),
                ('clf', LogisticRegression(max_iter=1000))
            ])
        elif model_type == 'svm':
            self.model = Pipeline([
                ('tfidf', self.vectorizer),
                ('clf', SVC(kernel='linear', probability=True))
            ])
        
        self.model.fit(texts, labels)
        print(f"✓ Model trained with {len(texts)} examples across {len(self.classes)} intents")
        return self.model
    
    def predict(self, text):
        """Predict intent for given text"""
        if self.model is None:
            return None, 0.0
        
        try:
            probabilities = self.model.predict_proba([text.lower()])[0]
            max_prob_idx = np.argmax(probabilities)
            intent = self.model.classes_[max_prob_idx]
            confidence = probabilities[max_prob_idx]
            return intent, confidence
        except:
            return None, 0.0
    
    def save(self, filepath):
        """Save trained model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'classes': self.classes,
                'vectorizer': self.vectorizer,
                'intent_examples': dict(self.intent_examples)
            }, f)
    
    def load(self, filepath):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.classes = data['classes']
            self.vectorizer = data['vectorizer']
            self.intent_examples = defaultdict(list, data.get('intent_examples', {}))

class AdaptiveLearningManager:
    def __init__(self):
        self.user_feedback = defaultdict(list)
        self.conversation_patterns = defaultdict(list)
        self.successful_responses = defaultdict(int)
        self.failed_responses = defaultdict(int)
        self.user_preferences = defaultdict(dict)
        
    def record_feedback(self, session_id, user_input, bot_response, feedback_score):
        """Record user feedback for learning"""
        self.user_feedback[session_id].append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'bot_response': bot_response,
            'feedback_score': feedback_score,
            'response_type': bot_response.get('type', 'unknown')
        })
        
        if feedback_score > 0.7:  # Positive feedback
            self.successful_responses[bot_response.get('type', 'unknown')] += 1
        else:  # Negative feedback
            self.failed_responses[bot_response.get('type', 'unknown')] += 1
    
    def analyze_conversation_patterns(self, session_id, conversation_history):
        """Analyze patterns in user conversations"""
        if len(conversation_history) > 2:
            last_interactions = conversation_history[-3:]
            pattern = tuple([interaction.get('intent', 'unknown') for interaction in last_interactions])
            self.conversation_patterns[session_id].append(pattern)
    
    def get_user_preference(self, session_id, preference_type):
        """Get user preferences based on history"""
        return self.user_preferences[session_id].get(preference_type)
    
    def update_user_preference(self, session_id, preference_type, value):
        """Update user preferences"""
        self.user_preferences[session_id][preference_type] = value
    
    def get_optimized_response_type(self, session_id):
        """Get the best response type for a user based on history"""
        user_feedback = self.user_feedback.get(session_id, [])
        if not user_feedback:
            return 'options'  # Default
        
        # Calculate success rates for different response types
        response_success = defaultdict(list)
        for feedback in user_feedback:
            response_type = feedback['response_type']
            response_success[response_type].append(feedback['feedback_score'])
        
        # Return response type with highest average success
        if response_success:
            best_type = max(response_success.keys(), 
                          key=lambda x: np.mean(response_success[x]) if response_success[x] else 0)
            return best_type
        return 'options'

class DialogFlowManager:
    """Original DialogFlowManager for backward compatibility"""
    def __init__(self):
        # Enhanced account types with variations
        self.account_types = {
            'ind': 'individual', 'individual': 'individual', 'personal': 'individual', 
            'single': 'individual', 'private': 'individual', 'personal account': 'individual',
            'join': 'joint', 'joint': 'joint', 'partnership': 'joint', 'couple': 'joint', 
            'family': 'joint', 'shared': 'joint', 'multiple': 'joint',
            'cmp': 'company', 'company': 'company', 'business': 'company', 'corporate': 'company', 
            'enterprise': 'company', 'organization': 'company', 'firm': 'company',
            'student': 'student', 'teen': 'student', 'youth': 'student', 'university': 'student',
            'college': 'student', 'school': 'student',
            'savings': 'savings', 'current': 'current', 'fixed': 'fixed_deposit', 
            'investment': 'investment', 'diaspora': 'diaspora', 'foreign': 'diaspora', 
            'international': 'diaspora', 'other': 'other', 'misc': 'other'
        }
        
        # Negation words and phrases
        self.negation_words = {
            'no', 'not', "don't", "do not", "dont", "cannot", "can't", "cant", "won't", 
            "will not", "wouldn't", "shouldn't", "couldn't", "isn't", "aren't", "wasn't", 
            "weren't", "haven't", "hasn't", "hadn't", "doesn't", "didn't", "never", 
            "nothing", "none", "nobody", "nowhere", "neither", "nor", "without", 
            "lack", "missing", "stop", "cancel", "end", "quit", "exit", "decline", 
            "refuse", "reject", "negative", "nah", "nope", "no thanks", "no thank you"
        }
        
        # Initialize banking data
        self._initialize_banking_data()
        
        # Conversation memory
        self.conversation_memory = {}
    
    def _initialize_banking_data(self):
        """Initialize banking information data"""
        self.account_info = {
            'individual': {
                'name': 'Individual Savings Account',
                'documents': ['Original National ID/Passport', '2 Recent passport-size photos', 
                             'Tax Identification Number (TIN) Certificate', 'Proof of income'],
                'min_amount': 100, 'monthly_fee': 0, 'interest_rate': '5.0% per annum',
                'features': ['Free mobile banking', 'Visa/Mastercard debit card', '24/7 customer support'],
                'process_time': '1-2 business days', 'requirements': 'Must be 18+ years old with valid ID'
            },
            'joint': {
                'name': 'Joint Account', 'min_amount': 200, 'monthly_fee': 50,
                'documents': ['Original IDs for all applicants', 'Joint application form'],
                'process_time': '2-3 business days'
            },
            'company': {
                'name': 'Business Account', 'min_amount': 1000, 'monthly_fee': 200,
                'documents': ['Certificate of Incorporation', 'Business registration documents'],
                'process_time': '3-5 business days'
            },
            'student': {
                'name': 'Student Account', 'min_amount': 50, 'monthly_fee': 0,
                'documents': ['Student ID card', 'Letter of admission'],
                'process_time': '1 business day'
            }
        }
        
        self.services = {
            'loan': {
                'personal': {
                    'name': 'Personal Loan', 
                    'amount_range': '5,000 - 500,000',
                    'interest_rate': '12-18% per annum',
                    'requirements': ['3 months bank statements', 'TIN certificate', 'ID copy']
                },
                'business': {
                    'name': 'Business Loan', 
                    'amount_range': '50,000 - 10,000,000',
                    'interest_rate': '10-15% per annum',
                    'requirements': ['Business registration', '6 months bank statements']
                }
            }
        }
        
        self.bank_info = {
            'branches': "We have over 1,800 branches across the country including major cities and regional capitals.",
            'hours': "**Banking Hours**:\n• Weekdays: 8:00 AM - 4:00 PM\n• Saturdays: 8:00 AM - 12:00 PM\n• Sundays: Closed",
            'contact': "**Contact Information**:\n• Phone: +251 115 57 57 57\n• Email: info@bank.com\n• Website: www.bank.com"
        }
    
    def contains_negation(self, user_input):
        """Check if user input contains negation"""
        user_input = user_input.lower()
        words = user_input.split()
        return any(word in self.negation_words for word in words)
    
    def extract_negation_context(self, user_input):
        """Extract what user is saying no to"""
        user_input = user_input.lower()
        patterns = {
            'account': r'(?:account|open|create)',
            'loan': r'(?:loan|borrow|credit)',
            'service': r'(?:service|help|assistance)',
            'information': r'(?:information|info|details)'
        }
        
        for context, pattern in patterns.items():
            if re.search(pattern, user_input):
                return context
        return 'general'
    
    def extract_account_type(self, user_input):
        """Extract account type from user input"""
        user_input = user_input.lower()
        
        for key, value in self.account_types.items():
            if key in user_input:
                return value
        
        return None
    
    def process_message(self, user_input, session_id=None, current_state=None):
        """Process user message and return response"""
        # Validate session_id
        if session_id is not None:
            if isinstance(session_id, dict):
                session_id = session_id.get('id') or session_id.get('session_id') or str(session_id)
            elif not isinstance(session_id, str):
                session_id = str(session_id)
        
        user_input_lower = user_input.lower().strip()
        
        # Initialize session context
        if session_id and session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = {
                'current_topic': None,
                'user_name': None,
                'step_count': 0
            }
        
        context = self.conversation_memory.get(session_id, {})
        context['step_count'] = context.get('step_count', 0) + 1
        
        # Check for negation first
        if self.contains_negation(user_input_lower):
            negation_context = self.extract_negation_context(user_input_lower)
            return self._handle_negation_response(negation_context)
        
        # Check for greetings
        if any(word in user_input_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return self._get_greeting_response(context.get('user_name'))
        
        # Account type detection
        account_type = self.extract_account_type(user_input_lower)
        if account_type:
            context['current_topic'] = f'account_{account_type}'
            self.conversation_memory[session_id] = context
            return self._get_account_type_response(account_type, context.get('user_name'))
        
        # Service inquiries
        if any(word in user_input_lower for word in ['loan', 'borrow', 'credit']):
            return self._get_loan_information()
        
        if any(word in user_input_lower for word in ['branch', 'location', 'where']):
            return {'text': self.bank_info['branches'], 'type': 'message'}
        
        if any(word in user_input_lower for word in ['contact', 'phone', 'email']):
            return {'text': self.bank_info['contact'], 'type': 'message'}
        
        # Fallback response
        return self._get_fallback_response()
    
    def _handle_negation_response(self, negation_context):
        """Handle negative responses"""
        responses = {
            'account': {
                'text': "✅ No problem! What other service can I help you with?",
                'options': [
                    {'value': 'loan_info', 'label': '💰 Loans'},
                    {'value': 'services', 'label': '🛎️ Services'},
                    {'value': 'support', 'label': '📞 Support'}
                ],
                'type': 'options'
            },
            'loan': {
                'text': "✅ Understood! Would you like information about other services?",
                'options': [
                    {'value': 'savings', 'label': '🏦 Savings Accounts'},
                    {'value': 'investments', 'label': '📈 Investments'},
                    {'value': 'support', 'label': '📞 Customer Support'}
                ],
                'type': 'options'
            },
            'general': {
                'text': "✅ I understand. How else can I assist you today?",
                'options': [
                    {'value': 'accounts', 'label': '🏦 Accounts'},
                    {'value': 'loans', 'label': '💰 Loans'},
                    {'value': 'services', 'label': '🛎️ Services'}
                ],
                'type': 'options'
            }
        }
        
        return responses.get(negation_context, responses['general'])
    
    def _get_greeting_response(self, user_name=None):
        """Get greeting response"""
        greeting = "Hello! 👋 "
        if user_name:
            greeting += f"{user_name}, "
        greeting += "Welcome to our Banking Services! \n\nI'm here to help you with all your banking needs."
        
        return {
            'text': greeting,
            'options': [
                {'value': 'open_account', 'label': '🏦 Open Account'},
                {'value': 'loan_info', 'label': '💰 Loans'},
                {'value': 'services', 'label': '🛎️ Services'},
                {'value': 'support', 'label': '📞 Support'}
            ],
            'type': 'options'
        }
    
    def _get_account_type_response(self, account_type, user_name=None):
        """Get account type response"""
        if account_type not in self.account_info:
            return self._get_fallback_response()
        
        account_data = self.account_info[account_type]
        personalized = f", {user_name}" if user_name else ""
        
        response_text = f"""
✅ Excellent choice{personalized}! You selected the **{account_data['name']}**.

💰 **Minimum Amount**: {account_data['min_amount']:,}
⏰ **Processing Time**: {account_data['process_time']}

What would you like to know about this account?
        """.strip()
        
        return {
            'text': response_text,
            'options': [
                {'value': 'documents', 'label': '📄 Documents'},
                {'value': 'features', 'label': '⭐ Features'},
                {'value': 'fees', 'label': '💵 Fees'},
                {'value': 'process', 'label': '⏰ Process'}
            ],
            'type': 'options',
            'state': {'account_type': account_type}
        }
    
    def _get_loan_information(self):
        """Get loan information"""
        return {
            'text': "💰 **Loan Services Available**:\n\n• Personal Loans (5,000 - 500,000)\n• Business Loans (50,000 - 10,000,000)\n\nWhich type are you interested in?",
            'options': [
                {'value': 'personal_loan', 'label': '👤 Personal Loan'},
                {'value': 'business_loan', 'label': '🏢 Business Loan'}
            ],
            'type': 'options'
        }
    
    def _get_fallback_response(self):
        """Get fallback response"""
        return {
            'text': "I'm here to help with your banking needs! What would you like to know?",
            'options': [
                {'value': 'accounts', 'label': '🏦 Accounts'},
                {'value': 'loans', 'label': '💰 Loans'},
                {'value': 'services', 'label': '🛎️ Services'},
                {'value': 'contact', 'label': '📞 Contact'}
            ],
            'type': 'options'
        }

class EnhancedDialogFlowManager(DialogFlowManager):
    """Enhanced version with ML capabilities"""
    
    def __init__(self, model_file=None):
        # Initialize parent class
        super().__init__()
        
        # Initialize ML components
        self.trainer = BankingChatbotTrainer()
        self.learning_manager = AdaptiveLearningManager()
        
        # Load or create initial model
        if model_file:
            try:
                self.trainer.load(model_file)
                print("✓ Pre-trained model loaded successfully")
            except Exception as e:
                print(f"⚠ Could not load model: {e}, initializing with default training")
                self._train_initial_model()
        else:
            self._train_initial_model()
        
        # Enhanced conversation memory
        self.response_history = []
    
    def _train_initial_model(self):
        """Train the initial model with basic banking intents"""
        intent_examples = {
            'greeting': [
                "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                "hello there", "hey there", "hi there", "greetings"
            ],
            'account_inquiry': [
                "I want to open an account", "how to open account", "account opening",
                "create new account", "start banking account", "open savings account",
                "I need a bank account", "want to open individual account"
            ],
            'loan_inquiry': [
                "I need a loan", "apply for loan", "loan information", "personal loan",
                "business loan", "education loan", "how to get loan", "loan requirements"
            ],
            'service_inquiry': [
                "mobile banking", "internet banking", "card services", "debit card",
                "credit card", "online banking", "banking app", "digital banking"
            ],
            'branch_info': [
                "branch locations", "where is your branch", "nearest bank", "bank address",
                "find branch", "locations near me", "where are you located"
            ],
            'contact_info': [
                "contact number", "phone number", "email address", "how to contact",
                "customer service", "support number", "call bank", "contact details"
            ],
            'negation': [
                "no", "not interested", "no thanks", "I don't want", "maybe later",
                "not now", "cancel", "stop", "no thank you", "not today"
            ]
        }
        
        for intent, examples in intent_examples.items():
            for example in examples:
                self.trainer.add_example(intent, example)
        
        self.trainer.train_model()
        print("✓ Initial model trained with banking intents")
    
    def continue_training(self, new_examples):
        """Continue training with new examples"""
        for intent, examples in new_examples.items():
            for example in examples:
                self.trainer.add_example(intent, example)
        
        self.trainer.train_model()
        print(f"✓ Model updated with {len(new_examples)} new intents")
    
    def add_training_example(self, intent, example):
        """Add a single training example"""
        self.trainer.add_example(intent, example)
        self.trainer.train_model()
        print(f"✓ Learned: '{example}' → {intent}")
    
    def process_message(self, user_input, session_id=None, current_state=None, feedback_score=None):
        """Enhanced message processing with machine learning"""
        
        # Handle feedback from previous interaction
        if feedback_score is not None and session_id and self.response_history:
            last_response = self.response_history[-1] if self.response_history else None
            if last_response:
                self.learning_manager.record_feedback(session_id, user_input, last_response, feedback_score)
        
        # Use ML to predict intent
        ml_intent, confidence = self.trainer.predict(user_input)
        
        # Get user's preferred response type
        preferred_type = self.learning_manager.get_optimized_response_type(session_id) if session_id else 'options'
        
        # Process based on ML intent and confidence
        if confidence > 0.6:  # Good confidence threshold
            response = self._handle_ml_intent(ml_intent, user_input, session_id, preferred_type)
        else:  # Low confidence - use parent's rule-based processing
            response = super().process_message(user_input, session_id, current_state)
        
        # Store response in history for feedback learning
        self.response_history.append(response)
        response['ml_confidence'] = confidence
        response['detected_intent'] = ml_intent
        
        return response
    
    def _handle_ml_intent(self, intent, user_input, session_id, preferred_type):
        """Handle message based on ML-predicted intent"""
        
        intent_handlers = {
            'greeting': lambda: self._get_greeting_response(),
            'account_inquiry': lambda: self._handle_ml_account_inquiry(user_input, session_id),
            'loan_inquiry': lambda: self._get_loan_information(),
            'service_inquiry': lambda: self._handle_ml_service_inquiry(),
            'branch_info': lambda: {'text': self.bank_info['branches'], 'type': 'message'},
            'contact_info': lambda: {'text': self.bank_info['contact'], 'type': 'message'},
            'negation': lambda: self._handle_ml_negation(user_input)
        }
        
        handler = intent_handlers.get(intent, lambda: super()._get_fallback_response())
        response = handler()
        
        # Apply preferred response type if applicable
        if 'type' in response and preferred_type != 'options':
            response['type'] = preferred_type
            
        return response
    
    def _handle_ml_account_inquiry(self, user_input, session_id):
        """Handle account inquiries with ML"""
        account_type = self.extract_account_type(user_input)
        if account_type:
            # Update session context
            if session_id and session_id in self.conversation_memory:
                self.conversation_memory[session_id]['current_topic'] = f'account_{account_type}'
            return self._get_account_type_response(account_type)
        else:
            return {
                'text': "I'd be happy to help you open an account! What type of account are you interested in?",
                'options': [
                    {'value': 'individual', 'label': '👤 Individual Account'},
                    {'value': 'joint', 'label': '👥 Joint Account'},
                    {'value': 'business', 'label': '🏢 Business Account'},
                    {'value': 'student', 'label': '🎓 Student Account'}
                ],
                'type': 'options'
            }
    
    def _handle_ml_service_inquiry(self):
        """Handle service inquiries with ML"""
        return {
            'text': "Here are our digital banking services:",
            'options': [
                {'value': 'mobile_banking', 'label': '📱 Mobile Banking'},
                {'value': 'internet_banking', 'label': '💻 Internet Banking'},
                {'value': 'card_services', 'label': '💳 Card Services'},
                {'value': 'atm_services', 'label': '🏧 ATM Services'}
            ],
            'type': 'options'
        }
    
    def _handle_ml_negation(self, user_input):
        """Handle negation with ML"""
        negation_context = self.extract_negation_context(user_input)
        return self._handle_negation_response(negation_context)
    
    def get_learning_stats(self):
        """Get learning statistics"""
        return {
            'total_feedback_records': sum(len(fb) for fb in self.learning_manager.user_feedback.values()),
            'successful_responses': dict(self.learning_manager.successful_responses),
            'failed_responses': dict(self.learning_manager.failed_responses),
            'conversation_patterns': len(self.learning_manager.conversation_patterns),
            'trained_intents': len(self.trainer.intent_examples)
        }
    
    def save_knowledge(self, filepath):
        """Save learned knowledge to file"""
        knowledge = {
            'conversation_memory': self.conversation_memory,
            'user_feedback': dict(self.learning_manager.user_feedback),
            'successful_responses': dict(self.learning_manager.successful_responses),
            'failed_responses': dict(self.learning_manager.failed_responses),
            'user_preferences': dict(self.learning_manager.user_preferences)
        }
        
        with open(filepath, 'w') as f:
            json.dump(knowledge, f, indent=2)
        
        # Also save the classifier
        self.trainer.save(filepath + '.model')
    
    
    def load_knowledge(self, filepath):
        """Load learned knowledge from file"""
        try:
            with open(filepath, 'r') as f:
                knowledge = json.load(f)
                
            self.conversation_memory = knowledge.get('conversation_memory', {})
            self.learning_manager.user_feedback.update(knowledge.get('user_feedback', {}))
            self.learning_manager.successful_responses.update(knowledge.get('successful_responses', {}))
            self.learning_manager.failed_responses.update(knowledge.get('failed_responses', {}))
            self.learning_manager.user_preferences.update(knowledge.get('user_preferences', {}))
            
            # Load classifier
            self.trainer.load(filepath + '.model')
            
            print("✓ Learned knowledge loaded successfully")
        except FileNotFoundError:
            print("⚠ No previous knowledge found, starting fresh")

# Export both classes for import
__all__ = ['DialogFlowManager', 'EnhancedDialogFlowManager']