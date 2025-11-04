# dialog_flow.py
import re
import json
import pickle
import numpy as np
from difflib import SequenceMatcher
from datetime import datetime
from collections import defaultdict, Counter
from enum import Enum
from typing import Dict, List, Optional, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# 🧠 Enhanced Knowledge Base: Account Opening & Onboarding
# ==============================================
onboarding_knowledge = {
    "open account": "To open an account, you can start onboarding yourself using your National ID (NID). Once you enter or scan your NID, your information loads automatically. You'll then upload your signature and provide your mother's name for verification.",

    "account types": "There are three main account types: Individual, Joint, and Cooperative (Company). Each can be opened under either the Saving or Alhuda category.",

    "alhuda vs saving": "Alhuda follows Islamic banking (interest-free, profit-sharing Mudarabah), while Saving is conventional and earns interest.",

    "alhuda": "Alhuda is the Islamic banking category that operates under profit-sharing (Mudarabah) principles — completely interest-free.",

    "saving": "Saving is the conventional banking category, where customers earn interest on their deposits.",

    "individual account": "An Individual Account is opened by one person. You can choose either Alhuda (interest-free) or Saving (interest-based). You'll need your NID, mother's name, and signature.",

    "joint account": "A Joint Account is opened by two or more people. Each person must provide their NID, mother's name, and signature.",

    "cooperative account": "A Cooperative or Company Account is for registered businesses or cooperatives. It can be opened as Saving or Current. You'll need company registration, NIDs of signatories, and authorized signatures.",

    "sinqe saving": "Sinqe Saving is available under both Alhuda and Saving systems. Under Saving, it earns interest; under Alhuda, it's profit-sharing and interest-free.",

    "youth account": "The Youth Account is designed for young customers, encouraging saving habits. It's available under both Alhuda and Saving categories.",

    "requirements": "To open any account, you'll need your National ID (NID), mother's name, and signature (upload or sign). For cooperative accounts, you'll also need company registration and authorized signatories.",

    "difference between accounts": "Individual accounts are for one person, joint accounts are shared between two or more people, and cooperative accounts are for registered companies or organizations.",

    "documents needed": "For individual accounts: National ID, mother's name, signature. For joint accounts: Same for all applicants. For cooperative: Company registration, NIDs of signatories.",

    "how to open": "Start with your National ID scan/entry → automatic info loading → upload signature → provide mother's name → choose account type → complete verification.",

    "mother name": "Your mother's name is required for identity verification and security purposes during account opening.",

    "national id": "Your National ID (NID) is used to automatically load your personal information during the account opening process.",

    "signature": "You can upload your signature digitally or provide it during branch visits for account verification."
}

class MenuType(Enum):
    MAIN = "main"
    ACCOUNTS = "accounts"
    SERVICES = "services"
    SUPPORT = "support"
    ONBOARDING = "onboarding"
    LOANS = "loans"

class ResponseType(Enum):
    OPTIONS = "options"
    MESSAGE = "message"
    MENU = "menu"
    QUICK_REPLIES = "quick_replies"

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

class KnowledgeBaseManager:
    """Manages the knowledge base and menu system"""
    
    def __init__(self):
        self.knowledge_base = onboarding_knowledge
        self.current_menu = MenuType.MAIN
        self.menus = self._initialize_menus()
    
    def _initialize_menus(self) -> Dict[MenuType, Dict]:
        """Initialize all menu structures"""
        return {
            MenuType.MAIN: {
                "title": "🏦 Welcome to Banking Services",
                "subtitle": "How can I help you today?",
                "options": [
                    {"value": "open_account", "label": "📝 Open New Account", "menu": MenuType.ONBOARDING},
                    {"value": "account_types", "label": "🏦 Account Types", "menu": MenuType.ACCOUNTS},
                    {"value": "loan_info", "label": "💰 Loans", "menu": MenuType.LOANS},
                    {"value": "banking_services", "label": "🛎️ Banking Services", "menu": MenuType.SERVICES},
                    {"value": "support", "label": "📞 Customer Support", "menu": MenuType.SUPPORT}
                ]
            },
            MenuType.ACCOUNTS: {
                "title": "🏦 Account Types",
                "subtitle": "Choose an account type to learn more:",
                "options": [
                    {"value": "individual", "label": "👤 Individual Account"},
                    {"value": "joint", "label": "👥 Joint Account"},
                    {"value": "cooperative", "label": "🏢 Cooperative Account"},
                    {"value": "alhuda", "label": "🕌 Alhuda Account"},
                    {"value": "saving", "label": "💰 Saving Account"},
                    {"value": "youth", "label": "🎓 Youth Account"},
                    {"value": "back", "label": "⬅️ Back to Main Menu", "menu": MenuType.MAIN}
                ]
            },
            MenuType.ONBOARDING: {
                "title": "📝 Account Opening & Onboarding",
                "subtitle": "What would you like to know?",
                "options": [
                    {"value": "how_to_open", "label": "❓ How to Open Account"},
                    {"value": "requirements", "label": "📄 Requirements"},
                    {"value": "documents", "label": "📋 Documents Needed"},
                    {"value": "process", "label": "⚙️ Process Steps"},
                    {"value": "difference", "label": "🔄 Account Differences"},
                    {"value": "back", "label": "⬅️ Back to Main Menu", "menu": MenuType.MAIN}
                ]
            },
            MenuType.LOANS: {
                "title": "💰 Loan Services",
                "subtitle": "Which loan type are you interested in?",
                "options": [
                    {"value": "personal_loan", "label": "👤 Personal Loan"},
                    {"value": "business_loan", "label": "🏢 Business Loan"},
                    {"value": "education_loan", "label": "🎓 Education Loan"},
                    {"value": "back", "label": "⬅️ Back to Main Menu", "menu": MenuType.MAIN}
                ]
            },
            MenuType.SERVICES: {
                "title": "🛎️ Banking Services",
                "subtitle": "What service do you need?",
                "options": [
                    {"value": "mobile_banking", "label": "📱 Mobile Banking"},
                    {"value": "internet_banking", "label": "💻 Internet Banking"},
                    {"value": "card_services", "label": "💳 Card Services"},
                    {"value": "transfers", "label": "🔄 Money Transfers"},
                    {"value": "back", "label": "⬅️ Back to Main Menu", "menu": MenuType.MAIN}
                ]
            },
            MenuType.SUPPORT: {
                "title": "📞 Customer Support",
                "subtitle": "How can we assist you?",
                "options": [
                    {"value": "contact", "label": "📞 Contact Information"},
                    {"value": "branches", "label": "🏢 Branch Locations"},
                    {"value": "hours", "label": "⏰ Banking Hours"},
                    {"value": "emergency", "label": "🚨 Emergency Services"},
                    {"value": "back", "label": "⬅️ Back to Main Menu", "menu": MenuType.MAIN}
                ]
            }
        }
    
    def find_best_match(self, user_input: str) -> Optional[str]:
        """
        Matches user input with the most relevant key in the knowledge base.
        """
        user_input = user_input.lower().strip()
        best_match = None
        highest_ratio = 0

        # Check for exact matches first
        for key in self.knowledge_base.keys():
            if key in user_input:
                return self.knowledge_base[key]

        # Then check for similar matches
        for key in self.knowledge_base.keys():
            ratio = SequenceMatcher(None, key, user_input).ratio()
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = key

        # Match threshold for confidence
        if highest_ratio > 0.5:
            return self.knowledge_base[best_match]
        
        return None
    
    def get_menu_response(self, menu_type: MenuType) -> Dict:
        """Get structured menu response"""
        menu = self.menus[menu_type]
        self.current_menu = menu_type
        return {
            "type": ResponseType.MENU.value,
            "text": f"{menu['title']}\n\n{menu['subtitle']}",
            "menu_title": menu['title'],
            "menu_subtitle": menu['subtitle'],
            "options": menu['options'],
            "current_menu": menu_type.value
        }
    
    def handle_menu_selection(self, selection: str) -> Dict:
        """Handle menu option selections"""
        menu = self.menus[self.current_menu]
        selected_option = None
        
        for option in menu["options"]:
            if option["value"] == selection:
                selected_option = option
                break
        
        if not selected_option:
            return self._get_unknown_selection_response()
        
        # Handle navigation to other menus
        if "menu" in selected_option:
            return self.get_menu_response(selected_option["menu"])
        
        # Handle specific actions in current menu
        return self._handle_menu_action(selection)
    
    def _handle_menu_action(self, selection: str) -> Dict:
        """Handle specific menu actions"""
        action_handlers = {
            MenuType.ACCOUNTS: {
                "individual": lambda: self._get_knowledge_response("individual account"),
                "joint": lambda: self._get_knowledge_response("joint account"),
                "cooperative": lambda: self._get_knowledge_response("cooperative account"),
                "alhuda": lambda: self._get_knowledge_response("alhuda"),
                "saving": lambda: self._get_knowledge_response("saving"),
                "youth": lambda: self._get_knowledge_response("youth account")
            },
            MenuType.ONBOARDING: {
                "how_to_open": lambda: self._get_knowledge_response("how to open"),
                "requirements": lambda: self._get_knowledge_response("requirements"),
                "documents": lambda: self._get_knowledge_response("documents needed"),
                "process": lambda: self._get_knowledge_response("how to open"),
                "difference": lambda: self._get_knowledge_response("difference between accounts")
            },
            MenuType.SUPPORT: {
                "contact": lambda: self._get_support_response("contact"),
                "branches": lambda: self._get_support_response("branches"),
                "hours": lambda: self._get_support_response("hours"),
                "emergency": lambda: self._get_support_response("emergency")
            }
        }
        
        handler = action_handlers.get(self.current_menu, {}).get(selection)
        if handler:
            response = handler()
            # Add navigation options
            response["options"] = [
                {"value": "back_to_menu", "label": "⬅️ Back to Menu"},
                {"value": "main_menu", "label": "🏠 Main Menu"}
            ]
            return response
        
        return self._get_fallback_response()
    
    def _get_knowledge_response(self, key: str) -> Dict:
        """Get response from knowledge base"""
        response_text = self.knowledge_base.get(key, "I'm sorry, I don't have information about that yet.")
        return {
            "type": ResponseType.MESSAGE.value,
            "text": response_text,
            "source": "knowledge_base",
            "knowledge_key": key
        }
    
    def _get_support_response(self, support_type: str) -> Dict:
        """Get support-related responses"""
        support_responses = {
            "contact": "📞 **Contact Information:**\n• Phone: +251 115 57 57 57\n• Email: info@bank.com\n• WhatsApp: +251 911 123 456",
            "branches": "🏢 **Branch Locations:**\nWe have over 1,800 branches nationwide. Visit our website www.bank.com/branches to find the nearest one.",
            "hours": "⏰ **Banking Hours:**\n• Weekdays: 8:00 AM - 4:00 PM\n• Saturdays: 8:00 AM - 12:00 PM\n• Sundays: Closed",
            "emergency": "🚨 **Emergency Services:**\n• Lost Card: +251 115 57 57 58\n• Fraud Reporting: +251 115 57 57 59\n• 24/7 Helpline: +251 900 123 456"
        }
        
        return {
            "type": ResponseType.MESSAGE.value,
            "text": support_responses.get(support_type, "Please contact our customer service for assistance."),
            "source": "support"
        }
    
    def _get_unknown_selection_response(self) -> Dict:
        """Handle unknown menu selections"""
        return {
            "type": ResponseType.MESSAGE.value,
            "text": "I didn't understand that selection. Please choose from the available options.",
            "options": [
                {"value": "back_to_menu", "label": "⬅️ Back to Current Menu"},
                {"value": "main_menu", "label": "🏠 Main Menu"}
            ]
        }
    
    def _get_fallback_response(self) -> Dict:
        """Get fallback response"""
        return {
            "type": ResponseType.MESSAGE.value,
            "text": "I'm here to help with your banking needs!",
            "options": self.menus[MenuType.MAIN]["options"]
        }

class DialogFlowManager:
    """Enhanced DialogFlowManager with Knowledge Base and Menu System"""
    
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
            'international': 'diaspora', 'other': 'other', 'misc': 'other',
            'alhuda': 'alhuda', 'islamic': 'alhuda', 'sharia': 'alhuda'
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
        
        # Initialize Knowledge Base and Menu System
        self.knowledge_manager = KnowledgeBaseManager()
        
        # Initialize ML components
        self.trainer = BankingChatbotTrainer()
        self.learning_manager = AdaptiveLearningManager()
        self.response_history = []
        
        # Train initial model
        self._train_initial_model()
    
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
            },
            'alhuda': {
                'name': 'Alhuda Islamic Account',
                'documents': ['National ID', 'Signature', "Mother's name"],
                'min_amount': 100, 'monthly_fee': 0, 'interest_rate': 'Profit-sharing',
                'features': ['Sharia-compliant', 'Interest-free', 'Ethical banking'],
                'process_time': '1-2 business days'
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
                },
                'education': {
                    'name': 'Education Loan',
                    'amount_range': '10,000 - 200,000',
                    'interest_rate': '8-12% per annum',
                    'requirements': ['Admission letter', 'Fee structure']
                }
            }
        }
        
        self.bank_info = {
            'branches': "We have over 1,800 branches across the country including major cities and regional capitals.",
            'hours': "**Banking Hours**:\n• Weekdays: 8:00 AM - 4:00 PM\n• Saturdays: 8:00 AM - 12:00 PM\n• Sundays: Closed",
            'contact': "**Contact Information**:\n• Phone: +251 115 57 57 57\n• Email: info@bank.com\n• Website: www.bank.com",
            'emergency': "**24/7 Emergency**:\n• Lost cards: +251 115 57 57 58\n• Fraud: +251 115 57 57 59"
        }
    
    def _train_initial_model(self):
        """Train the initial model with enhanced banking intents including knowledge base topics"""
        intent_examples = {
            'greeting': [
                "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                "hello there", "hey there", "hi there", "greetings"
            ],
            'account_inquiry': [
                "I want to open an account", "how to open account", "account opening",
                "create new account", "start banking account", "open savings account",
                "I need a bank account", "want to open individual account", "account types",
                "what accounts do you have", "types of accounts"
            ],
            'loan_inquiry': [
                "I need a loan", "apply for loan", "loan information", "personal loan",
                "business loan", "education loan", "how to get loan", "loan requirements",
                "borrow money", "loan options"
            ],
            'service_inquiry': [
                "mobile banking", "internet banking", "card services", "debit card",
                "credit card", "online banking", "banking app", "digital banking",
                "ATM services", "online payments"
            ],
            'branch_info': [
                "branch locations", "where is your branch", "nearest bank", "bank address",
                "find branch", "locations near me", "where are you located", "branch hours"
            ],
            'contact_info': [
                "contact number", "phone number", "email address", "how to contact",
                "customer service", "support number", "call bank", "contact details",
                "help desk", "customer care"
            ],
            'negation': [
                "no", "not interested", "no thanks", "I don't want", "maybe later",
                "not now", "cancel", "stop", "no thank you", "not today", "I'm good"
            ],
            'onboarding_help': [
                "how to open account", "account requirements", "documents needed",
                "what do I need to open account", "onboarding process", "mother's name",
                "national id requirement", "signature upload", "account opening steps"
            ],
            'alhuda_inquiry': [
                "alhuda account", "islamic banking", "sharia compliant", "interest free",
                "islamic account", "alhuda vs saving", "profit sharing", "mudarabah"
            ],
            'menu_request': [
                "show menu", "what can you do", "options", "help", "main menu",
                "services", "what services", "banking services"
            ]
        }
        
        for intent, examples in intent_examples.items():
            for example in examples:
                self.trainer.add_example(intent, example)
        
        self.trainer.train_model()
        print("✓ Enhanced model trained with knowledge base intents")
    
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
    
    def process_message(self, user_input, session_id=None, current_state=None, feedback_score=None, is_menu_selection=False):
        """Enhanced message processing with ML, Knowledge Base, and Menu System"""
        
        # Handle feedback from previous interaction
        if feedback_score is not None and session_id and self.response_history:
            last_response = self.response_history[-1] if self.response_history else None
            if last_response:
                self.learning_manager.record_feedback(session_id, user_input, last_response, feedback_score)
        
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
                'step_count': 0,
                'current_menu': MenuType.MAIN.value
            }
        
        context = self.conversation_memory.get(session_id, {})
        context['step_count'] = context.get('step_count', 0) + 1
        
        # Handle menu selections first
        if is_menu_selection:
            response = self.knowledge_manager.handle_menu_selection(user_input)
            self.response_history.append(response)
            return response
        
        # Check for menu requests
        if any(word in user_input_lower for word in ['menu', 'options', 'help', 'what can you do', 'show menu']):
            response = self.knowledge_manager.get_menu_response(MenuType.MAIN)
            self.response_history.append(response)
            return response
        
        # Use ML to predict intent
        ml_intent, confidence = self.trainer.predict(user_input)
        
        # Get user's preferred response type
        preferred_type = self.learning_manager.get_optimized_response_type(session_id) if session_id else 'options'
        
        # Process based on ML intent and confidence
        response = None
        if confidence > 0.6:  # Good confidence threshold
            response = self._handle_ml_intent(ml_intent, user_input, session_id, preferred_type)
        else:  # Low confidence - try knowledge base
            knowledge_response = self.knowledge_manager.find_best_match(user_input)
            if knowledge_response:
                response = {
                    'text': knowledge_response,
                    'type': ResponseType.MESSAGE.value,
                    'source': 'knowledge_base',
                    'confidence': confidence,
                    'options': [
                        {'value': 'more_info', 'label': '🤔 More Information'},
                        {'value': 'onboarding_menu', 'label': '📝 Onboarding Help'},
                        {'value': 'main_menu', 'label': '🏠 Main Menu'}
                    ]
                }
            else:
                # Fallback to rule-based processing
                response = self._handle_rule_based_intent(user_input, session_id, context)
        
        # Apply preferred response type if applicable
        if response and 'type' in response and preferred_type != 'options' and response['type'] == 'options':
            response['type'] = preferred_type
        
        # Store response in history for feedback learning
        if response:
            response['ml_confidence'] = confidence
            response['detected_intent'] = ml_intent
            self.response_history.append(response)
        
        return response if response else self._get_fallback_response()
    
    def _handle_ml_intent(self, intent, user_input, session_id, preferred_type):
        """Handle message based on ML-predicted intent"""
        
        intent_handlers = {
            'greeting': lambda: self._get_greeting_response(session_id),
            'account_inquiry': lambda: self._handle_ml_account_inquiry(user_input, session_id),
            'loan_inquiry': lambda: self._get_loan_information(),
            'service_inquiry': lambda: self._handle_ml_service_inquiry(),
            'branch_info': lambda: {'text': self.bank_info['branches'], 'type': ResponseType.MESSAGE.value},
            'contact_info': lambda: {'text': self.bank_info['contact'], 'type': ResponseType.MESSAGE.value},
            'negation': lambda: self._handle_ml_negation(user_input),
            'onboarding_help': lambda: self.knowledge_manager.get_menu_response(MenuType.ONBOARDING),
            'alhuda_inquiry': lambda: self._get_knowledge_response("alhuda"),
            'menu_request': lambda: self.knowledge_manager.get_menu_response(MenuType.MAIN)
        }
        
        handler = intent_handlers.get(intent, lambda: self._get_fallback_response())
        response = handler()
        
        return response
    
    def _handle_rule_based_intent(self, user_input, session_id, context):
        """Handle message using rule-based approach"""
        user_input_lower = user_input.lower()
        
        # Check for greetings
        if any(word in user_input_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return self._get_greeting_response(session_id)
        
        # Check for negation
        if self.contains_negation(user_input_lower):
            negation_context = self.extract_negation_context(user_input_lower)
            return self._handle_negation_response(negation_context)
        
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
            return {'text': self.bank_info['branches'], 'type': ResponseType.MESSAGE.value}
        
        if any(word in user_input_lower for word in ['contact', 'phone', 'email']):
            return {'text': self.bank_info['contact'], 'type': ResponseType.MESSAGE.value}
        
        return None
    
    def _handle_ml_account_inquiry(self, user_input, session_id):
        """Handle account inquiries with ML"""
        account_type = self.extract_account_type(user_input)
        if account_type:
            # Update session context
            if session_id and session_id in self.conversation_memory:
                self.conversation_memory[session_id]['current_topic'] = f'account_{account_type}'
            return self._get_account_type_response(account_type)
        else:
            return self.knowledge_manager.get_menu_response(MenuType.ACCOUNTS)
    
    def _handle_ml_service_inquiry(self):
        """Handle service inquiries with ML"""
        return self.knowledge_manager.get_menu_response(MenuType.SERVICES)
    
    def _handle_ml_negation(self, user_input):
        """Handle negation with ML"""
        negation_context = self.extract_negation_context(user_input)
        return self._handle_negation_response(negation_context)
    
    def _get_knowledge_response(self, key):
        """Get response from knowledge base"""
        return self.knowledge_manager._get_knowledge_response(key)
    
    def _get_greeting_response(self, session_id=None):
        """Get greeting response"""
        user_name = None
        if session_id and session_id in self.conversation_memory:
            user_name = self.conversation_memory[session_id].get('user_name')
        
        greeting = "Hello! 👋 "
        if user_name:
            greeting += f"{user_name}, "
        greeting += "Welcome to our Banking Services! \n\nI'm here to help you with all your banking needs."
        
        return {
            'text': greeting,
            'type': ResponseType.OPTIONS.value,
            'options': self.knowledge_manager.menus[MenuType.MAIN]["options"]
        }
    
    def _get_account_type_response(self, account_type, user_name=None):
        """Get account type response"""
        if account_type not in self.account_info:
            # Try knowledge base
            knowledge_response = self.knowledge_manager.find_best_match(account_type)
            if knowledge_response:
                return {
                    'text': knowledge_response,
                    'type': ResponseType.MESSAGE.value,
                    'source': 'knowledge_base'
                }
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
            'type': ResponseType.OPTIONS.value,
            'options': [
                {'value': 'documents', 'label': '📄 Documents'},
                {'value': 'features', 'label': '⭐ Features'},
                {'value': 'fees', 'label': '💵 Fees'},
                {'value': 'process', 'label': '⏰ Process'}
            ],
            'state': {'account_type': account_type}
        }
    
    def _get_loan_information(self):
        """Get loan information"""
        return self.knowledge_manager.get_menu_response(MenuType.LOANS)
    
    def _handle_negation_response(self, negation_context):
        """Handle negative responses"""
        responses = {
            'account': {
                'text': "✅ No problem! What other service can I help you with?",
                'type': ResponseType.OPTIONS.value,
                'options': self.knowledge_manager.menus[MenuType.MAIN]["options"]
            },
            'loan': {
                'text': "✅ Understood! Would you like information about other services?",
                'type': ResponseType.OPTIONS.value,
                'options': [
                    {'value': 'accounts', 'label': '🏦 Accounts'},
                    {'value': 'services', 'label': '🛎️ Services'},
                    {'value': 'support', 'label': '📞 Support'}
                ]
            },
            'general': {
                'text': "✅ I understand. How else can I assist you today?",
                'type': ResponseType.OPTIONS.value,
                'options': self.knowledge_manager.menus[MenuType.MAIN]["options"]
            }
        }
        
        return responses.get(negation_context, responses['general'])
    
    def _get_fallback_response(self):
        """Get fallback response"""
        return {
            'text': "I'm here to help with your banking needs! What would you like to know?",
            'type': ResponseType.OPTIONS.value,
            'options': self.knowledge_manager.menus[MenuType.MAIN]["options"]
        }
    
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

# Export for import
__all__ = ['DialogFlowManager', 'MenuType', 'ResponseType']

# ==============================================
# 🧪 Example Test
# ==============================================
if __name__ == "__main__":
    ela = DialogFlowManager()
    
    print("🏦 Enhanced Banking Chatbot with ML + Knowledge Base + Menu System!")
    print("Type 'menu' for options, 'back' to go back, or 'exit' to quit.\n")
    
    test_scenarios = [
        "hello",
        "I want to open an account",
        "what is alhuda",
        "menu",
        "account types",
        "individual",
        "what documents do I need",
        "how to open account"
    ]
    
    for scenario in test_scenarios:
        print(f"You: {scenario}")
        response = ela.process_message(scenario)
        print(f"Ela: {response.get('text', '')}")
        if response.get('options'):
            print("Options:", [opt['label'] for opt in response['options']])
        print()