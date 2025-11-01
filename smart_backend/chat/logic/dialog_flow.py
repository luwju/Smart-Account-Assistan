class DialogFlowManager:
    def __init__(self):
        self.account_types = {
            'ind': 'individual',
            'individual': 'individual',
            'join': 'joint', 
            'joint': 'joint',
            'cmp': 'company',
            'company': 'company',
            'other': 'other'
        }
        
        self.account_info = {
            'individual': {
                'documents': [
                    'Original National ID/Passport',
                    'Passport-size photo',
                    'KRA PIN certificate',
                    'Proof of income (payslip, bank statement)'
                ],
                'min_amount': 1000,
                'personal_info': [
                    'Full Name',
                    'Date of Birth',
                    'Email Address', 
                    'Phone Number',
                    'Physical Address',
                    'Occupation'
                ],
                'features': [
                    'Free mobile banking',
                    'Visa debit card',
                    'Monthly statements',
                    '24/7 customer support'
                ]
            },
            'joint': {
                'documents': [
                    'Original National IDs/Passports for all applicants',
                    'Passport-size photos for all applicants',
                    'KRA PIN certificates for all applicants',
                    'Joint application form'
                ],
                'min_amount': 2000,
                'personal_info': [
                    'Full Names of all applicants',
                    'Dates of Birth',
                    'Email Addresses',
                    'Phone Numbers', 
                    'Physical Address',
                    'Occupations'
                ],
                'features': [
                    'All individual account features',
                    'Multiple card holders',
                    'Joint online banking access',
                    'Flexible signing arrangements'
                ]
            },
            'company': {
                'documents': [
                    'Certificate of Incorporation',
                    'Company KRA PIN',
                    'Directors IDs and photos',
                    'Company resolution to open account',
                    'Business registration documents'
                ],
                'min_amount': 5000,
                'personal_info': [
                    'Company Name',
                    'Business Email',
                    'Official Address',
                    'Company Phone Number',
                    'Directors Information',
                    'Business Nature'
                ],
                'features': [
                    'Corporate internet banking',
                    'Multiple authorized signatories',
                    'Business Visa cards',
                    'Payroll services',
                    'Merchant services'
                ]
            }
        }
        
        self.general_responses = {
            'hours': "Our banking hours are:\n• Weekdays: 8:00 AM - 5:00 PM\n• Saturdays: 9:00 AM - 1:00 PM\n• Sundays: Closed",
            'location': "We have branches nationwide! Visit our website for the nearest branch or use our mobile app to locate us.",
            'contact': "Contact us at:\n• Phone: 0703 000 000\n• Email: info@coopbank.co.ke\n• WhatsApp: 07XX XXX XXX",
            'services': "We offer:\n• Account Opening\n• Loans & Credit\n• Mobile Banking\n• International Banking\n• Investment Services"
        }
    
    def process_message(self, user_input, current_state=None):
        user_input = user_input.lower().strip()
        
        # Reset state if starting over
        if any(word in user_input for word in ['start over', 'reset', 'new', 'begin']):
            return self._get_greeting_response()
        
        # Greeting
        if any(word in user_input for word in ['hi', 'hello', 'hey', 'good morning', 'good afternoon']):
            return self._get_greeting_response()
        
        # Account type selection
        account_type = self._extract_account_type(user_input)
        if account_type:
            return self._get_account_type_response(account_type)
        
        # Information requests within account context
        if current_state and current_state.get('account_type'):
            return self._handle_info_request(user_input, current_state['account_type'])
        
        # General inquiries
        general_response = self._handle_general_inquiry(user_input)
        if general_response:
            return general_response
        
        return self._get_fallback_response()
    
    def _extract_account_type(self, user_input):
        for key, value in self.account_types.items():
            if key in user_input:
                return value
        return None
    
    def _get_greeting_response(self):
        return {
            'text': "👋 Hi, welcome to Coop Bank! I'm here to help you open an account.\n\nPlease choose your account type:",
            'options': [
                {'value': 'ind', 'label': '🏦 Individual Account'},
                {'value': 'join', 'label': '👥 Joint Account'},
                {'value': 'cmp', 'label': '🏢 Company Account'},
                {'value': 'other', 'label': '❓ Other Services'}
            ],
            'type': 'options',
            'state': {}  # Reset state
        }
    
    def _get_account_type_response(self, account_type):
        if account_type == 'other':
            return self._get_other_services_response()
            
        return {
            'text': f"✅ Great! You selected {account_type.capitalize()} account. What would you like to know about it?",
            'options': [
                {'value': 'doc', 'label': '📄 Documents Required'},
                {'value': 'min', 'label': '💰 Minimum Amount'},
                {'value': 'info', 'label': '👤 Personal Info Needed'},
                {'value': 'features', 'label': '⭐ Account Features'},
                {'value': 'other', 'label': '❓ Other Question'}
            ],
            'type': 'options',
            'state': {'account_type': account_type}
        }
    
    def _get_other_services_response(self):
        return {
            'text': "🔍 Please tell me what other banking service you're interested in:\n\n• Loans & Credit Facilities\n• Mobile Banking Setup\n• Card Services\n• Internet Banking\n• Investment Options\n• Or type your specific question",
            'type': 'text_input',
            'state': {'service_inquiry': True}
        }
    
    def _handle_info_request(self, user_input, account_type):
        user_input = user_input.lower()
        
        if any(word in user_input for word in ['doc', 'document', 'require', 'need', 'paper']):
            docs = self.account_info[account_type]['documents']
            return {
                'text': f"📋 Documents required for {account_type} account:\n\n" + "\n".join([f"• {doc}" for doc in docs]),
                'type': 'message',
                'state': {'account_type': account_type}
            }
        
        elif any(word in user_input for word in ['min', 'amount', 'money', 'deposit', 'open']):
            amount = self.account_info[account_type]['min_amount']
            return {
                'text': f"💰 Minimum amount to open {account_type} account: **KSh {amount:,}**\n\nThis is the initial deposit required to activate your account.",
                'type': 'message',
                'state': {'account_type': account_type}
            }
        
        elif any(word in user_input for word in ['info', 'personal', 'detail', 'information']):
            info = self.account_info[account_type]['personal_info']
            return {
                'text': f"👤 Personal information needed for {account_type} account:\n\n" + "\n".join([f"• {item}" for item in info]),
                'type': 'message',
                'state': {'account_type': account_type}
            }
        
        elif any(word in user_input for word in ['feature', 'benefit', 'offer', 'include']):
            features = self.account_info[account_type]['features']
            return {
                'text': f"⭐ {account_type.capitalize()} account features:\n\n" + "\n".join([f"• {feature}" for feature in features]),
                'type': 'message',
                'state': {'account_type': account_type}
            }
        
        else:
            return {
                'text': "🤔 I understand you have another question about this account type. Please type your specific question and I'll help you.",
                'type': 'text_input',
                'state': {'account_type': account_type}
            }
    
    def _handle_general_inquiry(self, user_input):
        if any(word in user_input for word in ['hour', 'time', 'open', 'close']):
            return {
                'text': self.general_responses['hours'],
                'type': 'message'
            }
        
        elif any(word in user_input for word in ['location', 'branch', 'where', 'address']):
            return {
                'text': self.general_responses['location'],
                'type': 'message'
            }
        
        elif any(word in user_input for word in ['contact', 'phone', 'email', 'call', 'whatsapp']):
            return {
                'text': self.general_responses['contact'],
                'type': 'message'
            }
        
        elif any(word in user_input for word in ['service', 'offer', 'product', 'provide']):
            return {
                'text': self.general_responses['services'],
                'type': 'message'
            }
        
        return None
    
    def _get_fallback_response(self):
        return {
            'text': "❓ I'm not sure I understand. You can:\n\n• Choose an account type from below\n• Ask about our services\n• Type your specific question",
            'options': [
                {'value': 'ind', 'label': '🏦 Individual Account'},
                {'value': 'join', 'label': '👥 Joint Account'}, 
                {'value': 'cmp', 'label': '🏢 Company Account'},
                {'value': 'other', 'label': '🔍 Other Services'}
            ],
            'type': 'options'
        }