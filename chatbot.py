"""
Intelligent Credit Risk Chatbot
Answers questions about loan eligibility and credit risk assessment
"""

import re
import pandas as pd
import numpy as np

class CreditRiskChatbot:
    def __init__(self, model, dataset_path=None):
        self.model = model
        self.dataset_path = dataset_path
        self.conversation_history = []
        
        # Keywords for credit/loan related questions
        self.credit_keywords = [
            'loan', 'credit', 'approval', 'eligibility', 'income', 'age', 'employment',
            'default', 'interest', 'rate', 'amount', 'grade', 'home', 'ownership',
            'history', 'length', 'reject', 'denied', 'approved', 'why', 'reason',
            'criteria', 'requirement', 'improve', 'chance', 'probability', 'risk',
            'score', 'ratio', 'intent', 'mortgage', 'rent', 'consolidation',
            'medical', 'education', 'personal', 'venture', 'improvement'
        ]
        
        # Out-of-scope keywords
        self.out_of_scope_keywords = [
            'time', 'weather', 'date', 'news', 'sports', 'movie', 'music', 'recipe',
            'cooking', 'travel', 'hotel', 'flight', 'restaurant', 'game', 'sport',
            'atlanta', 'location', 'place', 'map', 'direction'
        ]
    
    def is_credit_related(self, query):
        """Check if the query is related to credit/loan"""
        query_lower = query.lower()
        credit_count = sum(1 for keyword in self.credit_keywords if keyword in query_lower)
        out_of_scope_count = sum(1 for keyword in self.out_of_scope_keywords if keyword in query_lower)
        
        # If it has strong out-of-scope indicators, reject it
        if out_of_scope_count > 0 and credit_count == 0:
            return False
        return credit_count > 0
    
    def explain_rejection(self, input_data, prediction):
        """Explain credit risk assessment based on input features"""
        if prediction == 1:
            return "Your credit profile indicates **Low Credit Risk**. Your application shows favorable credit characteristics with good credit history, stable income, and reasonable loan-to-income ratio."
        
        reasons = []
        
        # Check loan percent of income
        loan_percent = input_data.get('loan_percent_income', 0)
        if loan_percent > 0.5:
            reasons.append(f"Your loan amount ({loan_percent*100:.1f}% of income) is too high relative to your income. Aim for less than 30% to reduce credit risk.")
        
        # Check default on file
        if input_data.get('cb_person_default_on_file') == 'Y':
            reasons.append("You have a default on file, which significantly increases credit risk. Consider improving your credit history to reduce risk.")
        
        # Check loan grade
        loan_grade = input_data.get('loan_grade', '')
        if loan_grade in ['E', 'F', 'G']:
            reasons.append(f"Your loan grade ({loan_grade}) is low, indicating higher credit risk. Consider improving your credit score first.")
        
        # Check employment length
        emp_length = input_data.get('person_emp_length', 0)
        if emp_length < 2:
            reasons.append("Your employment length is relatively short. Longer employment history helps reduce credit risk.")
        
        # Check credit history length
        cred_hist = input_data.get('cb_person_cred_hist_length', 0)
        if cred_hist < 3:
            reasons.append("Your credit history is relatively short. Building a longer credit history can help reduce credit risk.")
        
        # Check interest rate
        int_rate = input_data.get('loan_int_rate', 0)
        if int_rate > 18:
            reasons.append(f"Your interest rate ({int_rate:.1f}%) is quite high, indicating higher credit risk. Consider improving your credit profile.")
        
        # Check loan to income ratio
        loan_to_income = input_data.get('Loan_to_income_Ratio', 0)
        if loan_to_income > 0.6:
            reasons.append(f"Your loan-to-income ratio ({loan_to_income:.2f}) is high. Consider reducing the loan amount or increasing your income to lower credit risk.")
        
        # Check income
        income = input_data.get('person_income', 0)
        if income < 30000:
            reasons.append("Your annual income is relatively low. Higher income helps reduce credit risk.")
        
        if not reasons:
            reasons.append("Based on a combination of factors including credit history, income, loan amount, and risk profile, your credit profile indicates higher risk.")
        
        return "Your credit profile indicates **High Credit Risk**. Here are the main contributing factors:\n\n" + "\n".join(f"• {r}" for r in reasons)
    
    def answer_general_question(self, query, input_data=None):
        """Answer general questions about credit/loan features"""
        query_lower = query.lower()
        
        # Income related
        if any(word in query_lower for word in ['income', 'salary', 'earn', 'wage']):
            return """**About Income:**
            • Higher income generally improves loan approval chances
            • Aim to keep your loan amount below 30% of your annual income
            • Stable income history is important for lenders
            • Multiple income sources can strengthen your application"""
        
        # Age related
        if any(word in query_lower for word in ['age', 'old', 'young']):
            return """**About Age:**
            • Age is a factor in loan assessment but not the primary one
            • Applicants typically need to be at least 18 years old
            • Older applicants with stable income and credit history may have advantages
            • Focus on maintaining good credit regardless of age"""
        
        # Employment related
        if any(word in query_lower for word in ['employment', 'job', 'work', 'employ']):
            return """**About Employment:**
            • Longer employment history (2+ years) is preferred
            • Stable employment demonstrates financial reliability
            • Self-employed applicants may need additional documentation
            • Employment length is combined with income to assess stability"""
        
        # Default related
        if any(word in query_lower for word in ['default', 'missed', 'payment', 'delinquent']):
            return """**About Defaults:**
            • Having a default on file significantly reduces approval chances
            • Defaults indicate past payment difficulties
            • Work on building positive payment history to offset defaults
            • Consider waiting and improving credit before reapplying"""
        
        # Loan amount related
        if any(word in query_lower for word in ['loan amount', 'borrow', 'how much']):
            return """**About Loan Amount:**
            • Loan amount should be proportional to your income
            • Keep loan-to-income ratio below 0.5 (ideally 0.3 or less)
            • Higher loan amounts require stronger credit profiles
            • Consider borrowing only what you need and can afford"""
        
        # Interest rate related
        if any(word in query_lower for word in ['interest', 'rate', 'apr']):
            return """**About Interest Rates:**
            • Interest rates reflect the risk level of the loan
            • Higher rates indicate higher perceived risk
            • Better credit scores typically get lower rates
            • Rates between 7-15% are typical for good credit profiles"""
        
        # Loan grade related
        if any(word in query_lower for word in ['grade', 'rating', 'score']):
            return """**About Loan Grades:**
            • Loan grades range from A (best) to G (highest risk)
            • Grades A-C are generally favorable
            • Grades D-G indicate higher risk and may face stricter requirements
            • Improve credit score to get better loan grades"""
        
        # Home ownership related
        if any(word in query_lower for word in ['home', 'ownership', 'own', 'rent', 'mortgage']):
            return """**About Home Ownership:**
            • Home ownership status (OWN, MORTGAGE, RENT, OTHER) is considered
            • Owning a home can indicate stability
            • However, it's not the sole determining factor
            • Other factors like income and credit history are more important"""
        
        # Credit history related
        if any(word in query_lower for word in ['credit history', 'credit length', 'history length']):
            return """**About Credit History Length:**
            • Longer credit history (3+ years) is preferred
            • It demonstrates experience managing credit
            • Build credit history by maintaining accounts in good standing
            • Even with short history, good payment behavior matters"""
        
        # Loan intent related
        if any(word in query_lower for word in ['intent', 'purpose', 'why', 'reason for loan']):
            return """**About Loan Intent:**
            • Common purposes: EDUCATION, MEDICAL, PERSONAL, DEBTCONSOLIDATION, HOMEIMPROVEMENT, VENTURE
            • Purpose can influence risk assessment
            • Debt consolidation loans may be viewed differently than education loans
            • Be clear and honest about loan purpose"""
        
        # Credit risk criteria
        if any(word in query_lower for word in ['criteria', 'requirement', 'what need', 'how to get approved', 'low risk', 'risk factors']):
            return """**Low Credit Risk Criteria:**
            • Stable income that's sufficient for loan payments
            • Good credit history with no recent defaults
            • Reasonable loan-to-income ratio (under 30%)
            • Employment stability (2+ years preferred)
            • Longer credit history (3+ years preferred)
            • Loan grade of A-C is favorable
            • Interest rate appropriate for your risk profile"""
        
        # Improvement suggestions
        if any(word in query_lower for word in ['improve', 'better', 'increase chance', 'how to', 'reduce risk', 'lower risk']):
            return """**How to Reduce Credit Risk:**
            1. **Increase Income**: Higher income reduces credit risk
            2. **Reduce Loan Amount**: Request smaller loans relative to income
            3. **Clear Defaults**: Work on improving credit history
            4. **Build Credit History**: Maintain accounts in good standing
            5. **Stable Employment**: Longer employment history helps
            6. **Improve Credit Score**: Better scores lead to better loan grades and lower risk
            7. **Wait if Needed**: Sometimes waiting and improving profile helps reduce risk"""
        
        # Default response for credit-related but not specific
        return """I can help you understand:
        • Credit risk assessment criteria and factors
        • Why your credit profile might indicate high or low risk
        • How to reduce your credit risk
        • Information about income, credit history, employment, loan amounts, interest rates, and more
        
        Please ask a specific question about credit risk assessment or any feature in the application form."""
    
    def get_response(self, query, input_data=None, prediction=None):
        """Main method to get chatbot response"""
        # Handle empty query
        if not query or not query.strip():
            return "Please ask me a question about loan eligibility, credit risk, or loan approval."
        
        query = query.strip()
        
        # Store conversation
        self.conversation_history.append(("user", query))
        
        # Check if out of scope
        if not self.is_credit_related(query):
            response = "I'm a specialized credit risk assistant. I can only answer questions related to loan eligibility, credit risk assessment, and the features used in this application (income, age, employment, credit history, loan details, etc.).\n\nPlease ask me about credit, loans, or loan approval criteria instead."
            self.conversation_history.append(("assistant", response))
            return response
        
        # Check if asking about credit risk assessment
        if any(word in query.lower() for word in ['why', 'reason', 'not approved', 'denied', 'reject', 'not selected', 'credit risk high', 'credit risk low', 'high risk', 'low risk', 'why is my risk']):
            if input_data is not None and prediction is not None:
                response = self.explain_rejection(input_data, prediction)
            else:
                response = "To explain your credit risk assessment, please first submit your application details using the form above. Then ask me about your credit risk level."
            self.conversation_history.append(("assistant", response))
            return response
        
        # Answer general questions
        response = self.answer_general_question(query, input_data)
        self.conversation_history.append(("assistant", response))
        return response

