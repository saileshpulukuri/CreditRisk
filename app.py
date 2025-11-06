import streamlit as st
import pandas as pd
import joblib
from chatbot import CreditRiskChatbot
import time

# Page configuration
st.set_page_config(
    page_title="Credit Risk Prediction | Loan Approval AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .prediction-card {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .approved-card {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #155724;
        border-left: 4px solid #28a745;
    }
    
    .rejected-card {
        background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%);
        color: #721c24;
        border-left: 4px solid #dc3545;
    }
    
    /* Chatbot styling */
    .chat-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        max-height: 500px;
        overflow-y: auto;
        margin-bottom: 1rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        animation: fadeIn 0.3s;
    }
    
    .user-message {
        background: #667eea;
        color: white;
        margin-left: 20%;
        text-align: right;
    }
    
    .bot-message {
        background: white;
        color: #333;
        margin-right: 20%;
        border: 1px solid #e0e0e0;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Input form styling */
    .stNumberInput > div > div > input {
        border-radius: 8px;
    }
    
    .stSelectbox > div > div > select {
        border-radius: 8px;
    }
    
    /* Chat input field styling - blue border only, remove red */
    div[data-testid="stForm"] input[type="text"] {
        border: 2px solid #667eea !important;
        border-radius: 8px !important;
        outline: none !important;
        box-shadow: none !important;
    }
    
    div[data-testid="stForm"] input[type="text"]:focus {
        border: 2px solid #667eea !important;
        outline: none !important;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25) !important;
    }
    
    /* Remove any red borders from form elements */
    div[data-testid="stForm"] input[type="text"]:invalid {
        border: 2px solid #667eea !important;
        box-shadow: none !important;
    }
    
    /* Send button styling - blue gradient, remove red */
    div[data-testid="stForm"] button[type="submit"],
    div[data-testid="stForm"] button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        background-color: #667eea !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
    }
    
    div[data-testid="stForm"] button[type="submit"]:hover,
    div[data-testid="stForm"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #5568d3 0%, #6a3d8f 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Override any default Streamlit red/error styling */
    div[data-testid="stForm"] > div {
        border: none !important;
    }
    
    div[data-testid="stForm"] input {
        border-color: #667eea !important;
    }
    
    /* Remove all red borders and outlines */
    input[type="text"]:focus:not(:focus-visible) {
        outline: none !important;
        border-color: #667eea !important;
    }
    
    /* Target Streamlit's form input wrapper to remove red border */
    div[data-baseweb="input"] {
        border-color: #667eea !important;
    }
    
    div[data-baseweb="input"]:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25) !important;
    }
    
    /* Remove red from button - ensure blue */
    button[kind="primaryFormSubmit"],
    button[data-testid="baseButton-primaryFormSubmit"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        background-color: #667eea !important;
        border-color: #667eea !important;
    }
    
    /* Remove any red border from form container */
    form[data-testid="stForm"] {
        border: none !important;
    }
    
    /* Ensure input wrapper has no red border */
    div[data-baseweb="base-input"] {
        border-color: #667eea !important;
    }
    
    /* Comprehensive removal of all red styling from forms */
    div[data-baseweb="input"] input,
    div[data-baseweb="input"] > div,
    div[data-baseweb="input"] > div > div {
        border-color: #667eea !important;
        outline-color: #667eea !important;
    }
    
    /* Remove validation red borders */
    div[data-baseweb="input"][data-state*="error"],
    div[data-baseweb="input"]:has(input:invalid) {
        border-color: #667eea !important;
    }
    
    /* Force blue on all form inputs */
    form input[type="text"],
    form input[type="text"]:focus,
    form input[type="text"]:invalid,
    form input[type="text"]:valid {
        border: 2px solid #667eea !important;
        outline: none !important;
    }
    
    /* Override Streamlit's default button colors completely */
    button[type="submit"],
    button[kind="primaryFormSubmit"],
    button[data-testid*="submit"],
    button[data-testid*="button"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        background-color: #667eea !important;
        border: none !important;
        color: white !important;
    }
    
    /* Remove any red background or border from buttons */
    button[type="submit"]:hover,
    button[kind="primaryFormSubmit"]:hover {
        background: linear-gradient(135deg, #5568d3 0%, #6a3d8f 100%) !important;
        background-color: #5568d3 !important;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 0.75rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        .user-message {
            margin-left: 10%;
        }
        .bot-message {
            margin-right: 10%;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("loan_rf_pipeline.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Helper function to prepare input data with one-hot encoding
def prepare_input_data_with_onehot(input_dict):
    """
    Apply one-hot encoding to categorical columns to match the training data format.
    Based on the error, the model expects one-hot encoded columns like:
    - person_home_ownership_OWN, person_home_ownership_RENT, etc.
    - loan_intent_EDUCATION, loan_intent_MEDICAL, etc.
    
    Important: We need to ensure ALL possible one-hot encoded columns are created,
    even if they're 0, because the model expects a specific set of columns.
    """
    import numpy as np
    
    # Create base dataframe with all numerical and other features first
    # Keep only non-categorical columns initially
    df = pd.DataFrame([input_dict])
    
    # Categorical columns that need one-hot encoding
    # Based on the error message, these exact columns are expected:
    # person_home_ownership_OWN, person_home_ownership_RENT, person_home_ownership_OTHER
    # loan_intent_HOMEIMPROVEMENT, loan_intent_VENTURE, loan_intent_EDUCATION, 
    # loan_intent_PERSONAL, loan_intent_MEDICAL
    # Note: person_home_ownership_MORTGAGE and loan_intent_DEBTCONSOLIDATION might be dropped (first category)
    categorical_cols = {
        'person_home_ownership': ['OWN', 'RENT', 'MORTGAGE', 'OTHER'],
        'loan_intent': ['HOMEIMPROVEMENT', 'VENTURE', 'EDUCATION', 'PERSONAL', 'MEDICAL', 'DEBTCONSOLIDATION'],
        'cb_person_default_on_file': ['Y', 'N']  # Also encode this binary categorical column
    }
    
    # First, create all one-hot encoded columns for categorical features
    all_encoded_cols = {}
    for col, categories in categorical_cols.items():
        if col in df.columns:
            # Create one-hot encoded columns for ALL categories
            for category in categories:
                encoded_col_name = f"{col}_{category}"
                all_encoded_cols[encoded_col_name] = (df[col].iloc[0] == category).astype(int)
            
            # Drop the original categorical column
            df = df.drop(columns=[col])
    
    # Add all one-hot encoded columns to the dataframe
    for col_name, value in all_encoded_cols.items():
        df[col_name] = value
    
    # Ensure all columns are numeric (convert to float/int)
    # This is important for sparse output compatibility
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            except:
                df[col] = 0
        # Ensure all numeric columns are float64 for consistency
        elif df[col].dtype in ['int64', 'int32', 'float32']:
            df[col] = df[col].astype('float64')
    
    return df

# Helper function to prepare input data in the correct format
def prepare_input_data(input_dict, model):
    """
    Prepare input data to match what the model expects.
    
    CRITICAL: Based on the error messages, the model expects ONE-HOT ENCODED columns,
    not raw categorical values. The pipeline's OneHotEncoder with drop='first' means:
    - person_home_ownership_MORTGAGE is dropped (first category)
    - loan_intent_DEBTCONSOLIDATION is dropped (first category)
    
    So we need to manually create the one-hot encoded columns that the model expects.
    """
    # Create dataframe with raw values first
    input_df = pd.DataFrame([input_dict])
    
    # Based on the error, these are the expected one-hot encoded columns:
    # person_home_ownership_OWN, person_home_ownership_RENT, person_home_ownership_OTHER
    # (MORTGAGE is dropped - first category)
    # loan_intent_HOMEIMPROVEMENT, loan_intent_VENTURE, loan_intent_EDUCATION,
    # loan_intent_PERSONAL, loan_intent_MEDICAL
    # (DEBTCONSOLIDATION is dropped - first category)
    
    # One-hot encode person_home_ownership (drop MORTGAGE - first category)
    home_ownership = input_dict.get('person_home_ownership', 'RENT')
    input_df['person_home_ownership_OWN'] = int(home_ownership == 'OWN')
    input_df['person_home_ownership_RENT'] = int(home_ownership == 'RENT')
    input_df['person_home_ownership_OTHER'] = int(home_ownership == 'OTHER')
    # MORTGAGE is dropped (first category with drop='first')
    
    # One-hot encode loan_intent (drop DEBTCONSOLIDATION - first category)
    loan_intent = input_dict.get('loan_intent', 'PERSONAL')
    input_df['loan_intent_HOMEIMPROVEMENT'] = int(loan_intent == 'HOMEIMPROVEMENT')
    input_df['loan_intent_VENTURE'] = int(loan_intent == 'VENTURE')
    input_df['loan_intent_EDUCATION'] = int(loan_intent == 'EDUCATION')
    input_df['loan_intent_PERSONAL'] = int(loan_intent == 'PERSONAL')
    input_df['loan_intent_MEDICAL'] = int(loan_intent == 'MEDICAL')
    # DEBTCONSOLIDATION is dropped (first category with drop='first')
    
    # Drop original categorical columns
    if 'person_home_ownership' in input_df.columns:
        input_df = input_df.drop(columns=['person_home_ownership'])
    if 'loan_intent' in input_df.columns:
        input_df = input_df.drop(columns=['loan_intent'])
    
    # Ensure all numeric columns are properly typed
    numeric_cols = [
        'person_age', 'person_income', 'person_emp_length', 'loan_grade',
        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_default_on_file', 'cb_person_cred_hist_length',
        'Loan_to_income_Ratio', 'Loan_to_emp_length_ratio', 'int_rate_to_loan_amnt_ratio'
    ]
    
    for col in numeric_cols:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
    
    # Handle cb_person_default_on_file - might need encoding or numeric conversion
    if 'cb_person_default_on_file' in input_df.columns:
        # Convert Y/N to 1/0
        input_df['cb_person_default_on_file'] = (input_df['cb_person_default_on_file'] == 'Y').astype(int)
    
    # Handle loan_grade - might be ordinal, convert to numeric if needed
    if 'loan_grade' in input_df.columns:
        # Convert letter grades to numbers if it's a string
        if input_df['loan_grade'].dtype == 'object':
            grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
            input_df['loan_grade'] = input_df['loan_grade'].map(grade_map).fillna(0)
        input_df['loan_grade'] = pd.to_numeric(input_df['loan_grade'], errors='coerce').fillna(0)
    
    # Ensure all one-hot encoded columns are int (not float)
    onehot_cols = [
        'person_home_ownership_OWN', 'person_home_ownership_RENT', 'person_home_ownership_OTHER',
        'loan_intent_HOMEIMPROVEMENT', 'loan_intent_VENTURE', 'loan_intent_EDUCATION',
        'loan_intent_PERSONAL', 'loan_intent_MEDICAL'
    ]
    for col in onehot_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(int)
    
    return input_df

# Initialize chatbot
@st.cache_resource
def init_chatbot(_model):
    return CreditRiskChatbot(_model)

# Load model and initialize chatbot
model = load_model()
if model is None:
    st.error("Failed to load model. Please check if 'loan_rf_pipeline.pkl' exists.")
    st.stop()

chatbot = init_chatbot(model)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_processed_query' not in st.session_state:
    st.session_state.last_processed_query = None

# Main header
st.markdown("""
    <div class="main-header">
        <h1> Credit Risk Prediction System</h1>
        <p>AI-Powered Credit Risk Assessment with Interactive Assistant</p>
    </div>
""", unsafe_allow_html=True)

# Main form container
st.markdown("###  Application Form")

# Row 1: Personal Information and Loan Information side by side
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.markdown("####  Personal Information")
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1, key="age")
    person_income = st.number_input("Annual Income ($)", min_value=10000, max_value=1000000, value=50000, step=1000, key="income")
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"], key="home")
    person_emp_length = st.number_input("Employment Length (Years)", min_value=0.0, max_value=50.0, value=5.0, step=0.5, key="emp")

with row1_col2:
    st.markdown("####  Loan Information")
    loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT", "VENTURE"], key="intent")
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"], key="grade")
    loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, max_value=50000, value=10000, step=1000, key="amnt")
    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=1.0, max_value=40.0, value=10.0, step=0.1, key="rate")
    loan_percent_income = st.number_input("Loan Percent of Income", min_value=0.0, max_value=1.0, value=0.2, step=0.01, format="%.2f", key="percent")

# Row 2: Credit Information
st.markdown("####  Credit Information")
row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    cb_person_default_on_file = st.selectbox("Default on File", ["Y", "N"], key="default")
with row2_col2:
    cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, max_value=50, value=5, step=1, key="cred_hist")

# Calculate derived features
Loan_to_income_Ratio = loan_amnt / person_income if person_income > 0 else 0
Loan_to_emp_length_ratio = loan_amnt / (person_emp_length if person_emp_length != 0 else 1)
int_rate_to_loan_amnt_ratio = loan_int_rate / loan_amnt if loan_amnt > 0 else 0

# Store input data
input_data = {
    "person_age": person_age,
    "person_income": person_income,
    "person_home_ownership": person_home_ownership,
    "person_emp_length": person_emp_length,
    "loan_intent": loan_intent,
    "loan_grade": loan_grade,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_default_on_file": cb_person_default_on_file,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "Loan_to_income_Ratio": Loan_to_income_Ratio,
    "Loan_to_emp_length_ratio": Loan_to_emp_length_ratio,
    "int_rate_to_loan_amnt_ratio": int_rate_to_loan_amnt_ratio
}

# Display key metrics
st.markdown("###  Key Metrics")
metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.metric("Loan-to-Income Ratio", f"{Loan_to_income_Ratio:.2f}")
with metric_col2:
    st.metric("Loan Percent", f"{loan_percent_income*100:.1f}%")
with metric_col3:
    st.metric("Interest Rate", f"{loan_int_rate:.1f}%")

# Predict button
st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
if st.button("🔍 Assess Credit Risk", type="primary", use_container_width=True):
    with st.spinner("Assessing your credit risk..."):
        time.sleep(0.5)  # Small delay for better UX
        
        # Prepare input dataframe with one-hot encoding
        # The model expects one-hot encoded columns based on the error message
        input_df = prepare_input_data(input_data, model)
        
        # Debug: Show created columns (commented out in production)
        # st.write("Created columns:", list(input_df.columns))
        
        # Make prediction
        try:
            # The model expects one-hot encoded categorical columns
            prediction = model.predict(input_df)[0]
            prediction_proba = None
            try:
                prediction_proba = model.predict_proba(input_df)[0]
            except:
                pass
            
            st.session_state.prediction_made = True
            st.session_state.prediction_result = prediction
            st.session_state.input_data = input_data
            st.session_state.prediction_proba = prediction_proba
            
            st.rerun()
        except Exception as e:
            # More detailed error handling
            error_msg = str(e)
            st.error(f"Error making prediction: {error_msg}")
            
            # If it's a column mismatch error, provide helpful guidance
            if "columns are missing" in error_msg or "missing" in error_msg.lower():
                st.warning("""
                **Column Mismatch Detected**
                
                The model expects different column names than what we're providing. 
                This usually happens when the pipeline structure doesn't match the input format.
                
                **Trying to fix automatically...**
                """)
                
                # Try to get feature names from pipeline
                try:
                    # Check if we can inspect the pipeline structure
                    if hasattr(model, 'feature_names_in_'):
                        expected_cols = model.feature_names_in_
                        st.info(f"Expected columns: {list(expected_cols)[:10]}...")
                    elif hasattr(model, 'named_steps'):
                        st.info(f"Pipeline steps: {list(model.named_steps.keys())}")
                except:
                    pass
                
                # Try alternative: use prepare_input_data function
                try:
                    input_df_fixed = prepare_input_data(input_data, model)
                    prediction = model.predict(input_df_fixed)[0]
                    prediction_proba = None
                    try:
                        prediction_proba = model.predict_proba(input_df_fixed)[0]
                    except:
                        pass
                    
                    st.session_state.prediction_made = True
                    st.session_state.prediction_result = prediction
                    st.session_state.input_data = input_data
                    st.session_state.prediction_proba = prediction_proba
                    st.success("✅ Prediction successful after fixing column format!")
                    st.rerun()
                except Exception as e2:
                    st.error(f"Still getting error after fix attempt: {e2}")
                    st.info("""
                    **Possible Solutions:**
                    1. Check if the pipeline was saved correctly with all preprocessing steps
                    2. Verify that the model expects raw categorical values (not one-hot encoded)
                    3. The pipeline should include a OneHotEncoder or ColumnTransformer step
                    """)

# Prediction Result Section - Display below the button
if st.session_state.prediction_made:
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    st.markdown("### 🎯 Credit Risk Assessment Result")
    
    prediction = st.session_state.prediction_result
    input_data = st.session_state.input_data
    
    if prediction == 1:
            st.markdown("""
                <div class="prediction-card approved-card">
                    <h2 style="margin: 0; color: #155724;">✅ Credit Risk - Low</h2>
                    <p style="margin: 1rem 0 0 0; font-size: 1.1rem;">Your credit profile indicates low risk. Your application shows favorable credit characteristics.</p>
                </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.prediction_proba is not None:
                proba = st.session_state.prediction_proba[1] * 100
                st.metric("Low Risk Confidence", f"{proba:.1f}%")
    else:
            st.markdown("""
                <div class="prediction-card rejected-card">
                    <h2 style="margin: 0; color: #721c24;">⚠️ Credit Risk - High</h2>
                    <p style="margin: 1rem 0 0 0; font-size: 1.1rem;">Your credit profile indicates high risk. Consider improving your credit factors.</p>
                </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.prediction_proba is not None:
                proba = st.session_state.prediction_proba[0] * 100
                st.metric("High Risk Confidence", f"{proba:.1f}%")

# Chatbot Section
st.markdown("---")
st.markdown("##  Credit Risk Assistant Chatbot")

st.markdown("""
    <div style="background: #1e3a5f; padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #667eea;">
        <p style="margin: 0; color: #ffffff; font-size: 1.1rem;"><strong>💡 Tip:</strong> Ask me questions like:</p>
        <ul style="margin: 0.5rem 0 0 0; color: #ffffff; font-size: 1rem;">
            <li style="margin: 0.5rem 0;">"Why is my credit risk high?" (after submitting your application)</li>
            <li style="margin: 0.5rem 0;">"How does income affect credit risk?"</li>
            <li style="margin: 0.5rem 0;">"What is a good loan-to-income ratio?"</li>
            <li style="margin: 0.5rem 0;">"How can I reduce my credit risk?"</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Chat interface
chat_container = st.container()

# Display chat history
if st.session_state.chat_history:
    with chat_container:
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {message}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>Assistant:</strong><br>{message.replace(chr(10), '<br>')}
                    </div>
                """, unsafe_allow_html=True)

# Chat input using form for better clearing behavior
# Using st.form with clear_on_submit=True automatically clears the input after submission
with st.form(key="chat_form", clear_on_submit=True):
    # Create a row with input field and button
    input_col, btn_col = st.columns([5, 1])
    with input_col:
        user_query = st.text_input(
            "Ask me anything about loan eligibility, credit risk, or your application:",
            placeholder="e.g., Why is my credit risk high? How does income affect credit risk?",
            key="chat_input_form",
            label_visibility="visible"
        )
    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)  # Align button with input
        submitted = st.form_submit_button("Send ➤", type="primary", use_container_width=True)

# Process the query when form is submitted
if submitted and user_query and user_query.strip():
    query_trimmed = user_query.strip()
    
    # Check if this is a new query (not a duplicate)
    if query_trimmed != st.session_state.last_processed_query:
        # Get prediction and input data if available
        pred = st.session_state.prediction_result if st.session_state.prediction_made else None
        inp_data = st.session_state.input_data if st.session_state.prediction_made else None
        
        # Get chatbot response
        response = chatbot.get_response(query_trimmed, inp_data, pred)
        
        # Add to chat history
        st.session_state.chat_history.append(("user", query_trimmed))
        st.session_state.chat_history.append(("assistant", response))
        st.session_state.last_processed_query = query_trimmed
        
        st.rerun()

# Footer
st.markdown("---")
# st.markdown("""
#     <div style="text-align: center; color: #666; padding: 1rem;">
#         <p>Built with ❤️ using Streamlit & Machine Learning</p>
#         <p style="font-size: 0.9rem;">This is an AI-powered credit risk assessment tool. Results are for demonstration purposes.</p>
#     </div>
# """, unsafe_allow_html=True)
