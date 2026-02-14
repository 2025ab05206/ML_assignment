import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Bank Marketing ML Models",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    * {
        font-family: monospace;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
[data-testid="stSidebar"] [role="radio"][aria-checked="true"] > div:first-child {
    border-color: #1f77b4 !important;
}


[data-testid="stSidebar"] [role="radio"][aria-checked="true"] > div:first-child > div {
    background-color: #1f77b4 !important;
}

[data-testid="stSidebar"] [role="radio"] > div:first-child {
    border-color: #1f77b4 !important;
}
</style>
""", unsafe_allow_html=True)


if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None


@st.cache_data
def load_data(uploaded_file=None):
    """Load dataset from uploaded file or default path"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, sep=';')
        else:
            
            data_path = os.path.join('data', 'bank-additional-full.csv')
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, sep=';')
            else:
                return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Preprocess the dataset"""
    df_processed = df.copy()
    
   
    df_processed['y'] = df_processed['y'].map({'yes': 1, 'no': 0})
    
   
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    return df_processed, label_encoders

def train_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train and evaluate a single model"""
   
    model.fit(X_train, y_train)
    
  
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
 
    metrics = {
        'Model': model_name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'AUC': round(roc_auc_score(y_test, y_pred_proba), 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
    }
    
    return metrics, model, y_pred, y_pred_proba

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Create confusion matrix visualization"""
    cm = confusion_matrix(y_test, y_pred)
    
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['No', 'Yes'],
                    y=['No', 'Yes'],
                    text_auto=True,
                    title=f'Confusion Matrix - {model_name}',
                    color_continuous_scale='Blues')
    
    return fig

def plot_roc_curve(y_test, y_pred_proba, model_name):
    """Create ROC curve visualization"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_name} (AUC = {auc:.4f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        hovermode='x'
    )
    
    return fig

def plot_metrics_comparison(results_df):
    """Create comprehensive metrics comparison"""
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=metrics,
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    
    for idx, metric in enumerate(metrics):
        row, col = positions[idx]
        fig.add_trace(
            go.Bar(x=results_df['Model'], y=results_df[metric], name=metric,
                   showlegend=False, marker_color='lightblue'),
            row=row, col=col
        )
    
    fig.update_layout(height=600, title_text="Model Performance Comparison", showlegend=False)
    
    return fig

st.sidebar.title("Bank Marketing")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Data Overview", "Train Models", "Results & Comparison", "Make Predictions"]
)

if page == "Home":
    st.markdown('<h1 class="main-header">Bank Marketing Campaign Prediction</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Project Overview
        
        This application predicts whether a bank client will subscribe to a term deposit based on 
        direct marketing campaign data (phone calls) from a Portuguese banking institution.
        
        ### Objective
        
        Build and compare **6 different classification models** to predict customer subscription behavior, 
        helping banks optimize their marketing strategies.
        
        ### Machine Learning Models Implemented
        
        1. **Logistic Regression** - Linear probabilistic classifier
        2. **Decision Tree Classifier** - Tree-based rule learning
        3. **K-Nearest Neighbor** - Instance-based learning
        4. **Naive Bayes (Gaussian)** - Probabilistic classifier
        5. **Random Forest** - Ensemble of decision trees
        6. **XGBoost** - Gradient boosting ensemble
        
        ### Evaluation Metrics
        
        For each model, we calculate:
        - **Accuracy** - Overall correctness
        - **AUC Score** - Area under ROC curve
        - **Precision** - Positive predictive value
        - **Recall** - Sensitivity/True positive rate
        - **F1 Score** - Harmonic mean of precision and recall
        - **MCC Score** - Matthews correlation coefficient
        """)
    
    with col2:
        st.markdown("### Dataset Info")
        st.info("""
        **Source:** UCI Repository  
        **Instances:** 41,188  
        **Features:** 20 (without target)
        **Target:** Binary (yes/no)  
        **Type:** Classification
        """)
        
        st.markdown("### Quick Start")
        st.success("""
        1. Go to **Data Overview**
        2. Upload or use default dataset
        3. Navigate to **Train Models**
        4. View **Results & Comparison**
        5. Make **Predictions**
        """)

elif page == "Data Overview":
    st.header("Dataset Overview and Exploration")
    
    uploaded_file = st.file_uploader("Upload your CSV file (or use default dataset)", type=['csv'])
    
    df = load_data(uploaded_file)
    
    if df is not None:
        st.success(f"Dataset loaded successfully! Shape: {df.shape}")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Statistics", "Visualizations", "Feature Info"])
        
        with tab1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(100), use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", df.shape[0])
            col2.metric("Total Columns", df.shape[1])
            col3.metric("Numeric Features", df.select_dtypes(include=[np.number]).shape[1])
            col4.metric("Categorical Features", df.select_dtypes(include=['object']).shape[1])
        
        with tab2:
            st.subheader("Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
            
            st.subheader("Missing Values")
            missing = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            st.dataframe(missing, use_container_width=True)
        
        with tab3:
            st.subheader("Data Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Target Variable Distribution**")
                target_counts = df['y'].value_counts()
                fig = px.pie(values=target_counts.values, names=target_counts.index, 
                           title='Subscription Distribution', hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Age Distribution**")
                fig = px.histogram(df, x='age', marginal='box', title='Age Distribution')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Feature Information")
            st.markdown("""
            #### Bank Client Data
            - **age**: Age of client (numeric)
            - **job**: Type of job (categorical)
            - **marital**: Marital status (categorical)
            - **education**: Education level (categorical)
            - **default**: Has credit in default? (categorical)
            - **housing**: Has housing loan? (categorical)
            - **loan**: Has personal loan? (categorical)
            
            #### Campaign Data
            - **contact**: Contact communication type (categorical)
            - **month**: Last contact month (categorical)
            - **day_of_week**: Last contact day (categorical)
            - **duration**: Last contact duration in seconds (numeric)
            - **campaign**: Number of contacts during campaign (numeric)
            - **pdays**: Days since last contact (numeric)
            - **previous**: Number of previous contacts (numeric)
            - **poutcome**: Outcome of previous campaign (categorical)
            
            #### Economic Context
            - **emp.var.rate**: Employment variation rate (numeric)
            - **cons.price.idx**: Consumer price index (numeric)
            - **cons.conf.idx**: Consumer confidence index (numeric)
            - **euribor3m**: Euribor 3 month rate (numeric)
            - **nr.employed**: Number of employees (numeric)
            """)
    else:
        st.warning("Please upload a dataset or ensure 'bank-additional-full.csv' is in the data/ folder")
        st.info("""
        **How to get the dataset:**
        1. Visit: https://archive.ics.uci.edu/dataset/222/bank+marketing
        2. Download and extract the ZIP file
        3. Upload the 'bank-additional-full.csv' file here
        """)

elif page == "Train Models":
    st.header("Train Classification Models")
    
    uploaded_file = st.file_uploader("Upload dataset (CSV)", type=['csv'], key='train_upload')
    df = load_data(uploaded_file)
    
    if df is not None:
        st.success("Dataset loaded successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
        
        with col2:
            random_state = st.number_input("Random State (for reproducibility)", 0, 100, 42)
        
        if st.button("Train All Models", type="primary"):
            with st.spinner("Training models... Please wait"):
                
                df_processed, label_encoders = preprocess_data(df)
                
                X = df_processed.drop('y', axis=1)
                y = df_processed['y']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                models_dict = {}
                predictions_dict = {}
                
                models = [
                    (LogisticRegression(random_state=random_state, max_iter=1000), "Logistic Regression", True),
                    (DecisionTreeClassifier(random_state=random_state), "Decision Tree", False),
                    (KNeighborsClassifier(), "K-Nearest Neighbor", True),
                    (GaussianNB(), "Naive Bayes", True),
                    (RandomForestClassifier(random_state=random_state, n_estimators=100), "Random Forest", False),
                    (XGBClassifier(random_state=random_state, eval_metric='logloss', use_label_encoder=False), "XGBoost", False)
                ]
                
                for idx, (model, name, scale) in enumerate(models):
                    status_text.text(f"Training {name}...")
                    
                    X_tr = X_train_scaled if scale else X_train
                    X_te = X_test_scaled if scale else X_test
                    
                    metrics, trained_model, y_pred, y_pred_proba = train_evaluate_model(
                        model, name, X_tr, X_te, y_train, y_test
                    )
                    
                    results.append(metrics)
                    models_dict[name] = trained_model
                    predictions_dict[name] = {'y_pred': y_pred, 'y_pred_proba': y_pred_proba}
                    
                    progress_bar.progress((idx + 1) / len(models))
                
                st.session_state.results_df = pd.DataFrame(results)
                st.session_state.models_dict = models_dict
                st.session_state.predictions_dict = predictions_dict
                st.session_state.y_test = y_test
                st.session_state.X_test = X_test
                st.session_state.X_test_scaled = X_test_scaled
                st.session_state.scaler = scaler
                st.session_state.label_encoders = label_encoders
                st.session_state.feature_names = X.columns.tolist()
                st.session_state.model_scaling = {name: scale for _, name, scale in models}
                st.session_state.models_trained = True
                
                status_text.text("All models trained successfully!")
             
                
                st.subheader("Training Results")
                st.dataframe(st.session_state.results_df, use_container_width=True)
                
    else:
        st.warning("Please upload a dataset to train models")

elif page == "Results & Comparison":
    st.header("Model Results and Comparison")
    
    if st.session_state.models_trained:
        results_df = st.session_state.results_df
        
        st.subheader("Evaluation Metrics Comparison")
        st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']), 
                    use_container_width=True)
        
        best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
        st.success(f"Best Model (by Accuracy): **{best_model_name}**")
        
        st.subheader("Visual Comparison")
        fig = plot_metrics_comparison(results_df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Detailed Model Analysis")
        selected_model = st.selectbox("Select a model for detailed analysis:", results_df['Model'].tolist())
        
        if selected_model:
            col1, col2 = st.columns(2)
            
            y_pred = st.session_state.predictions_dict[selected_model]['y_pred']
            y_pred_proba = st.session_state.predictions_dict[selected_model]['y_pred_proba']
            y_test = st.session_state.y_test
            
            with col1:
                st.markdown("**Confusion Matrix**")
                fig_cm = plot_confusion_matrix(y_test, y_pred, selected_model)
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                st.markdown("**ROC Curve**")
                fig_roc = plot_roc_curve(y_test, y_pred_proba, selected_model)
                st.plotly_chart(fig_roc, use_container_width=True)
            
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, target_names=['No', 'Yes'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
        
    else:
        st.warning("No trained models found. Please train models first in the 'Train Models' page.")

elif page == "Make Predictions":
    st.header("Make Predictions on New Data")
    
    if st.session_state.models_trained:
        st.info("Upload a CSV file with test data (without target column) to make predictions")
        
        uploaded_test = st.file_uploader("Upload Test Data (CSV)", type=['csv'], key='pred_upload')
        
        if uploaded_test:
            test_df = pd.read_csv(uploaded_test, sep=';')
            st.write("Test Data Preview:")
            st.dataframe(test_df.head(), use_container_width=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                model_choice = st.selectbox("Select Model for Predictions:", 
                                           st.session_state.results_df['Model'].tolist())
            with col2:
                st.write("")
                st.write("")
                predict_button = st.button("Predict", type="primary")
            
            if predict_button:
                with st.spinner(f"Making predictions using {model_choice}..."):
                    try:
                        test_processed = test_df.copy()
                        
                        for col in test_processed.select_dtypes(include=['object']).columns:
                            if col in st.session_state.label_encoders:
                                le = st.session_state.label_encoders[col]
                                test_processed[col] = test_processed[col].map(
                                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                                )
                            else:
                                st.warning(f"Column '{col}' was not in training data. Encoding as -1.")
                                test_processed[col] = -1
                        
                        for col in st.session_state.feature_names:
                            if col not in test_processed.columns:
                                st.warning(f"Missing feature '{col}' in test data. Adding with value 0.")
                                test_processed[col] = 0
                        
                        test_processed = test_processed[st.session_state.feature_names]
                        
                        model = st.session_state.models_dict[model_choice]
                        
                        if st.session_state.model_scaling[model_choice]:
                            test_scaled = st.session_state.scaler.transform(test_processed)
                            predictions = model.predict(test_scaled)
                            if hasattr(model, 'predict_proba'):
                                predictions_proba = model.predict_proba(test_scaled)[:, 1]
                            else:
                                predictions_proba = predictions
                        else:
                            predictions = model.predict(test_processed)
                            if hasattr(model, 'predict_proba'):
                                predictions_proba = model.predict_proba(test_processed)[:, 1]
                            else:
                                predictions_proba = predictions
                        
                        results_df = test_df.copy()
                        results_df['Prediction'] = predictions
                        results_df['Prediction_Label'] = results_df['Prediction'].map({0: 'No', 1: 'Yes'})
                        results_df['Probability (Yes)'] = predictions_proba
                        results_df['Confidence'] = results_df['Probability (Yes)'].apply(
                            lambda x: f"{x*100:.2f}%" if x >= 0.5 else f"{(1-x)*100:.2f}%"
                        )
                        
                        st.success(f"Successfully made {len(predictions)} predictions using {model_choice}!")
                        
                        st.subheader("Prediction Summary")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Predictions", len(predictions))
                        col2.metric("Predicted 'Yes'", (predictions == 1).sum())
                        col3.metric("Predicted 'No'", (predictions == 0).sum())
                        
                        st.subheader("Predictions")
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            filter_option = st.radio("Show:", ["All", "Yes Only", "No Only"])
                        with col2:
                            num_rows = st.slider("Rows to display:", 10, min(500, len(results_df)), 50)
                        
                        if filter_option == "Yes Only":
                            display_df = results_df[results_df['Prediction'] == 1].head(num_rows)
                        elif filter_option == "No Only":
                            display_df = results_df[results_df['Prediction'] == 0].head(num_rows)
                        else:
                            display_df = results_df.head(num_rows)
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download All Predictions (CSV)",
                            data=csv,
                            file_name=f"{model_choice}_predictions.csv",
                            mime="text/csv",
                            key="download_predictions"
                        )
                        
                    except Exception as e:
                        st.error(f"Error making predictions: {e}")
                        st.info("Please ensure your data has the same features as the training data (excluding the target column 'y').")
    else:
        st.warning("Please train models first before making predictions.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>2025AB052026 - Srineveda R S</p>
</div>
""", unsafe_allow_html=True)
