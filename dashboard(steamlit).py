import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Set page configuration
st.set_page_config(layout="wide", page_title="Student Performance Analysis")

# Load your dataset
@st.cache_data
def load_sample_data():
    data = pd.read_csv('student.csv')
    return data

# Title
st.title("Student Performance Analysis Dashboard")

# Load and preprocess data
data = load_sample_data()

# Convert binary columns
binary_cols = ['schoolsup', 'famsup', 'fatherd', 'activities', 'nursery', 'higher', 'internet', 'romantic']
for col in binary_cols:
    data[col] = data[col].map({'yes': 1, 'no': 0})

# Create target variable
data['good_grade'] = (data['G3'] >= 15).astype(int)

# Define features categories
categorical_features = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']
numerical_features = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 
                     'goout', 'Dalc', 'Walc', 'health', 'absences'] + binary_cols

# Train logistic regression model
X = data.drop(['G1', 'G2', 'G3', 'good_grade'], axis=1)
y = data['good_grade']

# Create and train logistic regression pipeline
@st.cache_resource
def create_model(X_train, y_train):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
pipeline = create_model(X_train, y_train)

# Compute predictions and probabilities
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Get feature importance for display
def get_feature_importance(pipeline, numerical_features, categorical_features, data):
    feature_names = numerical_features.copy()
    for cat_col in categorical_features:
        cat_values = data[cat_col].unique()
        for val in cat_values:
            feature_names.append(f"{cat_col}_{val}")
    
    coefficients = pipeline.named_steps['classifier'].coef_[0]
    
    # Make sure we don't have more feature names than coefficients
    feature_names = feature_names[:len(coefficients)]
    
    features_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'AbsCoefficient': abs(coefficients)
    }).sort_values('AbsCoefficient', ascending=False)
    
    return features_df

# Get feature importance
feature_imp = get_feature_importance(pipeline, numerical_features, categorical_features, data)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create main layout with two columns
col1, col2 = st.columns(2)

# Left column: Model Performance
with col1:
    st.header("Model Performance")
    
    # Create tabs for different visualizations
    model_tab = st.tabs(["ROC Curve", "Confusion Matrix", "Feature Importance"])
    
    # ROC Curve Tab
    with model_tab[0]:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='darkorange', width=2)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Guess',
            line=dict(color='navy', width=2, dash='dash')
        ))
        fig_roc.update_layout(
            xaxis={'title': 'False Positive Rate'},
            yaxis={'title': 'True Positive Rate'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0.1, 'y': 0.9},
            hovermode='closest'
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    
    # Confusion Matrix Tab
    with model_tab[1]:
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            colorscale='Blues',
            showscale=False,
            text=cm,
            texttemplate="%{text}",
            textfont={"size":20}
        ))
        fig_cm.update_layout(
            title='Confusion Matrix',
            margin={'l': 60, 'b': 60, 't': 50, 'r': 10}
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    # Feature Importance Tab
    with model_tab[2]:
        fig_imp = go.Figure(data=go.Bar(
            x=feature_imp['Coefficient'].head(10),
            y=feature_imp['Feature'].head(10),
            orientation='h',
            marker=dict(
                color=feature_imp['Coefficient'].head(10),
                colorscale='RdBu',
                colorbar=dict(title='Coefficient Value'),
                cmin=-max(abs(feature_imp['Coefficient'].head(10))),
                cmax=max(abs(feature_imp['Coefficient'].head(10)))
            )
        ))
        fig_imp.update_layout(
            title='Top 10 Feature Importance',
            xaxis={'title': 'Coefficient Value'},
            yaxis={'title': 'Feature', 'autorange': 'reversed'},
            margin={'l': 150, 'b': 50, 't': 50, 'r': 10}
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# Right column: Student Grade Analysis
with col2:
    st.header("Student Grade Analysis")
    
    # Feature dropdown for analysis
    selected_feature = st.selectbox(
        "Select Feature to Analyze:",
        options=['sex', 'studytime', 'failures', 'internet', 'romantic', 'Medu', 'Fedu']
    )
    
    # Display different plot types based on the feature type
    if selected_feature in numerical_features:
        fig_grade = px.scatter(
            data, 
            x=selected_feature, 
            y='G3', 
            color='good_grade',
            color_discrete_map={0: 'red', 1: 'green'},
            labels={'good_grade': 'Good Grade'},
            title=f'Final Grade vs {selected_feature}'
        )
    else:
        fig_grade = px.box(
            data, 
            x=selected_feature, 
            y='G3',
            color=selected_feature,
            title=f'Grade Distribution by {selected_feature}'
        )
    
    st.plotly_chart(fig_grade, use_container_width=True)
    
    # Probability prediction tool
    st.subheader("Grade Success Probability Prediction")
    
    col_study, col_fail = st.columns(2)
    
    with col_study:
        studytime = st.slider(
            "Study Time (hours per week):",
            min_value=1,
            max_value=4,
            value=2
        )
    
    with col_fail:
        failures = st.slider(
            "Number of Past Failures:",
            min_value=0,
            max_value=3,
            value=0
        )
    
    # Make prediction based on slider values
    sample_student = X_train.iloc[0].copy()
    sample_student['studytime'] = studytime
    sample_student['failures'] = failures
    
    sample_df = pd.DataFrame([sample_student], columns=X_train.columns)
    prob = pipeline.predict_proba(sample_df)[0, 1]
    
    st.write(f"### Probability of Good Grade: {prob:.2%}")
    
    # Probability gauge visualization
    col_prob_a, col_prob_b = st.columns([prob, 1-prob])
    col_prob_a.markdown(
        f"""<div style='background-color:green; height:20px; border-radius:10px 0px 0px 10px; 
        margin-right:-10px'></div>""", 
        unsafe_allow_html=True
    )
    col_prob_b.markdown(
        f"""<div style='background-color:red; height:20px; border-radius:0px 10px 10px 0px; 
        margin-left:-10px'></div>""", 
        unsafe_allow_html=True
    )

# Data overview section (full width)
st.header("Data Overview")
corr = data[numerical_features + ['G3']].corr()

fig_corr = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.columns,
    colorscale='RdBu_r',
    colorbar=dict(title='Correlation'),
    text=corr.values,
    texttemplate="%{text:.2f}",
    textfont={"size":10}
))

fig_corr.update_layout(
    title='Correlation between Numerical Features',
    height=600
)

st.plotly_chart(fig_corr, use_container_width=True)

# Show raw data option
if st.checkbox("Show raw data"):
    st.dataframe(data)