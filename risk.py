import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.tree import DecisionTreeClassifier

# Load the data
risk = pd.read_csv("risk.csv")

# Drop the 'Unnamed: 0' column
risk = risk.drop(columns=['Unnamed: 0'])

# Convert categorical 'object' columns to 'category' for proper selection and processing
for col in risk.select_dtypes(include='object').columns:
    risk[col] = risk[col].astype('category')

risk['status'] = risk['status'].astype('int8')

# Redefine numerical and categorical columns based on corrected types
numerical_columns = risk.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = risk.select_dtypes(include=['category', 'int8']).columns

target_variable = 'status'

decision = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
pca_1_component = pickle.load(open('pca.pkl', 'rb'))

# Age mapping for predictions
age_labels = {
    '25-34': 0,
    '35-44': 1,
    '45-54': 2,
    '55-64': 3,
    '65-74': 4,
    '>74': 5,
    '<25': 6
}

# Function to predict default status
def predict_default(property_value, ltv, income, upfront_charges, age_label):
    input_data = np.array([[property_value, ltv, income, upfront_charges]])
    input_scaled = scaler.transform(input_data)
    input_pca = pca_1_component.transform(input_scaled)
    final_input = np.hstack([input_pca, [[age_label]]])
    prediction = decision.predict(final_input)
    return prediction[0]

# Streamlit App Title
st.title("Risk Analysis and Prediction")

# Sidebar for selecting mode
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Choose an option", 
                        ["Exploratory Data Analysis (EDA)", "Default Risk Prediction"])

# EDA Section
if mode == "Exploratory Data Analysis (EDA)":
    # Sidebar for selecting data type
    st.sidebar.title("EDA Options")
    data_type = st.sidebar.radio("Choose a data type", 
                                 ["Numerical Distribution", "Categorical Distribution", 
                                  "Correlation Heatmap", "Bar Chart (Numerical Variables relationship to Status)", 
                                  "Stacked Bar Chart (Categorical Variables relationship to Status)"])

    # Display histogram for numerical data
    if data_type == "Numerical Distribution":
        st.header("Histograms for Numerical Data")
        selected_num_col = st.selectbox("Select a Numerical Column", numerical_columns)
        fig = px.histogram(risk, x=selected_num_col, nbins=30, title=f'Histogram of {selected_num_col}')
        fig.update_layout(bargap=0.2)
        st.plotly_chart(fig)

    # Display count plot for categorical data
    elif data_type == "Categorical Distribution":
        st.header("Count Plots for Categorical Data")
        selected_cat_col = st.selectbox("Select a Categorical Column", categorical_columns)
        fig = px.histogram(risk, y=selected_cat_col, title=f'Count Plot of {selected_cat_col}')
        fig.update_layout(bargap=0.2)
        st.plotly_chart(fig)

    # Display correlation heatmap for numerical data
    elif data_type == "Correlation Heatmap":
        st.header("Correlation Heatmap")
        imputer = SimpleImputer(strategy='mean')
        risk[numerical_columns] = imputer.fit_transform(risk[numerical_columns])
        numeric_df = risk.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numeric_df.corr()
        correlation_df = correlation_matrix.reset_index().melt(id_vars='index')
        correlation_df.columns = ['Variable1', 'Variable2', 'Correlation']
        heatmap = alt.Chart(correlation_df).mark_rect().encode(
            x='Variable1:O',
            y='Variable2:O',
            color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='viridis')),
            tooltip=['Variable1', 'Variable2', 'Correlation']
        ).properties(
            title='Correlation Heatmap',
            width=500,
            height=500
        )
        st.altair_chart(heatmap, use_container_width=True)

    # Display median bar chart for each status
    elif data_type == "Bar Chart (Numerical Variables relationship to Status)":
        st.header("Bar Chart")
        selected_num_col = st.sidebar.selectbox("Select a Numerical Column", numerical_columns)
        median_values = risk.groupby(target_variable)[selected_num_col].median().reset_index()
        fig = px.bar(median_values, x=target_variable, y=selected_num_col,
                     title=f'Median of {selected_num_col} by {target_variable}',
                     labels={selected_num_col: f'Median {selected_num_col}', target_variable: target_variable})
        st.plotly_chart(fig)

    # Display stacked bar chart for categorical data by target variable
    elif data_type == "Stacked Bar Chart (Categorical Variables relationship to Status)":
        st.header("Stacked Bar Chart")
        categorical_column = [col for col in categorical_columns if col != target_variable]
        selected_cat_col = st.selectbox("Select a Categorical Column for Stacked Bar Chart", categorical_column)
        stacked_data = risk.groupby([selected_cat_col, target_variable]).size().reset_index(name='count')
        stacked_data['percentage'] = stacked_data.groupby(selected_cat_col)['count'].transform(lambda x: x / x.sum() * 100)
        fig = px.bar(stacked_data, x=selected_cat_col, y='percentage', color=target_variable, 
                     title=f'Stacked Bar Chart of {target_variable} by {selected_cat_col}',
                     labels={'percentage': 'Percentage', target_variable: 'Status'},
                     text=stacked_data['percentage'].apply(lambda x: f'{x:.1f}%'))
        fig.update_layout(barmode='stack', xaxis_title=selected_cat_col, yaxis_title="Percentage")
        st.plotly_chart(fig)

# Prediction Section
elif mode == "Default Risk Prediction":
    st.header("Predict Default Status")

    # Input fields for prediction
    property_value = st.number_input("Property Value ($)", min_value=0, value=500000, step=1000)
    ltv = st.number_input("Loan-to-Value Ratio (LTV)", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
    income = st.number_input("Annual Income ($)", min_value=0, value=60000, step=1000)
    upfront_charges = st.number_input("Upfront Charges ($)", min_value=0, value=5000, step=100)
    age_category = st.selectbox("Age Category", list(age_labels.keys()))
    age_label = age_labels[age_category]

    # Predict button
    # Predict button
    if st.button("Predict Default Status"):
        prediction = predict_default(property_value, ltv, income, upfront_charges, age_label)
        status = "Default" if prediction == 1 else "Not Default"
        
        # Use Streamlit's markdown with color styling for the status
        if prediction == 1:
            st.markdown(f"<p style='color:red; font-size:20px;'>Predicted Status: {status}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color:green; font-size:20px;'>Predicted Status: {status}</p>", unsafe_allow_html=True)

