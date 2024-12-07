import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import load_model

risk = pd.read_csv("reduced_asdf.csv")

for col in risk.select_dtypes(include='object').columns:
    risk[col] = risk[col].astype('category')

risk['status'] = risk['status'].astype('int8')

numerical_columns = risk.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = risk.select_dtypes(include=['category', 'int8']).columns

target_variable = 'status'

age_labels = {
    '25-34': 0,
    '35-44': 1,
    '45-54': 2,
    '55-64': 3,
    '65-74': 4,
    '>74': 5,
    '<25': 6
}

model = joblib.load('best_model.pkl')
scaler = joblib.load('scalers.pkl')

def predict_default(property_value, ltv, income, dtir, interest, age):
        input_data = np.array([[property_value, ltv, income, dtir, interest]])
        input_data = np.log1p(input_data)  
        input_scaled = scaler.transform(input_data)
        final_input = np.hstack([input_scaled, [[age]]])
        prediction = model.predict(final_input)
        return (prediction >= 0.3).astype(int)

st.title("Risk Analysis and Prediction")

st.sidebar.title("Navigation")
mode = st.sidebar.radio("Choose an option", 
                        ["Exploratory Data Analysis (EDA)", "Default Risk Prediction"])

if mode == "Exploratory Data Analysis (EDA)":
    st.sidebar.title("EDA Options")
    data_type = st.sidebar.radio("Choose a data type", 
                                 ["Numerical Distribution", "Categorical Distribution", 
                                  "Correlation Heatmap", "Bar Chart (Numerical Variables relationship to Status)", 
                                  "Stacked Bar Chart (Categorical Variables relationship to Status)"])
    if data_type == "Numerical Distribution":
        st.header("Histograms for Numerical Data")
        selected_num_col = st.selectbox("Select a Numerical Column", numerical_columns)
        fig = px.histogram(risk, x=selected_num_col, nbins=30, title=f'Histogram of {selected_num_col}')
        fig.update_layout(bargap=0.2)
        st.plotly_chart(fig)
    elif data_type == "Categorical Distribution":
        st.header("Count Plots for Categorical Data")
        selected_cat_col = st.selectbox("Select a Categorical Column", categorical_columns)
        fig = px.histogram(risk, y=selected_cat_col, title=f'Count Plot of {selected_cat_col}')
        fig.update_layout(bargap=0.2)
        st.plotly_chart(fig)
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

    elif data_type == "Bar Chart (Numerical Variables relationship to Status)":
        st.header("Bar Chart")
        selected_num_col = st.selectbox("Select a Numerical Column", numerical_columns)
        median_values = risk.groupby(target_variable)[selected_num_col].median().reset_index()
        fig = px.bar(median_values, x=target_variable, y=selected_num_col,
                     title=f'Median of {selected_num_col} by {target_variable}',
                     labels={selected_num_col: f'Median {selected_num_col}', target_variable: target_variable})
        st.plotly_chart(fig)

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

elif mode == "Default Risk Prediction":
    st.header("Predict Default Status")

    property_value = st.number_input("Property Value (부동산가치 억원)", min_value=0.0, value=5.0)
    property_value *= 100000
    ltv = st.number_input("Loan-to-Value Ratio (주택담보인정비율) = (부동산 대출금액 / 주택 담보가치) * 100", min_value=0.0, value=50.0)
    income = st.number_input("Monthly Income (월급 백만원)", min_value=0.0, value=3.0)
    income *= 1000
    dtir = st.number_input("Debt to Income Ratio (총부채상환비율) = (연간 대출이자 상환액 / 연봉) * 100 ", min_value=0.0, value=50.0)
    interest = st.number_input("Interest rate (대출금리)", min_value=0.0, value=5.0)
    interest -= 1.0
    age_category = st.selectbox("Age (나이)", list(age_labels.keys()))
    age_label = age_labels[age_category]
    
    if st.button("Predict Default Status"):
        prediction = predict_default(property_value, ltv, income, dtir, interest, age_label)
        if prediction == 1:
            status = "<span style='color: red; font-weight: bold;'>Default</span>"
        else:
            status = "<span style='color: green; font-weight: bold;'>Not Default</span>"
        st.markdown(f"Predicted Status: {status}", unsafe_allow_html=True)
