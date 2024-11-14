import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
import numpy as np

# Load data
risk = pd.read_csv("risk.csv")

# Drop the 'Unnamed: 0' column
risk = risk.drop(columns=['Unnamed: 0'])

# Convert categorical 'object' columns to 'category' for proper selection and processing
for col in risk.select_dtypes(include='object').columns:
    risk[col] = risk[col].astype('category')
    
risk['status'] = risk['status'].astype('int8')

# Redefine numerical and categorical columns based on corrected types
numerical_columns = risk.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = risk.select_dtypes(include=['category','int8']).columns

target_variable = 'status'

# Title
st.title("Exploratory Data Analysis (EDA)")

# Sidebar for selecting data type
st.sidebar.title("Select Data Type")
data_type = st.sidebar.radio("Choose a data type", 
                             ("Numerical Distribution", "Categorical Distribution", 
                              "Correlation Heatmap", "Bar Chart (Numerical Variables relationship to Status)", 
                              "Stacked Bar Chart (Categorical Variables relationship to Status)"))

# Display histogram for numerical data
if data_type == "Numerical Distribution":
    st.header("Histograms for Numerical Data")
    selected_num_col = st.selectbox("Select a Numerical Column", numerical_columns)
    
    # Plotly histogram
    fig = px.histogram(risk, x=selected_num_col, nbins=30, title=f'Histogram of {selected_num_col}')
    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig)

# Display count plot for categorical data
elif data_type == "Categorical Distribution":
    st.header("Count Plots for Categorical Data")
    selected_cat_col = st.selectbox("Select a Categorical Column", categorical_columns)
    
    # Plotly count plot
    fig = px.histogram(risk, y=selected_cat_col, title=f'Count Plot of {selected_cat_col}')
    fig.update_layout(bargap=0.2)
    st.plotly_chart(fig)

# Display correlation heatmap for numerical data
elif data_type == "Correlation Heatmap":
    st.header("Correlation Heatmap")
    # Calculating the correlation matrix
    numeric_df = risk.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_df.corr()

    # Converting the correlation matrix to a long format for Altair
    correlation_df = correlation_matrix.reset_index().melt(id_vars='index')
    correlation_df.columns = ['Variable1', 'Variable2', 'Correlation']

    # Creating an Altair heatmap
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

    # Display the heatmap in Streamlit
    st.altair_chart(heatmap, use_container_width=True)

# Display median bar chart for each status
elif data_type == "Bar Chart (Numerical Variables relationship to Status)":
    st.header("Bar Chart")

    # Sidebar option for selecting numerical column
    selected_num_col = st.sidebar.selectbox("Select a Numerical Column", numerical_columns)

    # Calculate median values for the selected column grouped by target variable
    median_values = risk.groupby(target_variable)[selected_num_col].mean().reset_index()

    # Plot bar chart of median values
    fig = px.bar(median_values, x=target_variable, y=selected_num_col,
                 title=f'Mean of {selected_num_col} by {target_variable}',
                 labels={selected_num_col: f'Median {selected_num_col}', target_variable: target_variable})
    
    st.plotly_chart(fig)

# Display stacked bar chart for categorical data by target variable
elif data_type == "Stacked Bar Chart (Categorical Variables relationship to Status)":
    st.header("Stacked Bar Chart")
    categorical_column = [col for col in categorical_columns if col != target_variable]
    # Select a categorical column for stacked bar chart
    selected_cat_col = st.selectbox("Select a Categorical Column for Stacked Bar Chart", categorical_column)

    # Calculate percentages for each category within the target variable
    stacked_data = risk.groupby([selected_cat_col, target_variable]).size().reset_index(name='count')
    stacked_data['percentage'] = stacked_data.groupby(selected_cat_col)['count'].transform(lambda x: x / x.sum() * 100)

    # Plotly stacked bar chart
    fig = px.bar(stacked_data, x=selected_cat_col, y='percentage', color=target_variable, 
                 title=f'Stacked Bar Chart of {target_variable} by {selected_cat_col}',
                 labels={'percentage': 'Percentage', target_variable: 'Status'},
                 text=stacked_data['percentage'].apply(lambda x: f'{x:.1f}%'))

    fig.update_layout(barmode='stack', xaxis_title=selected_cat_col, yaxis_title="Percentage")
    st.plotly_chart(fig)
