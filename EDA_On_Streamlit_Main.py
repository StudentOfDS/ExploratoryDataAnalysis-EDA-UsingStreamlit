# Importing necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Function to load dataset
def load_dataset(file_path, file_format):
    """
    Load dataset from file path.

    Parameters:
        file_path (str): Path to the dataset file.
        file_format (str): Format of the dataset file ('csv' or 'excel').

    Returns:
        DataFrame: Loaded dataset.
    """
    if file_format == 'csv':
        df = pd.read_csv(file_path)
    elif file_format == 'excel':
        df = pd.read_excel(file_path)
    return df

# Function to display dataset information
def display_dataset_info(df):
    """
    Display basic information about the dataset.

    Parameters:
        df (DataFrame): DataFrame containing the dataset.
    """
    st.subheader("Dataset Information")
    st.write("Number of Rows:", df.shape[0])
    st.write("Number of Columns:", df.shape[1])
    st.dataframe(df.head())

# Function to handle missing values
def handle_missing_values(df):
    """
    Handle missing values in the dataset.

    Parameters:
        df (DataFrame): DataFrame containing the dataset.
    """
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        st.write("No missing values found in the dataset.")
    else:
        st.write(missing_values)

# Function to analyze outliers
def analyze_outliers(df):
    """
    Analyze outliers in numerical columns of the dataset.

    Parameters:
        df (DataFrame): DataFrame containing the dataset.
    """
    st.subheader("Outlier Analysis")
    numerical_columns = df.select_dtypes(include=np.number).columns
    if len(numerical_columns) == 0:
        st.write("No numerical columns found in the dataset.")
    else:
        outliers = pd.DataFrame(index=numerical_columns, columns=['Count of Outliers'])
        for col in numerical_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers.loc[col, 'Count of Outliers'] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        st.write(outliers)

# Function to visualize data distributions
def visualize_data_distribution(df):
    """
    Visualize data distributions of numerical columns in the dataset.

    Parameters:
        df (DataFrame): DataFrame containing the dataset.
    """
    st.subheader("Data Distribution")
    numerical_columns = df.select_dtypes(include=np.number).columns
    if len(numerical_columns) == 0:
        st.write("No numerical columns found in the dataset.")
    else:
        selected_columns = st.multiselect("Select columns for visualization:", numerical_columns)
        for col in selected_columns:
            fig = px.histogram(df, x=col, title=f'Distribution of {col}')
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))  # Apply tight layout
            st.plotly_chart(fig)

# Function to visualize count plots of categorical columns
def visualize_count_plots(df):
    """
    Visualize count plots of categorical columns in the dataset.

    Parameters:
        df (DataFrame): DataFrame containing the dataset.
    """
    st.subheader("Count Plots of Categorical Columns")
    categorical_columns = df.select_dtypes(include='object').columns
    if len(categorical_columns) == 0:
        st.write("No categorical columns found in the dataset.")
    else:
        selected_columns = st.multiselect("Select columns for visualization:", categorical_columns)
        for col in selected_columns:
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=col, ax=ax)
            plt.xticks(rotation=45)
            plt.tight_layout()  # Apply tight layout
            st.pyplot(fig)

# Function to display descriptive analysis
def display_descriptive_analysis(df):
    """
    Display descriptive analysis of the dataset.

    Parameters:
        df (DataFrame): DataFrame containing the dataset.
    """
    st.subheader("Descriptive Analysis")
    st.write(df.describe())

# Function to visualize box plots
def visualize_box_plots(df):
    """
    Visualize box plots of numerical columns in the dataset.

    Parameters:
        df (DataFrame): DataFrame containing the dataset.
    """
    st.subheader("Box Plots")
    numerical_columns = df.select_dtypes(include=np.number).columns
    if len(numerical_columns) == 0:
        st.write("No numerical columns found in the dataset.")
    else:
        selected_columns = st.multiselect("Select columns for visualization:", numerical_columns)
        for col in selected_columns:
            fig = px.box(df, y=col, title=f'Box Plot of {col}')
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))  # Apply tight layout
            st.plotly_chart(fig)

# Function to visualize scatter plots
def visualize_scatter_plots(df):
    """
    Visualize scatter plots of numerical columns in the dataset.

    Parameters:
        df (DataFrame): DataFrame containing the dataset.
    """
    st.subheader("Scatter Plots")
    numerical_columns = df.select_dtypes(include=np.number).columns
    if len(numerical_columns) < 2:
        st.write("Insufficient numerical columns for scatter plots.")
    else:
        x_col = st.selectbox("Select X-axis column:", numerical_columns)
        y_col = st.selectbox("Select Y-axis column:", numerical_columns)
        fig = px.scatter(df, x=x_col, y=y_col, title="Scatter Plot")
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))  # Apply tight layout
        st.plotly_chart(fig)

# Function to visualize pair plots
def visualize_pair_plots(df):
    """
    Visualize pair plots of numerical columns in the dataset.

    Parameters:
        df (DataFrame): DataFrame containing the dataset.
    """
    st.subheader("Pair Plots")
    numerical_columns = df.select_dtypes(include=np.number).columns
    if len(numerical_columns) < 2:
        st.write("Insufficient numerical columns for pair plots.")
    else:
        fig = px.scatter_matrix(df, dimensions=numerical_columns, title="Pair Plot")
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))  # Apply tight layout
        st.plotly_chart(fig)

# Function to visualize correlation heatmap
def visualize_correlation_heatmap(df):
    """
    Visualize correlation heatmap of numerical columns in the dataset.

    Parameters:
        df (DataFrame): DataFrame containing the dataset.
    """
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    plt.tight_layout()  # Apply tight layout
    st.pyplot(fig)

# Main function to run the EDA tool
def main():
    # Page configuration
    st.set_page_config(layout="wide", page_icon="📊", page_title="EDA Tool")

    # Title and description
    st.title("Exploratory Data Analysis Tool")
    st.write("Upload your dataset to explore its characteristics and distributions.")

    # Sidebar for file upload and EDA options
    st.sidebar.title("Options")
    file_path = st.sidebar.file_uploader("Upload Dataset", type=['csv', 'xlsx'])
    if file_path:
        df = load_dataset(file_path, file_format=file_path.name.split('.')[-1])
        display_dataset_info(df)

        st.sidebar.subheader("Exploratory Data Analysis")
        options = st.sidebar.multiselect("Choose EDA tasks:", 
                                         ["Display Dataset Info", "Handle Missing Values", "Analyze Outliers", 
                                          "Visualize Data Distribution", "Visualize Count Plots", "Display Descriptive Analysis",
                                          "Visualize Box Plots", "Visualize Scatter Plots", "Visualize Pair Plots", "Visualize Correlation Heatmap"])

        if "Display Dataset Info" in options:
            display_dataset_info(df)
        
        if "Handle Missing Values" in options:
            handle_missing_values(df)
        
        if "Analyze Outliers" in options:
            analyze_outliers(df)
        
        if "Visualize Data Distribution" in options:
            visualize_data_distribution(df)
        
        if "Visualize Count Plots" in options:
            visualize_count_plots(df)
        
        if "Display Descriptive Analysis" in options:
            display_descriptive_analysis(df)
        
        if "Visualize Box Plots" in options:
            visualize_box_plots(df)
        
        if "Visualize Scatter Plots" in options:
            visualize_scatter_plots(df)
        
        if "Visualize Pair Plots" in options:
            visualize_pair_plots(df)
        
        if "Visualize Correlation Heatmap" in options:
            visualize_correlation_heatmap(df)

# Run the main function
if __name__ == "__main__":
    main()
