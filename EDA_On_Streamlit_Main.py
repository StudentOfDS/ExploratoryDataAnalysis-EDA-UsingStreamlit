# Importing necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedEDA:
    def __init__(self):
        # Set page configuration and styling
        st.set_page_config(layout="wide", page_icon="📊", page_title="Enhanced EDA Tool")
        self.set_custom_styles()  # Apply custom colors
        st.title("Enhanced Exploratory Data Analysis Tool")
        st.write("Upload your dataset to explore, clean, and visualize it interactively.")
        self.df = None  # Placeholder for the dataset

    def set_custom_styles(self):
        """
        Apply custom colors to the sidebar and background.
        """
        st.markdown(
            """
            <style>
            /* Main app background */
            .stApp {
                background-color: #191970 !important;  /* Midnight Blue background */
                color: white;  /* Change text color to white */
            }
            
            /* Sidebar styles */
            [data-testid="stSidebar"] {
                background-color: #6699CC !important;  /* Livid for the sidebar */
                color: white !important;  /* Ensure text in the sidebar is white */
            }
            
            .stSidebar .stSidebarHeader {
                background-color: #6699CC !important;  /* Livid for header */
                color: white !important;  /* Ensure header text is white */
            }

            .stButton>button {
                background-color: #6699CC !important;  /* Livid for buttons */
                color: white !important;
            }
            
            .stTextInput>div>input {
                background-color: #f0f0f0 !important;  /* Light gray for text inputs */
                color: black !important;
            }
            
            .stSelectbox>div>select {
                background-color: #f0f0f0 !important;  /* Light gray for select boxes */
                color: black !important;
            }

            /* Custom color for headers */
            h1, h2, h3, h4, h5, h6 {
                color: #6699CC !important;  /* Livid for headers */
            }
            </style>
            """, unsafe_allow_html=True
        )

    def load_dataset(self):
        """
        Load dataset from user upload.
        """
        st.sidebar.header("Upload Your Dataset")
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "txt", "excel", "json", "parquet"])
        if uploaded_file:
            file_format = uploaded_file.name.split('.')[-1].lower()
            try:
                if file_format == 'csv':
                    self.df = pl.read_csv(uploaded_file).to_pandas()
                elif file_format == 'txt':
                    self.df = pl.read_csv(uploaded_file, separator='\t').to_pandas()
                elif file_format == 'excel':
                    self.df = pd.read_excel(uploaded_file)
                elif file_format == 'json':
                    self.df = pd.read_json(uploaded_file)
                elif file_format == 'parquet':
                    self.df = pd.read_parquet(uploaded_file)
                else:
                    st.error("Unsupported file format. Please upload a CSV, Excel, TSV, JSON, or Parquet file.")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    def display_dataset_info(self):
        """
        Display basic information about the dataset.
        """
        if self.df is not None:
            st.subheader("Dataset Information")
            st.write(f"Number of Rows: {self.df.shape[0]}")
            st.write(f"Number of Columns: {self.df.shape[1]}")
            st.dataframe(self.df.head())
        else:
            st.warning("Please upload a dataset to view information.")

    def handle_missing_values(self):
        """
        Handle missing values and provide data cleaning options.
        """
        if self.df is not None:
            st.subheader("Missing Values and Data Cleaning")
            missing_values = self.df.isnull().sum()
            if missing_values.sum() == 0:
                st.write("No missing values found in the dataset.")
            else:
                st.write(missing_values)
                action = st.selectbox("Choose an action for missing values:", 
                                    ["None", "Drop rows", "Fill with mean", "Fill with median", "Fill with mode"])
                if action == "Drop rows":
                    self.df = self.df.dropna()
                    st.success("Rows with missing values dropped.")
                elif action == "Fill with mean":
                    self.df = self.df.fillna(self.df.mean())
                    st.success("Missing values filled with mean.")
                elif action == "Fill with median":
                    self.df = self.df.fillna(self.df.median())
                    st.success("Missing values filled with median.")
                elif action == "Fill with mode":
                    self.df = self.df.fillna(self.df.mode().iloc[0])
                    st.success("Missing values filled with mode.")

            # Advanced Data Cleaning Options
            st.subheader("Advanced Data Cleaning")
            cleaning_option = st.selectbox("Choose a data cleaning option:", 
                                         ["None", "Remove duplicates", "Standardize columns", "Rename columns", 
                                          "Drop columns", "Advanced Missing Value Handling"])
            if cleaning_option == "Remove duplicates":
                self.df = self.df.drop_duplicates()
                st.success("Duplicate rows removed.")
            elif cleaning_option == "Standardize columns":
                self.df.columns = self.df.columns.str.lower().str.replace('[^a-zA-Z0-9_]', '', regex=True)
                st.success("Column names standardized.")
            elif cleaning_option == "Rename columns":
                column_to_rename = st.selectbox("Select a column to rename:", self.df.columns)
                new_name = st.text_input("Enter the new column name:")
                if st.button("Rename Column"):
                    self.df.rename(columns={column_to_rename: new_name}, inplace=True)
                    st.success(f"Column '{column_to_rename}' renamed to '{new_name}'.")
            elif cleaning_option == "Drop columns":
                columns_to_drop = st.multiselect("Select columns to drop:", self.df.columns)
                if st.button("Drop Columns"):
                    self.df.drop(columns=columns_to_drop, inplace=True)
                    st.success(f"Dropped columns: {columns_to_drop}.")
            elif cleaning_option == "Advanced Missing Value Handling":
                st.write("### Advanced Data Cleaning")
                columns_to_drop = st.multiselect("Select columns to drop:", self.df.columns)
                selected_mv_columns = st.multiselect("Select columns to handle missing values:", self.df.columns)
                missing_value_strategy = st.selectbox("Missing value strategy:", ["drop", "fill", "keep"])
                fill_value = None
                
                if missing_value_strategy == "fill":
                    fill_value_input = st.text_input("Fill value (e.g., 0, mean, median, mode):")
                    if fill_value_input.lower() == 'mean':
                        if selected_mv_columns:
                            fill_value = self.df[selected_mv_columns].mean().to_dict()
                        else:
                            fill_value = self.df.mean().to_dict()
                    elif fill_value_input.lower() == 'median':
                        if selected_mv_columns:
                            fill_value = self.df[selected_mv_columns].median().to_dict()
                        else:
                            fill_value = self.df.median().to_dict()
                    elif fill_value_input.lower() == 'mode':
                        if selected_mv_columns:
                            fill_value = self.df[selected_mv_columns].mode().iloc[0].to_dict()
                        else:
                            fill_value = self.df.mode().iloc[0].to_dict()
                    else:
                        try:
                            fill_value = float(fill_value_input)
                        except ValueError:
                            fill_value = fill_value_input  # Treat as string or other
                
                if st.button("Apply Advanced Cleaning"):
                    try:
                        # Drop specified columns
                        if columns_to_drop:
                            self.df = self.df.drop(columns=columns_to_drop, errors='ignore')
                            st.success(f"Dropped columns: {columns_to_drop}")

                        # Handle missing values for selected columns
                        if selected_mv_columns:
                            if missing_value_strategy == "drop":
                                self.df = self.df.dropna(subset=selected_mv_columns)
                                st.success(f"Removed rows with missing values in columns: {selected_mv_columns}")
                            elif missing_value_strategy == "fill":
                                if isinstance(fill_value, dict):
                                    # Fill each selected column with its corresponding value from the dict
                                    self.df[selected_mv_columns] = self.df[selected_mv_columns].fillna(fill_value)
                                else:
                                    # Fill all selected columns with the scalar fill_value
                                    self.df[selected_mv_columns] = self.df[selected_mv_columns].fillna(fill_value)
                                st.success(f"Filled missing values in columns {selected_mv_columns} with: {fill_value}")
                            elif missing_value_strategy == "keep":
                                st.info("Missing values were not modified in selected columns")

                        # Normalize column names
                        self.df.columns = [col.strip().lower().replace(" ", "_") for col in self.df.columns]
                        st.success("Normalized column names")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload a dataset to clean.")

    def analyze_outliers(self):
        """
        Analyze outliers in numerical columns of the dataset.
        """
        if self.df is not None:
            st.subheader("Outlier Analysis")
            numerical_columns = self.df.select_dtypes(include=np.number).columns
            if len(numerical_columns) == 0:
                st.write("No numerical columns found in the dataset.")
            else:
                outliers = pd.DataFrame(index=numerical_columns, columns=['Count of Outliers'])
                for col in numerical_columns:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers.loc[col, 'Count of Outliers'] = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].shape[0]
                st.write(outliers)
        else:
            st.warning("Please upload a dataset to analyze outliers.")

    def visualize_data_distribution(self):
        """
        Visualize data distributions of numerical columns.
        """
        if self.df is not None:
            st.subheader("Data Distribution")
            numerical_columns = self.df.select_dtypes(include=np.number).columns
            if len(numerical_columns) == 0:
                st.write("No numerical columns found in the dataset.")
            else:
                selected_column = st.selectbox("Select a column for histogram:", numerical_columns)
                bins = st.slider("Number of bins:", 5, 100, 20)
                color = st.color_picker("Choose a color for the histogram:", "#6699CC")
                title = st.text_input("Enter a title for the histogram:", f"Histogram of {selected_column}")
                
                fig = px.histogram(self.df, x=selected_column, nbins=bins, title=title, color_discrete_sequence=[color])
                st.plotly_chart(fig)
        else:
            st.warning("Please upload a dataset to visualize.")

    def visualize_count_plots(self):
        """
        Visualize count plots of categorical columns in the dataset.
        """
        if self.df is not None:
            st.subheader("Count Plots of Categorical Columns")
            categorical_columns = self.df.select_dtypes(include='object').columns
            if len(categorical_columns) == 0:
                st.write("No categorical columns found in the dataset.")
            else:
                selected_columns = st.multiselect("Select columns for visualization:", categorical_columns)
                for col in selected_columns:
                    fig, ax = plt.subplots()
                    sns.countplot(data=self.df, x=col, ax=ax, color="#6699CC")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
        else:
            st.warning("Please upload a dataset to visualize.")

    def display_descriptive_analysis(self):
        """
        Display descriptive analysis of the dataset.
        """
        if self.df is not None:
            st.subheader("Descriptive Analysis")
            st.write(self.df.describe())
        else:
            st.warning("Please upload a dataset to view descriptive analysis.")

    def visualize_box_plots(self):
        """
        Visualize box plots of numerical columns in the dataset.
        """
        if self.df is not None:
            st.subheader("Box Plots")
            numerical_columns = self.df.select_dtypes(include=np.number).columns
            if len(numerical_columns) == 0:
                st.write("No numerical columns found in the dataset.")
            else:
                selected_columns = st.multiselect("Select columns for visualization:", numerical_columns)
                for col in selected_columns:
                    fig = px.box(self.df, y=col, title=f'Box Plot of {col}', color_discrete_sequence=["#6699CC"])
                    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig)
        else:
            st.warning("Please upload a dataset to visualize.")

    def visualize_scatter_plots(self):
        """
        Visualize scatter plots of numerical columns in the dataset.
        """
        if self.df is not None:
            st.subheader("Scatter Plots")
            numerical_columns = self.df.select_dtypes(include=np.number).columns
            if len(numerical_columns) < 2:
                st.write("Insufficient numerical columns for scatter plots.")
            else:
                x_col = st.selectbox("Select X-axis column:", numerical_columns)
                y_col = st.selectbox("Select Y-axis column:", numerical_columns)
                fig = px.scatter(self.df, x=x_col, y=y_col, title="Scatter Plot", color_discrete_sequence=["#6699CC"])
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig)
        else:
            st.warning("Please upload a dataset to visualize.")

    def visualize_pair_plots(self):
        """
        Visualize pair plots of numerical columns in the dataset.
        """
        if self.df is not None:
            st.subheader("Pair Plots")
            numerical_columns = self.df.select_dtypes(include=np.number).columns
            if len(numerical_columns) < 2:
                st.write("Insufficient numerical columns for pair plots.")
            else:
                selected_columns = st.multiselect("Select columns for pair plot:", numerical_columns)
                if len(selected_columns) > 0:
                    fig = px.scatter_matrix(self.df, dimensions=selected_columns, title="Pair Plot")
                    st.plotly_chart(fig)
        else:
            st.warning("Please upload a dataset to visualize.")

    def visualize_correlation_heatmap(self):
        """
        Visualize correlation heatmap of numerical columns in the dataset.
        """
        if self.df is not None:
            st.subheader("Correlation Heatmap")
            numerical_columns = self.df.select_dtypes(include=np.number).columns
            if len(numerical_columns) == 0:
                st.write("No numerical columns found in the dataset.")
            else:
                # Compute correlation matrix
                corr_matrix = self.df[numerical_columns].corr().round(2)
                
                # Create interactive heatmap with Plotly
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale="RdBu",
                    zmin=-1,
                    zmax=1,
                    labels=dict(x="Columns", y="Columns", color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    title="Correlation Heatmap"
                )
                
                fig.update_layout(
                    width=800,
                    height=800,
                    margin=dict(l=0, r=0, t=50, b=0)
                )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please upload a dataset to visualize.")

    def download_processed_data(self):
        """
        Allow users to download the processed dataset.
        """
        if self.df is not None:
            st.subheader("Download Processed Data")
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                self.df.to_excel(writer, index=False, sheet_name='Processed_Data')
            st.download_button(
                label="Download Excel file",
                data=buffer.getvalue(),
                file_name="processed_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("Please upload a dataset to download.")

    def run(self):
        """
        Run the entire EDA tool.
        """
        self.load_dataset()
        if self.df is not None:
            st.sidebar.subheader("Exploratory Data Analysis")
            options = st.sidebar.multiselect("Choose EDA tasks:", 
                                             ["Display Dataset Info", "Handle Missing Values", "Analyze Outliers", 
                                              "Visualize Data Distribution", "Visualize Count Plots", "Display Descriptive Analysis",
                                              "Visualize Box Plots", "Visualize Scatter Plots", "Visualize Pair Plots", "Visualize Correlation Heatmap",
                                              "Download Processed Data"])

            if "Display Dataset Info" in options:
                self.display_dataset_info()
            
            if "Handle Missing Values" in options:
                self.handle_missing_values()
            
            if "Analyze Outliers" in options:
                self.analyze_outliers()
            
            if "Visualize Data Distribution" in options:
                self.visualize_data_distribution()
            
            if "Visualize Count Plots" in options:
                self.visualize_count_plots()
            
            if "Display Descriptive Analysis" in options:
                self.display_descriptive_analysis()
            
            if "Visualize Box Plots" in options:
                self.visualize_box_plots()
            
            if "Visualize Scatter Plots" in options:
                self.visualize_scatter_plots()
            
            if "Visualize Pair Plots" in options:
                self.visualize_pair_plots()
            
            if "Visualize Correlation Heatmap" in options:
                self.visualize_correlation_heatmap()
            
            if "Download Processed Data" in options:
                self.download_processed_data()

# Run the EDA tool
if __name__ == "__main__":
    eda_tool = EnhancedEDA()
    eda_tool.run()