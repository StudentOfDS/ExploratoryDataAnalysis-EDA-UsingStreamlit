# Exploratory Data Analysis (EDA) Tool

This Streamlit app enables you to conduct exploratory data analysis on various datasets. It offers features for loading CSV and Excel files, visualizing data distributions, handling missing values, analyzing outliers, and generating descriptive statistics.

**Key Features:**

- **Data Loading:** Supports CSV and Excel file formats.
- **Exploratory Tasks:**
    - Display dataset information (number of rows, columns, sample data).
    - Handle missing values (display summary).
    - Analyze outliers (identify potential outliers in numerical columns).
    - Generate visualizations for:
        - Data distributions (histograms)
        - Count plots (for categorical columns)
        - Box plots (distribution and outliers for numerical columns)
        - Scatter plots (relationships between numerical columns)
        - Pair plots (matrix of scatter plots for numerical columns)
        - Correlation heatmap (correlations between numerical columns)
    - Display descriptive statistics (summary of numerical features).

**Installation:**

1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`

**Usage:**

1. Run the app: `streamlit run EDA_On_Streamlit_Main.py`
2. Upload a dataset file (CSV or Excel) using the file uploader in the sidebar.
3. Select desired EDA tasks from the sidebar to explore the dataset.

**Libraries Used:**

- Streamlit: For building the interactive web app.
- pandas: Data manipulation and analysis.
- Plotly Express: Interactive visualizations.
- Seaborn: Statistical visualizations.
- matplotlib: Visualization library.
- NumPy: Numerical computing.
