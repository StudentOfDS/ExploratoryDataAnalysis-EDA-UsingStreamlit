import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import logging
import re
import time
import tempfile
import psutil
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from scipy import stats
from pandas.api.types import is_numeric_dtype
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTENC
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde
from collections import Counter

# Constants
class Constants:
    LARGE_FILE_THRESHOLD = 100  # MB
    MAX_HISTORY_SIZE = 10
    SAMPLE_SIZE = 10000
    MEMORY_MONITOR_INTERVAL = 5  # seconds
    DEFAULT_THEME = {
        'primary': '#191970',
        'secondary': '#6699CC',
        'text': 'white'
    }
    IMBALANCE_METHODS_CLASSIFICATION = {
        'Random Over-sampling': 'Duplicate minority class samples',
        'Random Under-sampling': 'Remove majority class samples',
        'SMOTE': 'Synthetic Minority Over-sampling Technique',
        'ADASYN': 'Adaptive Synthetic Sampling',
        'SMOTEENN': 'Combination of SMOTE and Edited Nearest Neighbors'
    }
    IMBALANCE_METHODS_REGRESSION = {
        'SmoteR': 'SMOTE for Regression (interpolation)',
        'Random Oversampling': 'Duplicate rare value samples',
        'Gaussian Noise': 'Add Gaussian noise to rare values',
        'Target Binning': 'Bin target and balance bins'
    }
    REGRESSION_SAMPLING_STRATEGIES = {
        'uniform': 'Uniform distribution',
        'median': 'Median-based',
        'min': 'Match smallest bin'
    }

# Utility Functions
class Utils:
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_numerical_columns(df: pd.DataFrame) -> List[str]:
        return [] if df is None or df.empty else df.select_dtypes(include=np.number).columns.tolist()

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_categorical_columns(df: pd.DataFrame) -> List[str]:
        return [] if df is None or df.empty else df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_datetime_columns(df: pd.DataFrame) -> List[str]:
        return [] if df is None or df.empty else [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    @staticmethod
    def sanitize_column_name(name: str) -> str:
        clean = name.lower().strip()
        clean = re.sub(r'\s+', '_', clean)
        clean = re.sub(r'[^a-z0-9_]', '', clean)
        return clean if clean else 'column'
    
    @staticmethod
    def validate_input(value: str, pattern: str = r'^[a-zA-Z0-9_ ]+$') -> bool:
        return re.match(pattern, value) is not None
    
    @staticmethod
    def get_memory_usage() -> float:
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 2)
    
    @staticmethod
    def monitor_memory(interval: int = Constants.MEMORY_MONITOR_INTERVAL):
        mem_usage = Utils.get_memory_usage()
        if mem_usage > Constants.LARGE_FILE_THRESHOLD:
            st.sidebar.warning(f"High memory usage: {mem_usage:.2f} MB")
        st.session_state.last_memory_check = time.time()
    
    @staticmethod
    def should_sample(df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return False
        return df.memory_usage(deep=True).sum() > Constants.LARGE_FILE_THRESHOLD * 1024 * 1024
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def get_columns_by_type(df: pd.DataFrame, dtype: str) -> List[str]:
        if df is None or df.empty:
            return []
        if dtype == 'numeric':
            return df.select_dtypes(include=np.number).columns.tolist()
        elif dtype == 'categorical':
            return df.select_dtypes(include=['object', 'category']).columns.tolist()
        elif dtype == 'datetime':
            return [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        return []
    
    @staticmethod
    def is_classification_target(series: pd.Series) -> bool:
        """Determine if a target is suitable for classification imbalance handling"""
        return series.dtype in ['object', 'category'] or series.nunique() < 10

# File Handling
class FileHandler:
    def __init__(self, app):
        self.app = app
    
    def load_dataset(self):
        st.sidebar.header('1. Upload Dataset')
        uploaded = st.sidebar.file_uploader(
            'Choose file', type=['csv', 'txt', 'xlsx', 'json', 'parquet']
        )
        
        if not uploaded:
            return
        
        try:
            file_size = uploaded.size / (1024 ** 2)  # MB
            use_polars = file_size > Constants.LARGE_FILE_THRESHOLD
            
            with st.spinner('Reading file…'):
                buf = io.BytesIO(uploaded.read())
                ext = uploaded.name.split('.')[-1].lower()
                
                if use_polars:
                    if ext == 'csv':
                        df_pl = pl.read_csv(buf)
                    elif ext == 'txt':
                        df_pl = pl.read_csv(buf, separator='\t')
                    elif ext == 'parquet':
                        df_pl = pl.read_parquet(buf)
                    else:
                        st.error('Large file format must be CSV, TXT or Parquet')
                        return
                    
                    # Store both full and sampled data
                    st.session_state.original_df = df_pl
                    df = df_pl.head(Constants.SAMPLE_SIZE).to_pandas()
                else:
                    if ext == 'csv':
                        df = pd.read_csv(buf)
                    elif ext == 'txt':
                        df = pd.read_csv(buf, sep='\t')
                    elif ext == 'xlsx':
                        df = pd.read_excel(buf)
                    elif ext == 'json':
                        df = pd.read_json(buf)
                    elif ext == 'parquet':
                        df = pd.read_parquet(buf)
                    else:
                        st.error('Unsupported file format')
                        return
                
                if df.empty:
                    st.error('Uploaded file is empty')
                    return
                
                df.columns = [Utils.sanitize_column_name(col) for col in df.columns]
                
                st.session_state.df = df
                self.app._save_state(f"Loaded {uploaded.name}")
                st.success(f'Loaded {uploaded.name}: {df.shape[0]} rows, {df.shape[1]} cols')
                st.info(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")
                
                if use_polars:
                    st.info(f"Large file detected. Using sample of {Constants.SAMPLE_SIZE} rows for preview.")
        
        except Exception as e:
            st.error(f'Error loading file: {str(e)}')
            logging.error(f'Load error: {str(e)}')

# Data Cleaning
class DataCleaner:
    def __init__(self, app):
        self.app = app
    
    def handle_missing(self):
        df = st.session_state.df
        if df is None or df.empty:
            return
        
        try:
            st.subheader('3. Data Cleaning & Missing Values')
            
            mv = df.isnull().sum()
            if mv.sum() == 0:
                st.success('No missing values detected')
            else:
                st.write('Missing values per column:')
                st.dataframe(mv[mv > 0])
                
                self._advanced_imputation(df)
            
            st.markdown('---')
            st.markdown('### Advanced Cleaning Operations')
            self._advanced_cleaning(df)
            
            st.markdown('---')
            self._undo_functionality()
        
        except Exception as e:
            st.error(f'Cleaning error: {str(e)}')
            logging.error(f'Cleaning error: {str(e)}')
    
    def _advanced_imputation(self, df):
        st.markdown('### Advanced Imputation')
        
        cols = st.multiselect('Select columns to fill:', df.columns)
        strat = st.selectbox('Imputation strategy:', ['mean', 'median', 'mode', 'custom', 'group'])
        
        group_col = None
        if strat == 'group':
            group_col = st.selectbox('Group by column:', [c for c in df.columns if c not in cols])
        
        custom_val = None
        if strat == 'custom':
            custom_val = st.text_input('Enter custom value:')
            if not custom_val:
                st.warning('Please enter a custom value')
        
        if cols and st.button('Preview Fill Values'):
            fill_vals = self._calculate_fill_values(df, cols, strat, group_col, custom_val)
            st.write("Calculated fill values:")
            if strat == 'group' and group_col:
                for col in fill_vals:
                    st.write(f"{col}:")
                    st.dataframe(pd.DataFrame.from_dict(fill_vals[col], orient='index', columns=['Fill Value']))
            else:
                st.dataframe(pd.DataFrame(list(fill_vals.items()), columns=['Column', 'Fill Value']))
        
        if cols and st.button('Apply Imputation'):
            fill_vals = self._calculate_fill_values(df, cols, strat, group_col, custom_val)
            
            if fill_vals:
                if strat == 'group' and group_col:
                    for col in cols:
                        # Group-based imputation
                        df[col] = df.groupby(group_col)[col].transform(
                            lambda x: x.fillna(x.median() if is_numeric_dtype(x) else x.mode()[0])
                        )
                else:
                    # Column-based imputation
                    for col, val in fill_vals.items():
                        df[col] = df[col].fillna(val)
                
                st.session_state.df = df
                self.app._save_state(f"Imputation: {strat}")
                st.success(f'Imputation applied to {len(cols)} columns')
            else:
                st.warning('No valid imputation strategy selected')
    
    def _calculate_fill_values(self, df, cols, strat, group_col=None, custom_val=None):
        fill_vals = {}
        for c in cols:
            if strat == 'mean' and c in Utils.get_numerical_columns(df):
                fill_vals[c] = df[c].mean()
            elif strat == 'median' and c in Utils.get_numerical_columns(df):
                fill_vals[c] = df[c].median()
            elif strat == 'mode':
                mode_vals = df[c].mode()
                if not mode_vals.empty:
                    fill_vals[c] = mode_vals.iloc[0]
            elif strat == 'custom' and custom_val:
                try:
                    if c in Utils.get_numerical_columns(df):
                        fill_vals[c] = float(custom_val)
                    else:
                        fill_vals[c] = custom_val
                except ValueError:
                    fill_vals[c] = custom_val
            elif strat == 'group' and group_col:
                # Calculate group medians/modes
                if is_numeric_dtype(df[c]):
                    group_vals = df.groupby(group_col)[c].median().to_dict()
                else:
                    group_vals = df.groupby(group_col)[c].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan).to_dict()
                fill_vals[c] = group_vals
        return fill_vals
    
    def _advanced_cleaning(self, df):
        adv = st.selectbox(
            'Select operation:',
            ['None', 'Remove Duplicates', 'Standardize Columns', 
             'Rename Column', 'Drop Columns', 'Convert Data Type',
             'Missing Value Patterns']
        )
        
        if adv == 'Remove Duplicates' and st.button('Execute'):
            before = df.shape[0]
            df = df.drop_duplicates()
            st.session_state.df = df
            self.app._save_state("Removed duplicates")
            st.success(f'Removed {before - df.shape[0]} duplicate rows')
        
        elif adv == 'Standardize Columns' and st.button('Execute'):
            new_cols = [Utils.sanitize_column_name(name) for name in df.columns]
            df.columns = [f'col_{i}' if not col else col for i, col in enumerate(new_cols)]
            st.session_state.df = df
            self.app._save_state("Standardized columns")
            st.success('Column names standardized')
        
        elif adv == 'Rename Column':
            col = st.selectbox('Select column:', df.columns)
            new = st.text_input('New name:', value=col)
            if new and st.button('Rename'):
                if not Utils.validate_input(new):
                    st.error('Invalid column name! Use alphanumeric characters and underscores only.')
                elif new in df.columns:
                    st.error('Column name already exists!')
                else:
                    df = df.rename(columns={col: new})
                    st.session_state.df = df
                    self.app._save_state(f"Renamed {col} to {new}")
                    st.success(f'Renamed {col} to {new}')
        
        elif adv == 'Drop Columns':
            drops = st.multiselect('Select columns to drop:', df.columns)
            if drops and st.button('Execute'):
                df = df.drop(columns=drops)
                st.session_state.df = df
                self.app._save_state(f"Dropped {len(drops)} columns")
                st.success(f"Dropped columns: {', '.join(drops)}")
        
        elif adv == 'Convert Data Type':
            col = st.selectbox('Select column:', df.columns)
            current_type = str(df[col].dtype)
            new_type = st.selectbox('Select new type:', 
                                   ['string', 'integer', 'float', 'category', 'datetime'],
                                   index=['string', 'integer', 'float', 'category', 'datetime'].index(
                                       'integer' if 'int' in current_type else
                                       'float' if 'float' in current_type else
                                       'datetime' if 'datetime' in current_type else
                                       'category' if 'category' in current_type else 'string'
                                   ))
            
            if st.button('Convert'):
                try:
                    if new_type == 'string':
                        df[col] = df[col].astype(str)
                    elif new_type == 'integer':
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                    elif new_type == 'float':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    elif new_type == 'category':
                        df[col] = df[col].astype('category')
                    elif new_type == 'datetime':
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    
                    st.session_state.df = df
                    self.app._save_state(f"Converted {col} to {new_type}")
                    st.success(f'Converted {col} to {new_type}')
                except Exception as e:
                    st.error(f'Conversion error: {str(e)}')
        
        elif adv == 'Missing Value Patterns' and st.button('Execute'):
            st.subheader('Missing Value Patterns')
            
            st.markdown('**Missing Value Matrix**')
            fig, ax = plt.subplots(figsize=(10, 6))
            msno.matrix(df, ax=ax)
            st.pyplot(fig)
            
            st.markdown('**Nullity Correlation**')
            fig, ax = plt.subplots(figsize=(10, 6))
            msno.heatmap(df, ax=ax)
            st.pyplot(fig)
            
            st.markdown('**Nullity Dendrogram**')
            fig, ax = plt.subplots(figsize=(10, 6))
            msno.dendrogram(df, ax=ax)
            st.pyplot(fig)
    
    def _undo_functionality(self):
        if st.button('Undo Last Action') and len(st.session_state.history) > 1:
            st.session_state.history.pop()
            st.session_state.df = st.session_state.history[-1]['df']
            st.success(f"Undo: {st.session_state.history[-1]['action']}")

# Visualization
class Visualizer:
    def __init__(self, app):
        self.app = app
        self.filtered_df = None
    
    def visualize(self):
        df = st.session_state.df
        if df is None or df.empty:
            return
        
        try:
            if Utils.should_sample(df):
                sample_size = st.slider('Sample size for visualization:', 
                                       100, min(5000, len(df)), 
                                       min(1000, len(df)//10))
                df = df.sample(min(sample_size, len(df)))
                st.info(f"Using sample of {len(df)} rows for visualization")
            
            self.filtered_df = self._apply_filters(df)
            
            st.subheader('5. Data Visualization')
            opts = [
                'Histogram', 'Count Plot', 'Box Plot',
                'Scatter Plot', 'Pair Plot', 'Correlation Heatmap',
                'Time Series', 'Pie Chart', 'Advanced Charts',
                'Univariate Analysis', 'Full Correlation Analysis'
            ]
            choice = st.selectbox('Select visualization type:', opts)
            
            if choice == 'Histogram':
                self._render_histogram()
            elif choice == 'Count Plot':
                self._render_count_plot()
            elif choice == 'Box Plot':
                self._render_box_plot()
            elif choice == 'Scatter Plot':
                self._render_scatter_plot()
            elif choice == 'Pair Plot':
                self._render_pair_plot()
            elif choice == 'Correlation Heatmap':
                self._render_correlation_heatmap()
            elif choice == 'Time Series':
                self._render_time_series()
            elif choice == 'Pie Chart':
                self._render_pie_chart()
            elif choice == 'Advanced Charts':
                self._render_advanced_charts()
            elif choice == 'Univariate Analysis':
                self._render_univariate_analysis()
            elif choice == 'Full Correlation Analysis':
                self._render_full_correlation_analysis()
        
        except Exception as e:
            st.error(f'Visualization error: {str(e)}')
            logging.error(f'Visualization error: {str(e)}')
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        st.sidebar.subheader('Data Filters')
        
        date_cols = Utils.get_datetime_columns(df)
        if date_cols:
            date_col = st.sidebar.selectbox('Filter by date column:', ['None'] + date_cols)
            if date_col != 'None':
                min_date = df[date_col].min().to_pydatetime()
                max_date = df[date_col].max().to_pydatetime()
                date_range = st.sidebar.slider(
                    'Select date range:',
                    min_value=min_date,
                    max_value=max_date,
                    value=(min_date, max_date)
                )
                df = df[(df[date_col] >= date_range[0]) & (df[date_col] <= date_range[1])]
        
        num_cols = Utils.get_numerical_columns(df)
        for col in num_cols:
            if st.sidebar.checkbox(f'Filter {col}', key=f'num_filter_{col}'):
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                range_vals = st.sidebar.slider(
                    f'Range for {col}:',
                    min_val, max_val, (min_val, max_val)
                )
                df = df[(df[col] >= range_vals[0]) & (df[col] <= range_vals[1])]
        
        cat_cols = Utils.get_categorical_columns(df)
        for col in cat_cols:
            if st.sidebar.checkbox(f'Filter {col}', key=f'cat_filter_{col}'):
                options = st.sidebar.multiselect(
                    f'Select values for {col}:',
                    options=df[col].unique().tolist(),
                    default=df[col].unique().tolist()
                )
                df = df[df[col].isin(options)]
        
        return df
    
    def _render_histogram(self):
        col = st.selectbox('Numerical column:', Utils.get_numerical_columns(self.filtered_df))
        bins = st.slider('Number of bins:', 5, 100, 20)
        color_col = st.selectbox('Color by (optional):', ['None'] + Utils.get_categorical_columns(self.filtered_df))
        color = None if color_col == 'None' else color_col
        
        with st.spinner('Generating histogram…'):
            fig = px.histogram(
                self.filtered_df, x=col, nbins=bins, color=color,
                title=f'Distribution of {col}',
                marginal='box'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_count_plot(self):
        col = st.selectbox('Categorical column:', Utils.get_categorical_columns(self.filtered_df))
        top_n = st.slider('Show top N categories:', 5, 50, 10)
        
        with st.spinner('Generating count plot…'):
            counts = self.filtered_df[col].value_counts().reset_index().head(top_n)
            counts.columns = [col, 'Count']
            fig = px.bar(
                counts, x=col, y='Count',
                title=f'Top {top_n} Categories in {col}',
                color=col
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_box_plot(self):
        cols = st.multiselect('Numerical columns:', Utils.get_numerical_columns(self.filtered_df))
        if cols:
            melted = self.filtered_df[cols].melt(var_name='Variable', value_name='Value')
            with st.spinner('Generating box plot…'):
                fig = px.box(
                    melted, x='Variable', y='Value',
                    title='Distribution of Numerical Columns',
                    color='Variable'
                )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_scatter_plot(self):
        nums = Utils.get_numerical_columns(self.filtered_df)
        if len(nums) >= 2:
            x = st.selectbox('X-axis:', nums)
            y = st.selectbox('Y-axis:', nums)
            color_col = st.selectbox('Color by:', ['None'] + Utils.get_categorical_columns(self.filtered_df))
            size_col = st.selectbox('Size by (optional):', ['None'] + Utils.get_numerical_columns(self.filtered_df))
            hover_col = st.selectbox('Hover info:', ['None'] + list(self.filtered_df.columns))
            
            color = None if color_col == 'None' else color_col
            size = None if size_col == 'None' else size_col
            hover_data = [hover_col] if hover_col != 'None' else None
            
            with st.spinner('Generating scatter plot…'):
                fig = px.scatter(
                    self.filtered_df, x=x, y=y, color=color, size=size,
                    hover_data=hover_data, title=f'{x} vs {y}',
                    trendline='ols'
                )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_pair_plot(self):
        nums = Utils.get_numerical_columns(self.filtered_df)
        if len(nums) >= 2:
            sel = st.multiselect('Select columns:', nums)
            
            if len(sel) > 5:
                st.warning("Pair plots limited to 5 columns for performance")
                sel = sel[:5]
            
            if sel:
                sample_size = st.slider('Sample size:', 100, min(2000, len(self.filtered_df)), 500)
                with st.spinner('Rendering pair plot…'):
                    fig = px.scatter_matrix(
                        self.filtered_df[sel].sample(sample_size),
                        dimensions=sel,
                        title='Pairwise Relationships',
                        color=sel[0] if sel else None
                    )
                st.plotly_chart(fg, use_container_width=True)
    
    def _render_correlation_heatmap(self):
        nums = Utils.get_numerical_columns(self.filtered_df)
        if len(nums) >= 2:
            with st.spinner('Generating heatmap…'):
                corr = self.filtered_df[nums].corr().round(2)
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale='RdBu',
                    title='Correlation Matrix',
                    zmin=-1, zmax=1
                )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_time_series(self):
        date_cols = Utils.get_datetime_columns(self.filtered_df)
        if not date_cols:
            st.warning('No datetime columns found')
        else:
            date_col = st.selectbox('Date column:', date_cols)
            value_col = st.selectbox('Value column:', Utils.get_numerical_columns(self.filtered_df))
            agg_func = st.selectbox('Aggregation:', ['sum', 'mean', 'count'])
            
            with st.spinner('Generating time series…'):
                # Handle datetime index
                if not pd.api.types.is_datetime64_any_dtype(self.filtered_df[date_col]):
                    self.filtered_df[date_col] = pd.to_datetime(self.filtered_df[date_col])
                
                ts_df = self.filtered_df.set_index(date_col)[value_col].resample('D').agg(agg_func).reset_index()
                fig = px.line(
                    ts_df, x=date_col, y=value_col,
                    title=f'{agg_func.capitalize()} of {value_col} over Time'
                )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_pie_chart(self):
        cat_col = st.selectbox('Categorical column:', Utils.get_categorical_columns(self.filtered_df))
        top_n = st.slider('Show top N:', 3, 20, 5)
        
        with st.spinner('Generating pie chart…'):
            counts = self.filtered_df[cat_col].value_counts().reset_index().head(top_n)
            counts.columns = [cat_col, 'Count']
            fig = px.pie(
                counts, names=cat_col, values='Count',
                title=f'Distribution of {cat_col} (Top {top_n})'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_advanced_charts(self):
        chart_type = st.selectbox('Select chart type:', ['Violin Plot', 'Density Contour', 'Facet Grid'])
        
        if chart_type == 'Violin Plot':
            num_col = st.selectbox('Numerical column:', Utils.get_numerical_columns(self.filtered_df))
            cat_col = st.selectbox('Categorical column:', Utils.get_categorical_columns(self.filtered_df))
            
            with st.spinner('Generating violin plot…'):
                fig = px.violin(
                    self.filtered_df, x=cat_col, y=num_col,
                    box=True, points='all',
                    title=f'Distribution of {num_col} by {cat_col}'
                )
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == 'Density Contour':
            x_col = st.selectbox('X-axis:', Utils.get_numerical_columns(self.filtered_df))
            y_col = st.selectbox('Y-axis:', Utils.get_numerical_columns(self.filtered_df))
            
            with st.spinner('Generating density contour…'):
                fig = px.density_contour(
                    self.filtered_df, x=x_col, y=y_col,
                    marginal_x='histogram', marginal_y='histogram',
                    title=f'Density of {x_col} vs {y_col}'
                )
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == 'Facet Grid':
            num_col = st.selectbox('Numerical column:', Utils.get_numerical_columns(self.filtered_df))
            row_col = st.selectbox('Row facet:', ['None'] + Utils.get_categorical_columns(self.filtered_df))
            col_col = st.selectbox('Column facet:', ['None'] + Utils.get_categorical_columns(self.filtered_df))
            
            if row_col == 'None' and col_col == 'None':
                st.warning('Select at least one facet dimension')
                return
            
            with st.spinner('Generating facet grid…'):
                # Use Plotly instead of Seaborn for consistency
                fig = px.histogram(
                    self.filtered_df,
                    x=num_col,
                    facet_row=row_col if row_col != 'None' else None,
                    facet_col=col_col if col_col != 'None' else None,
                    title=f'Distribution of {num_col}'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_univariate_analysis(self):
        st.subheader('Univariate Analysis')
        df = self.filtered_df
        
        num_cols = Utils.get_numerical_columns(df)
        if num_cols:
            st.markdown('### Numerical Distributions')
            cols_per_row = 2
            cols = st.columns(cols_per_row)
            
            for i, col in enumerate(num_cols):
                with cols[i % cols_per_row]:
                    st.markdown(f"**{col}**")
                    
                    fig = px.histogram(
                        df, x=col, nbins=50, 
                        marginal='rug', 
                        title=f'Distribution of {col}'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("**Normality Check (QQ Plot)**")
                    qq_fig = self._create_qq_plot(df, col)
                    st.plotly_chart(qq_fig, use_container_width=True)
        
        cat_cols = Utils.get_categorical_columns(df)
        if cat_cols:
            st.markdown('### Categorical Distributions')
            cols_per_row = 2
            cols = st.columns(cols_per_row)
            
            for i, col in enumerate(cat_cols):
                with cols[i % cols_per_row]:
                    st.markdown(f"**{col}**")
                    
                    counts = df[col].value_counts().reset_index().head(20)
                    counts.columns = [col, 'Count']
                    
                    fig = px.bar(
                        counts, x=col, y='Count',
                        title=f'Top Categories in {col}',
                        color=col
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        if not num_cols and not cat_cols:
            st.warning("No numerical or categorical columns found for analysis")
    
    def _create_qq_plot(self, df, col):
        data = df[col].dropna()
        
        (quantiles, values), (slope, intercept, r) = stats.probplot(data, dist="norm")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=quantiles, y=values, 
            mode='markers', name='Actual'
        ))
        fig.add_trace(go.Scatter(
            x=quantiles, y=slope * quantiles + intercept,
            mode='lines', name='Theoretical'
        ))
        
        fig.update_layout(
            title=f'QQ Plot for {col}',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Ordered Values',
            showlegend=True
        )
        return fig
    
    def _render_full_correlation_analysis(self):
        st.subheader('Full Correlation Analysis')
        df = self.filtered_df
        num_cols = Utils.get_numerical_columns(df)
        
        if len(num_cols) < 2:
            st.warning("Need at least 2 numerical columns for correlation analysis")
            return
        
        method = st.selectbox(
            'Correlation Method:',
            ['pearson', 'spearman', 'kendall']
        )
        
        corr = df[num_cols].corr(method=method).round(2)
        
        fig = px.imshow(
            corr, 
            text_auto=True,
            color_continuous_scale='RdBu',
            title=f'Correlation Matrix ({method.capitalize()})',
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Detailed Correlation Matrix")
        st.dataframe(corr.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1))
        
        st.markdown("### Strongest Correlations")
        self._show_strong_correlations(corr)
    
    def _show_strong_correlations(self, corr_matrix):
        corr_stack = corr_matrix.stack().reset_index()
        corr_stack.columns = ['Var1', 'Var2', 'Correlation']
        
        corr_stack = corr_stack[corr_stack['Var1'] != corr_stack['Var2']]
        corr_stack['Pairs'] = corr_stack.apply(
            lambda x: tuple(sorted([x['Var1'], x['Var2']])), axis=1
        )
        corr_stack = corr_stack.drop_duplicates('Pairs').drop(columns='Pairs')
        
        threshold = st.slider('Correlation Threshold:', 0.5, 0.95, 0.7, 0.05)
        strong_corrs = corr_stack[
            (abs(corr_stack['Correlation']) >= threshold) &
            (corr_stack['Correlation'] < 1)
        ]
        
        if strong_corrs.empty:
            st.info(f"No correlations above {threshold} found")
            return
        
        strong_corrs['AbsCorr'] = abs(strong_corrs['Correlation'])
        strong_corrs = strong_corrs.sort_values('AbsCorr', ascending=False)
        
        st.dataframe(strong_corrs.drop(columns='AbsCorr'))

# Data Export
class Exporter:
    def __init__(self, app):
        self.app = app
    
    def download_data(self):
        df = st.session_state.df
        if df is None or df.empty:
            st.warning('No data to export')
            return
        
        try:
            st.subheader('6. Export Processed Data')
            
            fmt = st.selectbox('Select export format:', ['CSV', 'Excel', 'Parquet', 'JSON'])
            buf = io.BytesIO()
            mime = ''
            file_ext = ''
            
            if fmt == 'CSV':
                buf.write(df.to_csv(index=False).encode())
                mime = 'text/csv'
                file_ext = 'csv'
            elif fmt == 'JSON':
                buf.write(df.to_json(orient='records').encode())
                mime = 'application/json'
                file_ext = 'json'
            elif fmt == 'Parquet':
                # Use Polars for large datasets if available
                if 'original_df' in st.session_state and isinstance(st.session_state.original_df, pl.DataFrame):
                    st.session_state.original_df.write_parquet(buf)
                else:
                    df.to_parquet(buf)
                mime = 'application/octet-stream'
                file_ext = 'parquet'
            else:  # Excel
                try:
                    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='ProcessedData')
                    mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    file_ext = 'xlsx'
                except ImportError:
                    st.error('Excel export requires openpyxl. Install with: pip install openpyxl')
                    return
            
            buf.seek(0)
            size_mb = len(buf.getvalue()) / (1024 ** 2)
            
            st.info(f"File size: {size_mb:.2f} MB")
            if size_mb > Constants.LARGE_FILE_THRESHOLD:
                st.warning('Large file warning: Download may be slow')
            
            st.download_button(
                '⬇️ Download Processed Data',
                data=buf,
                file_name=f'processed_data.{file_ext}',
                mime=mime
            )
        except Exception as e:
            st.error(f'Export error: {str(e)}')
            logging.error(f'Export error: {str(e)}')

# Data Imbalance Handling
class ImbalanceHandler:
    def __init__(self, app):
        self.app = app
    
    def handle_imbalance(self):
        df = st.session_state.df
        if df is None or df.empty:
            return
        
        try:
            st.subheader('7. Handle Data Imbalance')
            st.info("""
            **Data imbalance** affects both classification (categorical targets) and regression (continuous targets). 
            This tool helps address both types of imbalance.
            """)
            
            # Select target column
            target = st.selectbox('Select target column:', df.columns)
            
            # Determine problem type
            problem_type = "classification" if Utils.is_classification_target(df[target]) else "regression"
            st.info(f"Detected problem type: **{problem_type.upper()}**")
            
            if problem_type == "classification":
                self._handle_classification_imbalance(df, target)
            else:
                self._handle_regression_imbalance(df, target)
        
        except Exception as e:
            st.error(f'Imbalance handling error: {str(e)}')
            logging.error(f'Imbalance error: {str(e)}')
    
    def _handle_classification_imbalance(self, df, target):
        """Handle imbalance for classification problems"""
        # Show current class distribution
        st.markdown("### Current Class Distribution")
        self._plot_class_distribution(df, target)
        
        # Check if imbalance exists
        class_counts = df[target].value_counts()
        min_count = class_counts.min()
        max_count = class_counts.max()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio < 2:
            st.success("No significant class imbalance detected (ratio < 2:1)")
            return
        else:
            st.warning(f"Class imbalance detected! Ratio: {imbalance_ratio:.1f}:1")
        
        # Select imbalance handling method
        method = st.selectbox('Select imbalance handling method:', 
                            list(Constants.IMBALANCE_METHODS_CLASSIFICATION.keys()),
                            format_func=lambda x: f"{x} - {Constants.IMBALANCE_METHODS_CLASSIFICATION[x]}")
        
        # Additional parameters
        sampling_strategy = st.slider('Sampling ratio (minority/majority):', 
                                    0.1, 1.0, 0.5, 0.1,
                                    help="Target ratio between minority and majority classes")
        
        # Handle categorical features
        cat_cols = Utils.get_categorical_columns(df.drop(columns=[target]))
        use_cat_features = st.checkbox('Handle categorical features', 
                                      value=len(cat_cols) > 0,
                                      help="Apply specialized handling for categorical columns")
        
        if st.button('Preview Balanced Distribution'):
            df_resampled = self._apply_classification_method(df, target, method, sampling_strategy, use_cat_features)
            st.markdown("### Balanced Class Distribution")
            self._plot_class_distribution(df_resampled, target)
            st.success(f"Resampled dataset: {df_resampled.shape[0]} rows")
            
            if st.button('Apply to Dataset'):
                st.session_state.df = df_resampled
                self.app._save_state(f"Classification imbalance: {method}")
                st.success("Dataset updated with balanced classes!")
    
    def _handle_regression_imbalance(self, df, target):
        """Handle imbalance for regression problems (continuous targets)"""
        # Show target distribution
        st.markdown("### Target Value Distribution")
        self._plot_target_distribution(df, target)
        
        # Check for imbalance
        imbalance_score = self._calculate_regression_imbalance(df[target])
        st.warning(f"Imbalance score: {imbalance_score:.2f} (higher values indicate greater imbalance)")
        
        # Select imbalance handling method
        method = st.selectbox('Select imbalance handling method:', 
                            list(Constants.IMBALANCE_METHODS_REGRESSION.keys()),
                            format_func=lambda x: f"{x} - {Constants.IMBALANCE_METHODS_REGRESSION[x]}")
        
        # Method-specific parameters
        if method == 'SmoteR':
            k_neighbors = st.slider('Number of neighbors for interpolation:', 2, 20, 5)
            noise_level = st.slider('Noise level:', 0.0, 1.0, 0.1, 0.05)
        elif method == 'Target Binning':
            n_bins = st.slider('Number of bins:', 5, 50, 10)
            bin_strategy = st.selectbox('Bin balancing strategy:', 
                                      list(Constants.REGRESSION_SAMPLING_STRATEGIES.keys()),
                                      format_func=lambda x: Constants.REGRESSION_SAMPLING_STRATEGIES[x])
        else:  # Gaussian Noise or Random Oversampling
            noise_level = st.slider('Noise level (std multiplier):', 0.0, 1.0, 0.1, 0.05)
        
        # Relevance function for rare regions
        st.markdown("### Define Rare Regions")
        relevance_func = st.selectbox('Relevance function:', 
                                    ['auto', 'manual'], 
                                    help="Define which target values are rare and need more attention")
        
        if relevance_func == 'manual':
            min_val = df[target].min()
            max_val = df[target].max()
            rare_min = st.number_input('Rare region start:', min_val, max_val, min_val)
            rare_max = st.number_input('Rare region end:', min_val, max_val, max_val)
            relevance_factor = st.slider('Relevance factor:', 1.0, 10.0, 2.0, 0.5)
        else:
            rare_min, rare_max, relevance_factor = None, None, 2.0
        
        if st.button('Preview Balanced Distribution'):
            df_resampled = self._apply_regression_method(df, target, method, 
                                                       rare_min, rare_max, relevance_factor,
                                                       k_neighbors if method == 'SmoteR' else None,
                                                       noise_level,
                                                       n_bins if method == 'Target Binning' else None,
                                                       bin_strategy if method == 'Target Binning' else None)
            st.markdown("### Balanced Target Distribution")
            self._plot_target_distribution(df_resampled, target)
            st.success(f"Resampled dataset: {df_resampled.shape[0]} rows")
            
            if st.button('Apply to Dataset'):
                st.session_state.df = df_resampled
                self.app._save_state(f"Regression imbalance: {method}")
                st.success("Dataset updated with balanced distribution!")
    
    def _plot_class_distribution(self, df, target_col):
        counts = df[target_col].value_counts().reset_index()
        counts.columns = ['Class', 'Count']
        
        fig = px.bar(
            counts, x='Class', y='Count', color='Class',
            title=f'Class Distribution: {target_col}',
            text='Count'
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show exact counts
        st.dataframe(counts)
    
    def _plot_target_distribution(self, df, target_col):
        fig = px.histogram(
            df, x=target_col, nbins=50,
            title=f'Distribution of {target_col}',
            marginal='box'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add density plot
        fig = px.density_contour(
            df, x=target_col, 
            title=f'Density of {target_col}'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _calculate_regression_imbalance(self, target_series):
        """Calculate imbalance score for continuous target"""
        # Use coefficient of variation of bin counts
        hist, bin_edges = np.histogram(target_series, bins=10)
        bin_counts = hist[hist > 0]  # Ignore empty bins
        if len(bin_counts) < 2:
            return 0
        return np.std(bin_counts) / np.mean(bin_counts)
    
    def _apply_classification_method(self, df, target, method, sampling_strategy, use_cat_features):
        X = df.drop(columns=[target])
        y = df[target]
        
        # Encode categorical features if needed
        cat_features = None
        if use_cat_features:
            cat_cols = Utils.get_categorical_columns(X)
            if cat_cols:
                for col in cat_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                cat_features = [i for i, col in enumerate(X.columns) if col in cat_cols]
        
        # Apply selected resampling method
        if method == 'Random Over-sampling':
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        elif method == 'Random Under-sampling':
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        elif method == 'SMOTE':
            if cat_features:
                sampler = SMOTENC(categorical_features=cat_features, 
                                 sampling_strategy=sampling_strategy, 
                                 random_state=42)
            else:
                sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        elif method == 'ADASYN':
            sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
        elif method == 'SMOTEENN':
            if cat_features:
                smote = SMOTENC(categorical_features=cat_features, 
                               sampling_strategy=sampling_strategy, 
                               random_state=42)
                sampler = SMOTEENN(smote=smote, random_state=42)
            else:
                sampler = SMOTEENN(sampling_strategy=sampling_strategy, random_state=42)
        else:
            return df
        
        X_res, y_res = sampler.fit_resample(X, y)
        
        # Convert back to DataFrame
        df_resampled = pd.DataFrame(X_res, columns=X.columns)
        df_resampled[target] = y_res
        
        # Convert encoded categoricals back to original
        if use_cat_features and cat_cols:
            for col in cat_cols:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                df_resampled[col] = le.inverse_transform(df_resampled[col].astype(int))
        
        return df_resampled
    
    def _apply_regression_method(self, df, target, method, 
                                rare_min=None, rare_max=None, relevance_factor=2.0,
                                k_neighbors=5, noise_level=0.1,
                                n_bins=10, bin_strategy='uniform'):
        """Apply regression imbalance handling methods"""
        if method == 'SmoteR':
            return self._apply_smoter(df, target, rare_min, rare_max, relevance_factor, k_neighbors, noise_level)
        elif method == 'Random Oversampling':
            return self._apply_random_oversampling(df, target, rare_min, rare_max, relevance_factor)
        elif method == 'Gaussian Noise':
            return self._apply_gaussian_noise(df, target, rare_min, rare_max, relevance_factor, noise_level)
        elif method == 'Target Binning':
            return self._apply_target_binning(df, target, n_bins, bin_strategy)
        else:
            return df
    
    def _apply_smoter(self, df, target, rare_min, rare_max, relevance_factor, k_neighbors=5, noise_level=0.1):
        """Implement SmoteR algorithm for regression imbalance"""
        # Identify rare regions
        if rare_min is None or rare_max is None:
            rare_min, rare_max = self._auto_detect_rare_regions(df[target])
        
        # Prepare data
        X = df.drop(columns=[target])
        y = df[target]
        
        # Apply SmoteR
        synthetic_samples = []
        
        # For each rare sample
        rare_mask = (y >= rare_min) & (y <= rare_max)
        rare_samples = df[rare_mask]
        
        if rare_samples.empty:
            st.warning("No rare samples found in the specified range")
            return df
        
        # Get k-nearest neighbors for each rare sample
        knn = NearestNeighbors(n_neighbors=k_neighbors)
        knn.fit(rare_samples.drop(columns=[target]))
        
        for _, rare_row in rare_samples.iterrows():
            # Find neighbors
            distances, indices = knn.kneighbors([rare_row.drop(target)], n_neighbors=k_neighbors)
            
            # Generate synthetic samples
            for i in range(k_neighbors):
                neighbor = rare_samples.iloc[indices[0][i]]
                weight = np.random.rand()
                
                # Interpolate
                synthetic = rare_row.copy()
                for col in X.columns:
                    if is_numeric_dtype(df[col]):
                        # Interpolate with noise
                        diff = neighbor[col] - rare_row[col]
                        synthetic[col] = rare_row[col] + weight * diff
                        # Add noise
                        if noise_level > 0:
                            std = df[col].std() * noise_level
                            synthetic[col] += np.random.normal(0, std)
                    else:
                        # For categorical, choose randomly between the two
                        synthetic[col] = rare_row[col] if np.random.rand() > 0.5 else neighbor[col]
                
                # Adjust target value by relevance factor
                synthetic[target] = rare_row[target] + (neighbor[target] - rare_row[target]) * weight
                synthetic_samples.append(synthetic)
        
        # Combine with original
        if synthetic_samples:
            synthetic_df = pd.DataFrame(synthetic_samples)
            return pd.concat([df, synthetic_df], ignore_index=True)
        return df
    
    def _apply_random_oversampling(self, df, target, rare_min, rare_max, relevance_factor):
        """Random oversampling of rare regions"""
        if rare_min is None or rare_max is None:
            rare_min, rare_max = self._auto_detect_rare_regions(df[target])
        
        rare_mask = (df[target] >= rare_min) & (df[target] <= rare_max)
        rare_samples = df[rare_mask]
        
        if rare_samples.empty:
            st.warning("No rare samples found in the specified range")
            return df
        
        # Determine oversampling amount
        n_rare = len(rare_samples)
        n_total = len(df)
        oversample_factor = int(relevance_factor)
        
        # Oversample
        synthetic_samples = rare_samples.sample(n=n_rare * (oversample_factor - 1), replace=True)
        return pd.concat([df, synthetic_samples], ignore_index=True)
    
    def _apply_gaussian_noise(self, df, target, rare_min, rare_max, relevance_factor, noise_level=0.1):
        """Add Gaussian noise to rare samples"""
        if rare_min is None or rare_max is None:
            rare_min, rare_max = self._auto_detect_rare_regions(df[target])
        
        rare_mask = (df[target] >= rare_min) & (df[target] <= rare_max)
        rare_samples = df[rare_mask].copy()
        
        if rare_samples.empty:
            st.warning("No rare samples found in the specified range")
            return df
        
        # Add noise to numerical features
        num_cols = Utils.get_numerical_columns(rare_samples.drop(columns=[target]))
        for col in num_cols:
            std = df[col].std() * noise_level
            rare_samples[col] += np.random.normal(0, std, size=len(rare_samples))
        
        # Combine with original
        return pd.concat([df, rare_samples], ignore_index=True)
    
    def _apply_target_binning(self, df, target, n_bins=10, strategy='uniform'):
        """Bin target and balance bins"""
        # Create bins
        bins = pd.qcut(df[target], n_bins, duplicates='drop')
        df_binned = df.copy()
        df_binned['target_bin'] = bins
        
        # Calculate bin counts
        bin_counts = df_binned['target_bin'].value_counts()
        
        # Determine target counts
        if strategy == 'uniform':
            target_count = int(bin_counts.mean())
        elif strategy == 'median':
            target_count = int(bin_counts.median())
        elif strategy == 'min':
            target_count = bin_counts.min()
        else:
            target_count = bin_counts.mean()
        
        # Resample each bin
        resampled_dfs = []
        for bin_name in bin_counts.index:
            bin_df = df_binned[df_binned['target_bin'] == bin_name]
            n_samples = len(bin_df)
            
            if n_samples < target_count:
                # Oversample
                oversampled = bin_df.sample(target_count - n_samples, replace=True)
                resampled_dfs.append(pd.concat([bin_df, oversampled]))
            elif n_samples > target_count:
                # Undersample
                resampled_dfs.append(bin_df.sample(target_count))
            else:
                resampled_dfs.append(bin_df)
        
        # Combine and clean
        result = pd.concat(resampled_dfs, ignore_index=True)
        return result.drop(columns=['target_bin'])
    
    def _auto_detect_rare_regions(self, target_series, n_bins=10, threshold=0.1):
        """Automatically detect rare regions in continuous target"""
        # Create histogram
        hist, bin_edges = np.histogram(target_series, bins=n_bins)
        
        # Find sparse bins
        density = hist / hist.sum()
        sparse_bins = np.where(density < threshold)[0]
        
        if len(sparse_bins) == 0:
            # No sparse bins, use tails
            q_low = target_series.quantile(0.05)
            q_high = target_series.quantile(0.95)
            return q_low, q_high
        
        # Find min and max of sparse regions
        min_edge = bin_edges[sparse_bins.min()]
        max_edge = bin_edges[sparse_bins.max() + 1]
        
        return min_edge, max_edge

# Main Application
class EnhancedEDA:
    def __init__(self):
        timestamp_fmt = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=timestamp_fmt)
        
        # Initialize session state
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'original_df' not in st.session_state:
            st.session_state.original_df = None
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'last_memory_check' not in st.session_state:
            st.session_state.last_memory_check = time.time()
        
        # Initialize components
        self.file_handler = FileHandler(self)
        self.data_cleaner = DataCleaner(self)
        self.visualizer = Visualizer(self)
        self.exporter = Exporter(self)
        self.imbalance_handler = ImbalanceHandler(self)
        
        # Configure page
        st.set_page_config(
            layout='wide',
            page_icon='📊',
            page_title='Enhanced EDA Tool'
        )
        self._apply_styles()
        st.title('Enhanced Exploratory Data Analysis Tool')
        st.write('Upload your dataset to explore, clean, analyze, visualize, and export interactively.')
    
    def _apply_styles(self):
        theme = Constants.DEFAULT_THEME
        st.markdown(
            f'''<style>
            .stApp {{ background-color: {theme['primary']} !important; color: {theme['text']}; }}
            [data-testid="stSidebar"] {{ background-color: {theme['secondary']} !important; color: white !important; }}
            .stButton>button {{ background-color: {theme['secondary']} !important; color: white !important; }}
            .stTextInput>div>input, .stSelectbox>div>select {{
                background-color: #f0f0f0 !important;
                color: black !important;
            }}
            h1, h2, h3, h4, h5, h6 {{ color: {theme['secondary']} !important; }}
            .warning {{ background-color: #ffcc00; color: black; padding: 10px; border-radius: 5px; }}
            @media (max-width: 768px) {{
                .mobile-hidden {{ display: none; }}
                .mobile-full {{ width: 100% !important; }}
            }}
            </style>''',
            unsafe_allow_html=True
        )
    
    def _save_state(self, action: str):
        if st.session_state.df is not None:
            if len(st.session_state.history) >= Constants.MAX_HISTORY_SIZE:
                st.session_state.history.pop(0)
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.session_state.history.append({
                'df': st.session_state.df.copy(),
                'action': action,
                'timestamp': timestamp
            })
    
    def display_info(self):
        df = st.session_state.df
        if df is None or df.empty:
            st.warning('Please upload a dataset')
            return
        
        try:
            current_time = time.time()
            if current_time - st.session_state.last_memory_check > Constants.MEMORY_MONITOR_INTERVAL:
                Utils.monitor_memory()
            
            st.subheader('2. Dataset Information')
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Rows', df.shape[0])
            c2.metric('Columns', df.shape[1])
            missing_total = int(df.isnull().sum().sum())
            c3.metric(
                'Missing Values', missing_total,
                delta=f"{missing_total/df.size*100:.1f}%" if df.size > 0 else "0%"
            )
            mem_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
            c4.metric('Memory Usage', f"{mem_usage:.2f} MB")
            
            with st.expander('Preview (10 rows)'):
                st.dataframe(df.head(10))
            
            with st.expander('Column Summary'):
                with st.spinner('Generating summary…'):
                    summary = pd.DataFrame({
                        'Column': df.columns,
                        'Data Type': df.dtypes.values,
                        'Unique Values': df.nunique().values,
                        'Missing Values': df.isnull().sum().values
                    })
                st.dataframe(summary)
            
            with st.expander('Data Quality Report'):
                quality_report = pd.DataFrame({
                    'Column': df.columns,
                    'Completeness': (1 - df.isnull().mean()).values,
                    'Uniqueness': df.nunique().values / len(df)
                })
                st.dataframe(
                    quality_report.style.format({
                        'Completeness': '{:.1%}',
                        'Uniqueness': '{:.1%}'
                    })
                )
        
        except Exception as e:
            st.error(f'Error displaying info: {str(e)}')
            logging.error(f'Info error: {str(e)}')
    
    def analyze_outliers(self):
        df = st.session_state.df
        if df is None or df.empty:
            return
        
        try:
            st.subheader('4. Outlier Analysis')
            nums = Utils.get_numerical_columns(df)
            if not nums:
                st.warning('No numerical columns found')
                return
            
            out = {}
            for c in nums:
                q1 = df[c].quantile(0.25)
                q3 = df[c].quantile(0.75)
                iqr = q3 - q1
                low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
                out[c] = {
                    'Count': int(df[(df[c]<low)|(df[c]>high)].shape[0]),
                    'Min': df[c].min(), 'Q1': q1, 'Median': df[c].median(), 
                    'Q3': q3, 'Max': df[c].max(), 'IQR': iqr
                }
            st.dataframe(pd.DataFrame.from_dict(out, orient='index'))
            
            st.markdown('---')
            st.markdown('### Handle Outliers')
            col = st.selectbox('Column:', nums)
            strat = st.selectbox('Strategy:', ['None', 'Remove', 'Cap', 'Replace Median'])
            
            if strat != 'None' and st.button('Apply'):
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
                
                if strat == 'Remove':
                    df = df[(df[col] >= low) & (df[col] <= high)]
                    st.success(f'Removed {out[col]["Count"]} outliers')
                elif strat == 'Cap':
                    df[col] = df[col].clip(low, high)
                    st.success('Capped outliers')
                else:
                    median = df[col].median()
                    df.loc[df[col] < low, col] = median
                    df.loc[df[col] > high, col] = median
                    st.success('Replaced with median')
                
                st.session_state.df = df
                self._save_state(f"Outlier handling: {strat}")
        
        except Exception as e:
            st.error(f'Outlier error: {str(e)}')
            logging.error(f'Outlier error: {str(e)}')
    
    def run(self):
        self.file_handler.load_dataset()
        
        if st.session_state.df is None:
            st.info('Please upload a dataset to begin analysis')
            return
        
        tasks = st.sidebar.multiselect(
            'Select tasks to perform:',
            ['Data Overview', 'Data Cleaning', 'Outlier Analysis',
             'Visualization', 'Handle Imbalance', 'Data Export'],
            default=['Data Overview']
        )
        
        if 'Data Overview' in tasks:
            self.display_info()
        if 'Data Cleaning' in tasks:
            self.data_cleaner.handle_missing()
        if 'Outlier Analysis' in tasks:
            self.analyze_outliers()
        if 'Visualization' in tasks:
            self.visualizer.visualize()
        if 'Handle Imbalance' in tasks:
            self.imbalance_handler.handle_imbalance()
        if 'Data Export' in tasks:
            self.exporter.download_data()
        
        st.sidebar.markdown('---')
        st.sidebar.subheader('Dataset Status')
        st.sidebar.write(f"Rows: {st.session_state.df.shape[0]}")
        st.sidebar.write(f"Columns: {st.session_state.df.shape[1]}")
        miss = st.session_state.df.isnull().sum().sum()
        st.sidebar.write(f"Missing Values: {miss}")
        mem_usage = st.session_state.df.memory_usage(deep=True).sum() / (1024 ** 2)
        st.sidebar.write(f"Memory Usage: {mem_usage:.2f} MB")
        
        if st.session_state.history:
            st.sidebar.markdown('---')
            st.sidebar.subheader('History')
            for i, entry in enumerate(reversed(st.session_state.history)):
                if st.sidebar.button(f"{i+1}. {entry['action']} ({entry['timestamp']})"):
                    st.session_state.df = entry['df']
                    st.rerun()

if __name__ == '__main__':
    EnhancedEDA().run()