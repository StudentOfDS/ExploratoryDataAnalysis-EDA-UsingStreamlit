import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import plotly.express as px
import io
import logging
import re
from typing import List

# Configure logging
timestamp_fmt = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=timestamp_fmt)

class EnhancedEDA:
    def __init__(self):
        # Initialize session state
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'history' not in st.session_state:
            st.session_state.history = []

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
        """Apply custom CSS styling for theming."""
        st.markdown(
            '''<style>
            .stApp { background-color: #191970 !important; color: white; }
            [data-testid="stSidebar"] { background-color: #6699CC !important; color: white !important; }
            .stSidebar .stSidebarHeader { background-color: #6699CC !important; }
            .stButton>button { background-color: #6699CC !important; color: white !important; }
            .stTextInput>div>input, .stSelectbox>div>select {
                background-color: #f0f0f0 !important;
                color: black !important;
            }
            h1, h2, h3, h4, h5, h6 { color: #6699CC !important; }
            .warning { background-color: #ffcc00; color: black; padding: 10px; border-radius: 5px; }
            </style>''',
            unsafe_allow_html=True
        )

    def _save_state(self):
        """Save current DataFrame state to history for undo functionality."""
        if st.session_state.df is not None:
            st.session_state.history.append(st.session_state.df.copy())

    def load_dataset(self):
        """Handles file upload and loading into session state."""
        st.sidebar.header('1. Upload Dataset')
        uploaded = st.sidebar.file_uploader(
            'Choose file', type=['csv','txt','xlsx','json','parquet']
        )
        if not uploaded:
            return
        try:
            with st.spinner('Reading file…'):
                buf = io.BytesIO(uploaded.read())
                ext = uploaded.name.split('.')[-1].lower()
                if ext == 'csv':
                    df = pl.read_csv(buf).to_pandas()
                elif ext == 'txt':
                    df = pl.read_csv(buf, separator='\t').to_pandas()
                elif ext == 'xlsx':
                    df = pd.read_excel(buf)
                elif ext == 'json':
                    df = pd.read_json(buf)
                elif ext == 'parquet':
                    df = pd.read_parquet(buf)
                else:
                    st.error('Unsupported file format')
                    logging.error(f'Unsupported format: {ext}')
                    return

                if df.empty:
                    st.error('Uploaded file is empty')
                    logging.error('Empty DataFrame uploaded')
                    return

                st.session_state.df = df
                self._save_state()
                st.success(f'Loaded {uploaded.name}: {df.shape[0]} rows, {df.shape[1]} cols')
                logging.info(f'Dataset loaded: {uploaded.name}')
        except Exception as e:
            st.error(f'Error loading file: {e}')
            logging.error(f'Load error: {e}')

    @st.cache_data(ttl=3600)
    def _get_numerical_columns(self) -> List[str]:
        """Returns numerical columns, cached for performance."""
        try:
            df = st.session_state.df
            return [] if df is None else df.select_dtypes(include=np.number).columns.tolist()
        except Exception as e:
            st.error(f'Error identifying numerical columns: {e}')
            logging.error(f'Numeric cols error: {e}')
            return []

    @st.cache_data(ttl=3600)
    def _get_categorical_columns(self) -> List[str]:
        """Returns categorical columns, cached for performance."""
        try:
            df = st.session_state.df
            return [] if df is None else df.select_dtypes(include=['object','category']).columns.tolist()
        except Exception as e:
            st.error(f'Error identifying categorical columns: {e}')
            logging.error(f'Categorical cols error: {e}')
            return []

    def display_info(self):
        """Displays dataset metrics and summary."""
        df = st.session_state.df
        if df is None:
            st.warning('Please upload a dataset')
            return
        try:
            st.subheader('2. Dataset Information')
            c1, c2, c3 = st.columns(3)
            c1.metric('Rows', df.shape[0])
            c2.metric('Columns', df.shape[1])
            missing_total = int(df.isnull().sum().sum())
            c3.metric(
                'Missing Values', missing_total,
                delta=f"{missing_total/df.size*100:.1f}%" if df.size > 0 else "0%"
            )

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
            st.error(f'Error displaying info: {e}')
            logging.error(f'Info error: {e}')

    def handle_missing(self):
        """Imputes missing values and provides advanced cleaning."""
        df = st.session_state.df
        if df is None:
            return
        try:
            st.subheader('3. Data Cleaning & Missing Values')

            mv = df.isnull().sum()
            if mv.sum() == 0:
                st.success('No missing values detected')
            else:
                st.write('Missing values per column:')
                st.dataframe(mv[mv > 0])

                st.markdown('### Impute Missing Values')
                cols = st.multiselect('Select columns to fill:', df.columns)
                strat = st.selectbox('Imputation strategy:', ['mean','median','mode','custom'])

                custom_val = None
                if strat == 'custom':
                    custom_val = st.text_input('Enter custom value:')
                    if not custom_val:
                        st.warning('Please enter a custom value')

                if cols and st.button('Apply Imputation'):
                    fill_vals = {}
                    for c in cols:
                        if strat == 'mean' and c in self._get_numerical_columns():
                            fill_vals[c] = df[c].mean()
                        elif strat == 'median' and c in self._get_numerical_columns():
                            fill_vals[c] = df[c].median()
                        elif strat == 'mode':
                            mode_vals = df[c].mode()
                            if not mode_vals.empty:
                                fill_vals[c] = mode_vals.iloc[0]
                        elif strat == 'custom':
                            fill_vals[c] = custom_val
                    if fill_vals:
                        df = df.fillna(fill_vals)
                        st.session_state.df = df
                        self._save_state()
                        st.success(f'Imputation applied to {len(fill_vals)} columns')
                    else:
                        st.warning('No valid imputation strategy selected')

            st.markdown('---')
            st.markdown('### Advanced Cleaning Operations')
            adv = st.selectbox(
                'Select operation:',
                ['None','Remove Duplicates','Standardize Columns','Rename Column','Drop Columns','Convert Data Type']
            )

            if adv == 'Remove Duplicates' and st.button('Execute'):
                before = df.shape[0]
                df = df.drop_duplicates()
                st.session_state.df = df
                self._save_state()
                st.success(f'Removed {before - df.shape[0]} duplicate rows')

            elif adv == 'Standardize Columns' and st.button('Execute'):
                new_cols = []
                for name in df.columns:
                    clean = name.lower().strip()
                    clean = re.sub(r'\s+', '_', clean)
                    clean = re.sub(r'[^a-z0-9_]', '', clean)
                    new_cols.append(clean)
                df.columns = [f'col_{i}' if not col else col for i, col in enumerate(new_cols)]
                st.session_state.df = df
                self._save_state()
                st.success('Column names standardized')

            elif adv == 'Rename Column':
                col = st.selectbox('Select column:', df.columns)
                new = st.text_input('New name:')
                if new and st.button('Rename'):
                    if new in df.columns:
                        st.error('Column name already exists!')
                    else:
                        df = df.rename(columns={col: new})
                        st.session_state.df = df
                        self._save_state()
                        st.success(f'Renamed {col} to {new}')

            elif adv == 'Drop Columns' and st.button('Execute'):
                drops = st.multiselect('Select columns to drop:', df.columns)
                if drops:
                    df = df.drop(columns=drops)
                    st.session_state.df = df
                    self._save_state()
                    st.success(f"Dropped columns: {', '.join(drops)}")

            elif adv == 'Convert Data Type':
                col = st.selectbox('Select column:', df.columns)
                new_type = st.selectbox('Select new type:', ['string','integer','float','category','datetime'])
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
                        self._save_state()
                        st.success(f'Converted {col} to {new_type}')
                    except Exception as e:
                        st.error(f'Conversion error: {e}')

            st.markdown('---')
            if st.button('Undo Last Action') and len(st.session_state.history) > 1:
                st.session_state.history.pop()
                st.session_state.df = st.session_state.history[-1]
                st.success('Last action undone')

        except Exception as e:
            st.error(f'Cleaning error: {e}')
            logging.error(f'Cleaning error: {e}')

    def analyze_outliers(self):
        """Detects outliers using the IQR method."""
        df = st.session_state.df
        if df is None:
            return
        try:
            st.subheader('4. Outlier Analysis')
            nums = self._get_numerical_columns()
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
                    'Min': df[c].min(), 'Q1': q1, 'Median': df[c].median(), 'Q3': q3,
                    'Max': df[c].max(), 'IQR': iqr
                }
            st.dataframe(pd.DataFrame.from_dict(out, orient='index'))
            st.markdown('---')
            st.markdown('### Handle Outliers')
            col = st.selectbox('Column:', nums)
            strat = st.selectbox('Strategy:', ['None','Remove','Cap','Replace Median'])
            if strat!='None' and st.button('Apply'):
                q1, q3 = df[col].quantile([0.25,0.75])
                iqr = q3 - q1
                low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
                if strat=='Remove':
                    df = df[(df[col]>=low)&(df[col]<=high)]
                    st.success('Removed outliers')
                elif strat=='Cap':
                    df[col] = df[col].clip(low, high)
                    st.success('Capped outliers')
                else:
                    median = df[col].median()
                    df.loc[df[col]<low, col] = median
                    df.loc[df[col]>high, col] = median
                    st.success('Replaced with median')
                st.session_state.df = df
                self._save_state()
        except Exception as e:
            st.error(f'Outlier error: {e}')
            logging.error(f'Outlier error: {e}')

    def visualize(self):
        """Renders interactive Plotly visualizations."""
        df = st.session_state.df
        if df is None:
            return
        try:
            st.subheader('5. Data Visualization')
            opts = [
                'Histogram', 'Count Plot', 'Box Plot',
                'Scatter Plot', 'Pair Plot', 'Correlation Heatmap',
                'Time Series', 'Pie Chart'
            ]
            choice = st.selectbox('Select visualization type:', opts)

            if choice == 'Histogram':
                col = st.selectbox('Numerical column:', self._get_numerical_columns())
                bins = st.slider('Number of bins:', 5, 100, 20)
                color_col = st.selectbox('Color by (optional):', ['None'] + self._get_categorical_columns())
                color = None if color_col == 'None' else color_col
                
                with st.spinner('Generating histogram…'):
                    fig = px.histogram(
                        df, x=col, nbins=bins, color=color,
                        title=f'Distribution of {col}',
                        marginal='box'
                    )
                st.plotly_chart(fig, use_container_width=True)

            elif choice == 'Count Plot':
                col = st.selectbox('Categorical column:', self._get_categorical_columns())
                top_n = st.slider('Show top N categories:', 5, 50, 10)
                
                with st.spinner('Generating count plot…'):
                    counts = df[col].value_counts().reset_index().head(top_n)
                    counts.columns = [col, 'Count']
                    fig = px.bar(
                        counts, x=col, y='Count',
                        title=f'Top {top_n} Categories in {col}',
                        color=col
                    )
                st.plotly_chart(fig, use_container_width=True)

            elif choice == 'Box Plot':
                cols = st.multiselect('Numerical columns:', self._get_numerical_columns())
                if cols:
                    melted = df[cols].melt(var_name='Variable', value_name='Value')
                    with st.spinner('Generating box plot…'):
                        fig = px.box(
                            melted, x='Variable', y='Value',
                            title='Distribution of Numerical Columns',
                            color='Variable'
                        )
                    st.plotly_chart(fig, use_container_width=True)

            elif choice == 'Scatter Plot':
                nums = self._get_numerical_columns()
                if len(nums) >= 2:
                    x = st.selectbox('X-axis:', nums)
                    y = st.selectbox('Y-axis:', nums)
                    color_col = st.selectbox('Color by:', ['None'] + self._get_categorical_columns())
                    size_col = st.selectbox('Size by (optional):', ['None'] + self._get_numerical_columns())
                    hover_col = st.selectbox('Hover info:', ['None'] + list(df.columns))

                    color = None if color_col == 'None' else color_col
                    size = None if size_col == 'None' else size_col
                    hover_data = [hover_col] if hover_col != 'None' else None

                    with st.spinner('Generating scatter plot…'):
                        fig = px.scatter(
                            df, x=x, y=y, color=color, size=size,
                            hover_data=hover_data, title=f'{x} vs {y}',
                            trendline='ols'
                        )
                    st.plotly_chart(fig, use_container_width=True)

            elif choice == 'Pair Plot':
                nums = self._get_numerical_columns()
                if len(nums) >= 2:
                    sel = st.multiselect('Select columns:', nums)

                    # Limit to 5 columns for performance
                    if len(sel) > 5:
                        st.warning("Pair plots limited to 5 columns for performance")
                        sel = sel[:5]

                    if sel:
                        sample_size = st.slider('Sample size:', 100, min(2000, len(df)), 500)
                        with st.spinner('Rendering pair plot…'):
                            fig = px.scatter_matrix(
                                df[sel].sample(sample_size),
                                dimensions=sel,
                                title='Pairwise Relationships',
                                color=sel[0] if sel else None
                            )
                        st.plotly_chart(fig, use_container_width=True)

            elif choice == 'Correlation Heatmap':
                nums = self._get_numerical_columns()
                if len(nums) >= 2:
                    with st.spinner('Generating heatmap…'):
                        corr = df[nums].corr().round(2)
                        fig = px.imshow(
                            corr,
                            text_auto=True,
                            color_continuous_scale='RdBu',
                            title='Correlation Matrix',
                            zmin=-1, zmax=1
                        )
                    st.plotly_chart(fig, use_container_width=True)

            elif choice == 'Time Series':
                date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
                if not date_cols:
                    st.warning('No datetime columns found')
                else:
                    date_col = st.selectbox('Date column:', date_cols)
                    value_col = st.selectbox('Value column:', self._get_numerical_columns())
                    agg_func = st.selectbox('Aggregation:', ['sum', 'mean', 'count'])

                    with st.spinner('Generating time series…'):
                        ts_df = df.set_index(date_col)[value_col].resample('D').agg(agg_func).reset_index()
                        fig = px.line(
                            ts_df, x=date_col, y=value_col,
                            title=f'{agg_func.capitalize()} of {value_col} over Time'
                        )
                    st.plotly_chart(fig, use_container_width=True)

            elif choice == 'Pie Chart':
                cat_col = st.selectbox('Categorical column:', self._get_categorical_columns())
                top_n = st.slider('Show top N:', 3, 20, 5)
                
                with st.spinner('Generating pie chart…'):
                    counts = df[cat_col].value_counts().reset_index().head(top_n)
                    counts.columns = [cat_col, 'Count']
                    fig = px.pie(
                        counts, names=cat_col, values='Count',
                        title=f'Distribution of {cat_col} (Top {top_n})'
                    )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f'Visualization error: {e}')
            logging.error(f'Visualization error: {e}')

    def download_data(self):
        """Exports processed data in chosen format."""
        df = st.session_state.df
        if df is None:
            return
        try:
            st.subheader('6. Export Processed Data')
            if df.empty:
                st.warning('No data to export')
                return

            fmt = st.selectbox('Select export format:', ['CSV','Excel','Parquet','JSON'])
            buf = io.BytesIO()
            mime = ''

            if fmt == 'CSV':
                buf.write(df.to_csv(index=False).encode())
                mime = 'text/csv'
            elif fmt == 'JSON':
                buf.write(df.to_json(orient='records').encode())
                mime = 'application/json'
            elif fmt == 'Parquet':
                df.to_parquet(buf)
                mime = 'application/octet-stream'
            else:  # Excel
                try:
                    import xlsxwriter
                except ImportError:
                    st.error('Excel export requires xlsxwriter. Install with: pip install xlsxwriter')
                    return
                with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='ProcessedData')
                mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

            buf.seek(0)
            size_mb = len(buf.getvalue()) / (1024**2)

            st.info(f"File size: {size_mb:.2f} MB")
            if size_mb > 50:
                st.markdown('<div class="warning">Large file warning: Download may be slow</div>', unsafe_allow_html=True)

            st.download_button(
                '⬇️ Download Processed Data',
                data=buf,
                file_name=f'processed_data.{fmt.lower()}',
                mime=mime
            )
        except Exception as e:
            st.error(f'Export error: {e}')
            logging.error(f'Export error: {e}')

    def run(self):
        """Controls the application flow."""
        self.load_dataset()
        if st.session_state.df is None:
            st.info('Please upload a dataset to begin analysis')
            return
        tasks = st.sidebar.multiselect(
            'Select tasks to perform:',
            ['Data Overview', 'Data Cleaning', 'Outlier Analysis',
             'Visualization', 'Data Export'],
            default=['Data Overview']
        )
        if 'Data Overview' in tasks:
            self.display_info()
        if 'Data Cleaning' in tasks:
            self.handle_missing()
        if 'Outlier Analysis' in tasks:
            self.analyze_outliers()
        if 'Visualization' in tasks:
            self.visualize()
        if 'Data Export' in tasks:
            self.download_data()
        st.sidebar.markdown('---')
        st.sidebar.subheader('Dataset Status')
        st.sidebar.write(f"Rows: {st.session_state.df.shape[0]}")
        st.sidebar.write(f"Columns: {st.session_state.df.shape[1]}")
        miss = st.session_state.df.isnull().sum().sum()
        st.sidebar.write(f"Missing Values: {miss}")

if __name__ == '__main__':
    EnhancedEDA().run()