


# Enhanced Exploratory Data Analysis (EDA) Tool

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3D4F80?style=for-the-badge&logo=plotly&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.13.1-blue?style=for-the-badge&logo=python)

The **Enhanced EDA Tool** is an interactive, web-based application for performing advanced exploratory data analysis. Built with **Streamlit**, it combines modern visualizations, robust data cleaning tools, and customizable analysis options, making it a powerful solution for data scientists and analysts.

---

## 🚀 **Key Features**

- **Multi-format Support**:
  - Upload files in CSV, Excel, JSON, Parquet, and TSV formats.
- **Advanced Data Cleaning**:
  - Drop specific columns or patterns.
  - Handle missing values with strategies like drop, fill, or keep.
  - Smart fill values: mean, median, mode, or custom inputs.
  - Normalize column names (e.g., replace spaces with underscores).
- **Interactive Visualizations**:
  - Histograms with adjustable bins and custom colors.
  - Box plots, scatter plots, and pair plots.
  - Correlation heatmaps.
  - Categorical count plots.
- **Statistical Analysis**:
  - Outlier detection using the IQR method.
  - Descriptive statistics (mean, median, standard deviation, etc.).
  - Detailed missing value summaries.
- **Export Processed Data**:
  - Download cleaned datasets in Excel format.

---

## 🛠️ **Installation**

### Prerequisites:
- Python 3.13.1 or higher

### Clone the Repository:
```bash
git clone https://github.com/yourusername/enhanced-eda-tool.git
cd enhanced-eda-tool
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

---

## 📝 **Usage**

### Run the Application:
```bash
streamlit run EDA_On_Streamlit_Main.py
```

### Workflow:
1. **Upload a Dataset**:
   - Use the sidebar to upload files (CSV, Excel, JSON, Parquet, or TSV).
2. **Select EDA Tasks**:
   - Choose from cleaning, visualization, and analysis options in the sidebar.
3. **Apply Data Cleaning**:
   - Use advanced tools to clean and preprocess your data (e.g., drop columns, fill missing values).
4. **Customize Visualizations**:
   - Adjust parameters like bins, colors, and axes for charts.
5. **Export Processed Data**:
   - Save the cleaned dataset as an Excel file.

---

## 📊 **Features in Action**

### **Interactive Visualizations**:
![Image 1](https://i.postimg.cc/G2hqRwZd/IV00.png)
![Image 2](https://i.postimg.cc/MpmLJDv0/IV0.png)
![Image 3](https://i.postimg.cc/wB3Q4cJj/IV1.png)
![Image 4](https://i.postimg.cc/3R50jgF6/IV2.png)
![Image 5](https://i.postimg.cc/C1s5j91B/IV3.png)



### **Advanced Data Cleaning**:
![Image 6](https://i.postimg.cc/zBz81vzv/ADC1.png)
![Image 7](https://i.postimg.cc/yxWKkGSz/ADC2.png)
![Image 8](https://i.postimg.cc/VvT8MFj0/ADC3.png)
![Image 9](https://i.postimg.cc/SsgqQ4Rr/ADC4.png)


---

## 📚 **Technical Stack**

| **Library**   | **Purpose**                                                                 | **Version** |
|---------------|-----------------------------------------------------------------------------|-------------|
| Streamlit     | Web application framework for creating interactive UI                      | 1.41.1      |
| Pandas        | Data manipulation and analysis                                             | 2.2.3       |
| Plotly        | Interactive visualizations (histograms, box plots, scatter plots, etc.)    | 5.24.1      |
| Seaborn       | Statistical visualizations (correlation heatmaps, count plots, etc.)       | 0.13.2      |
| Matplotlib    | Foundational plotting library, often used with Seaborn                    | 3.10.0      |
| NumPy         | Numerical computing library for array operations and calculations          | 2.2.2       |
| Polars        | High-performance DataFrame library for efficient data ingestion            | 1.20.0      |
| XlsxWriter    | Excel file generation for exporting processed datasets                     | 3.2.0       |
| Openpyxl      | Excel file support for reading uploaded datasets                           | 3.1.5       |

---

## 🔍 **Libraries Used**

| Library       | Purpose                                                                                     | Version |
|---------------|---------------------------------------------------------------------------------------------|---------|
| **Streamlit** | Web application framework for creating interactive UI and hosting the EDA tool              | 1.41.1  |
| **Pandas**    | Data manipulation and analysis - handles DataFrame operations, file I/O, and data cleaning  | 2.2.3   |
| **Plotly**    | Interactive visualizations for histograms, box plots, scatter plots, and pair plots         | 5.24.1  |
| **Seaborn**   | Statistical visualizations - used for correlation heatmaps and enhanced matplotlib styling  | 0.13.2  |
| **Matplotlib**| Foundational plotting library - used as base for Seaborn plots and figure management        | 3.10.0  |
| **NumPy**     | Numerical computing - powers statistical calculations and array operations                 | 2.2.2   |
| **Polars**    | High-performance DataFrame library - used for fast CSV/TSV file loading                    | 1.20.0  |
| **XlsxWriter**| Excel file generation engine - required for exporting processed data to Excel               | 3.2.0   |
| **Openpyxl**  | Excel file reading support - enables pandas to handle .xlsx file uploads                    | 3.1.5   |

---

## 🖼️ **About the Tool**

The **Enhanced EDA Tool** empowers data professionals by offering a robust suite of tools for cleaning, analyzing, and visualizing datasets in real-time. Its integration of modern frameworks like **Streamlit**, **Pandas**, and **Plotly** ensures a seamless experience, making data analysis interactive, efficient, and accessible.



---

## 🌟 **Contribute**

We welcome contributions! If you'd like to report issues, suggest features, or contribute code:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m "Add new feature"`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a pull request

Happy analyzing! 😊

---

### **How to Use This**
1. Copy the entire content above.  
2. Paste it into your `README.md` file.  
3. Replace placeholders:  
   - `https://github.com/yourusername/enhanced-eda-tool.git` → Your GitHub repository URL.  
   - Placeholder images (`https://via.placeholder.com/...`) → Actual screenshots of your tool.  
4. Save and commit the file to your repository.
 

