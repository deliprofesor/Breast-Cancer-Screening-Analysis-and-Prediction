# Breast-Cancer-Screening-Analysis-and-Prediction

# Breast Cancer Screening Data Analysis and Modeling

This project involves an extensive analysis of a breast cancer screening dataset. The goal is to explore the data, clean it, visualize patterns, and build machine learning models to classify cancer types. The analysis includes preprocessing steps, feature importance analysis, and visualization to better understand the data. Additionally, advanced machine learning techniques like SMOTE and Random Forest with hyperparameter tuning are used.

---

## Features of the Project

### 1. **Data Loading and Exploration**
- Loaded the dataset `CSAW-CC_breast_cancer_screening_data.csv`.
- Inspected the structure and statistics of the dataset using `info()`, `describe()`, and `head()` functions.
- Identified missing values and their distribution across the dataset.

### 2. **Data Preprocessing**
- Filled missing categorical values with the most frequent category.
- Scaled numerical features using `StandardScaler`.
- Visualized missing values and handled them appropriately using techniques like median imputation.

### 3. **Exploratory Data Analysis (EDA)**
- Visualized various distributions and relationships in the data:
  - Age distribution.
  - Cancer type and laterality distribution.
  - Examination year trends.
  - Radiological scores and their distributions.
  - Libra breast area and dense area correlations.
- Box plots and scatter plots were used to understand feature relationships.

### 4. **Statistical Analysis**
- Conducted t-tests to analyze differences between age groups for different cancer types.
- Computed a correlation matrix to identify strong relationships between numerical features.

### 5. **Feature Engineering and Clustering**
- Applied KMeans clustering to segment data into groups based on numerical features.
- Visualized clusters to understand their characteristics.

### 6. **Machine Learning Models**
- Built a Random Forest model for cancer type classification.
- Applied SMOTE to handle imbalanced data.
- Optimized the Random Forest model using Grid Search for hyperparameter tuning.

### 7. **Visualization**
- Utilized `Matplotlib` and `Seaborn` for high-quality visualizations:
  - Heatmaps for correlation analysis.
  - Scatter plots for feature relationships.
  - Count plots for category distributions.
  - Feature importance bar chart.

### 8. **Deployment with Streamlit**
- Integrated Streamlit to provide an interactive visualization dashboard.
- Displayed descriptive statistics and EDA visualizations directly in the Streamlit app.

### 9. **Model Saving**
- Saved the trained Random Forest model using `joblib` for future use.

---

## Installation and Usage

### Prerequisites
- Python 3.8 or higher
- Required libraries:
  ```bash
  pandas
  matplotlib
  seaborn
  scikit-learn
  imbalanced-learn
  scipy
  streamlit
  joblib
  ```

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project
1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open the provided URL in your browser to interact with the dashboard.

3. Alternatively, run the script in a standard Python environment to execute the analysis and modeling pipeline.

---

## Files in the Repository

- **`main.py`**: Main script containing the data analysis, visualization, and model training pipeline.
- **`app.py`**: Streamlit application for interactive visualization.
- **`requirements.txt`**: List of required Python packages.
- **`csaw_cc_cleaned.csv`**: Cleaned dataset saved after preprocessing.
- **`breast_cancer_model.pkl`**: Trained Random Forest model saved for deployment.

---

## Key Results
- The Random Forest model achieved satisfactory classification performance for cancer types.
- Grid Search optimized the model, resulting in improved accuracy and balanced class performance.
- Clustering and EDA revealed significant relationships between features, offering insights into breast cancer patterns.

---

## Future Work
- Extend the model to predict additional outcomes, such as cancer laterality or metastasis probability.
- Integrate more advanced machine learning models like Gradient Boosting or Neural Networks.
- Enhance the Streamlit app to allow users to input data and get predictions directly.
- Perform longitudinal analysis using examination years to detect trends over time.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments
- **CSAW-CC Dataset**: The dataset used in this project is publicly available for breast cancer research.
- Libraries like Scikit-learn, Pandas, and Matplotlib enabled the completion of this project.

---

Feel free to contribute or raise issues to improve this project!
