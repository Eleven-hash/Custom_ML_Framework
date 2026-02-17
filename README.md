# Custom ML Framework

A comprehensive, interactive machine learning pipeline framework built with Python and sklearn. This framework provides an end-to-end solution for data loading, preprocessing, model training, evaluation, and automated report generation.

## Features

### 🔄 Flexible Data Loading
- **CSV files**: Load data from local CSV files
- **Kaggle datasets**: Support for locally downloaded Kaggle datasets
- **sklearn datasets**: Built-in support for sklearn toy datasets (breast cancer, iris, etc.)

### 🎯 Automatic Task Detection
- Automatically detects whether the problem is:
  - Classification
  - Regression
  - Clustering
- Manual override available through configuration

### ⚙️ Comprehensive Preprocessing Pipeline
Interactive configuration system to enable/disable:
- **Data sampling**: SMOTE, Random Over/Under sampling
- **Normalization**: MinMax scaling
- **Standardization**: Standard scaling
- **Dimensionality reduction**: PCA
- **Outlier removal**: Isolation Forest
- **Feature selection**: Automated feature engineering

### 📊 Exploratory Data Analysis (EDA)
- Dataset statistics and summary
- Missing value analysis
- Correlation heatmaps
- Feature distribution plots
- Target variable distribution

### 🤖 Multi-Model Training
Built-in support for multiple algorithms:

**Classification**:
- Logistic Regression
- Random Forest Classifier
- Support Vector Classifier (SVC)

**Regression**:
- Linear Regression
- Random Forest Regressor
- Support Vector Regressor (SVR)

**Clustering**:
- K-Means

### 📈 Model Evaluation
- **Classification metrics**: Accuracy, Precision, Recall, F1-score, ROC curves
- **Regression metrics**: RMSE, R-squared
- **Visualization**: Confusion matrices, residual plots, cluster plots
- Automated metrics comparison table

### 📄 Automated Report Generation
- PDF report generation with all visualizations
- Comprehensive metrics summary
- Model comparison tables
- All plots saved in organized directory structure

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn fpdf
```

## Quick Start

1. **Configure your pipeline** in the `CONFIG` dictionary:

```python
CONFIG = {
    'csv_path': None,                    # Path to CSV file
    'kaggle_local_path': None,          # Path to Kaggle dataset
    'sklearn_name': 'breast_cancer',    # sklearn dataset name
    
    'task': 'auto',                     # 'auto', 'classification', 'regression', 'clustering'
    'test_size': 0.25,
    'random_state': 42,
    
    # Preprocessing options
    'apply_sampling': False,
    'sampling_method': 'smote',
    'apply_normalization': False,
    'apply_standardization': True,
    'apply_dim_reduction': False,
    'dim_components': 2,
    'apply_outlier_removal': False,
    
    # Output settings
    'report_path': 'ML_Report.pdf',
    'fig_dir': 'graph_images'
}
```

2. **Run the pipeline**: Simply execute all cells in the Jupyter notebook

3. **Review results**:
   - Visualizations are saved in the `graph_images` directory
   - Comprehensive PDF report is generated as `ML_Report.pdf`
   - Model comparison metrics displayed in the notebook

## Project Structure

```
ML pipeline/
├── Custom_ML_Framework.ipynb    # Main pipeline notebook
├── graph_images/                 # Generated visualizations
│   ├── correlation_matrix.png
│   ├── target_distribution.png
│   ├── feature_distributions.png
│   ├── cm_*.png                 # Confusion matrices
│   ├── resid_*.png              # Residual plots
│   └── clusters_*.png           # Cluster visualizations
└── ML_Report.pdf                # Final report
```

## Workflow

1. **Data Loading**: Load from CSV, Kaggle, or sklearn datasets
2. **EDA**: Automated exploratory data analysis with visualizations
3. **Preprocessing**: Configurable data preprocessing pipeline
4. **Model Training**: Train multiple models simultaneously
5. **Evaluation**: Comprehensive model evaluation and comparison
6. **Reporting**: Generate PDF report with all results

## Configuration Options

### Data Sources
- Set `csv_path` for custom CSV files
- Set `kaggle_local_path` for Kaggle datasets
- Set `sklearn_name` for built-in datasets ('breast_cancer', 'iris')

### Task Types
- `'auto'`: Automatically detect task type
- `'classification'`: Binary/multi-class classification
- `'regression'`: Continuous value prediction
- `'clustering'`: Unsupervised grouping

### Preprocessing
All preprocessing steps are optional and can be toggled in CONFIG:
- Sampling methods: SMOTE, random over/under sampling
- Scaling: MinMax normalization or standard scaling
- Dimensionality reduction via PCA
- Outlier detection with Isolation Forest

## Model Customization

Add or modify models in the CONFIG dictionary:

```python
'classification_models': {
    'LogisticRegression': LogisticRegression(max_iter=500),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVC': SVC(probability=True)
},
```

## Output

The framework generates:
1. **Interactive notebooks** with all results
2. **Visualization files** (PNG format)
3. **Comprehensive PDF report** with:
   - Dataset summary
   - Model performance metrics
   - All generated plots
   - Comparison tables

## Example Use Cases

- Quick prototyping of ML models
- Comparing multiple algorithms on same dataset
- Educational purposes for learning ML workflows
- Baseline model generation for competitions
- Automated model evaluation and reporting

## Notes

- The framework automatically handles missing values
- Categorical encoding is handled automatically where applicable
- All random operations use the specified `random_state` for reproducibility
- Cross-validation can be easily added to the model training functions

## License

This project is open source and available for educational and commercial use.

## Contributing

Feel free to extend the framework by:
- Adding more preprocessing techniques
- Including additional models
- Enhancing visualization options
- Improving report generation

---

Built with ❤️ using Python, scikit-learn, and Jupyter
