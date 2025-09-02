# Student Performance Prediction

This repository contains a machine learning pipeline for predicting student exam scores based on various performance factors.

## Project Structure

```
Student-Scores/
│
├── data/                           # Dataset files
│   └── StudentPerformanceFactors.csv
│
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── data_processor.py          # Data loading and preprocessing
│   ├── model.py                   # Machine learning models
│   └── visualizer.py              # Data visualization utilities
│
├── notebooks/                     # Jupyter notebooks for exploration
│   └── Data_Visualization.ipynb   # Data analysis and visualization
│
├── outputs/                       # Generated outputs
│   ├── images/                    # Saved plots and visualizations
│   └── models/                    # Trained model files
│
├── main.py                        # Main script to run the complete workflow
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Features

- **Modular Design**: Clean separation of data processing, modeling, and visualization
- **Multiple Models**: Supports Random Forest, Linear Regression, and Polynomial Regression
- **Comprehensive Model Comparison**: Automated comparison of multiple algorithms
- **Regularization Support**: Ridge and Lasso regression for polynomial features
- **Comprehensive Visualization**: Automated generation of performance plots
- **Model Persistence**: Save and load trained models
- **Easy-to-Use**: Single script execution for complete workflow

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Required libraries (see requirements.txt)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Ahmed-Hajhamed/Student-Scores.git
   cd Student-Scores
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Quick Start
Run the complete workflow with a single command:
```bash
python main.py
```

#### Comprehensive Model Comparison
Compare Linear, Polynomial, and Random Forest models:
```bash
python model_comparison.py
```

#### Test Regularized Polynomial Regression
Explore regularization techniques for polynomial features:
```bash
python test_regularized_polynomial.py
```

The main workflow will:
- Load and preprocess the data
- Generate data visualizations
- Train multiple models
- Evaluate model performance
- Save results and trained models

#### Using Individual Modules

```python
from src.data_processor import DataProcessor
from src.model import StudentPerformanceModel
from src.visualizer import Visualizer

# Load and process data
processor = DataProcessor("data/StudentPerformanceFactors.csv")
processor.load_data()
X, y = processor.prepare_features()

# Train a model
model = StudentPerformanceModel(processor.create_preprocessor())
model.create_random_forest_model()
model.train(X_train, y_train)

# Visualize results
visualizer = Visualizer()
visualizer.plot_actual_vs_predicted(y_test, predictions, metrics)
```

## Dataset

The dataset (`StudentPerformanceFactors.csv`) contains various factors that influence student performance:

- **Academic Factors**: Hours studied, Previous scores, Attendance
- **Personal Factors**: Sleep hours, Motivation level, Physical activity
- **Social Factors**: Parental involvement, Peer influence, Family income
- **Resource Factors**: Access to resources, Internet access, Tutoring sessions
- **Institutional Factors**: Teacher quality, School type, Distance from home

## Model Performance

Comprehensive comparison of different algorithms:

| Model | MAE | RMSE | R² Score | Performance |
|-------|-----|------|----------|-------------|
| **Linear Regression** | 0.45 | 1.80 | **0.7699** | 🥇 Best |
| Polynomial Regression (d=2) | 0.66 | 1.90 | 0.7445 | 🥈 Good |
| Random Forest | 1.23 | 2.25 | 0.6415 | 🥉 Moderate |
| Polynomial Regression (d=3) | 2.77 | 4.34 | -0.3296 | ❌ Overfitted |

**Key Findings:**
- Linear Regression achieves the best performance with R² = 0.7699
- Higher-degree polynomial regression suffers from overfitting
- Regularization (Ridge/Lasso) helps but doesn't surpass linear regression
- The dataset appears to have predominantly linear relationships

## Visualizations

The project automatically generates several visualizations:

- **Data Distribution**: Histogram and box plot of exam scores
- **Correlation Matrix**: Heatmap showing feature relationships
- **Categorical Analysis**: Impact of categorical variables on performance
- **Model Comparison**: Bar charts comparing all model metrics
- **Actual vs Predicted**: Scatter plots for each model
- **Residuals Analysis**: Error distribution and patterns for all models
- **Performance Heatmap**: Normalized comparison matrix
- **Feature Importance**: Most influential factors for predictions

All visualizations are saved in the `outputs/images/` directory.

## Module Documentation

### DataProcessor (`src/data_processor.py`)
- Handles data loading, cleaning, and preprocessing
- Creates train/test splits and preprocessing pipelines
- Provides data exploration utilities

### StudentPerformanceModel (`src/model.py`)
- Implements Random Forest, Linear Regression, and Polynomial Regression models
- Supports regularization techniques (Ridge, Lasso) for polynomial features
- Handles model training, evaluation, and persistence
- Provides feature importance analysis and comprehensive performance metrics

### Visualizer (`src/visualizer.py`)
- Creates comprehensive data and model visualizations
- Automatically saves plots to the outputs directory
- Supports customizable plot styling and formats

## Future Improvements

- Implement additional algorithms (XGBoost, Neural Networks)
- Add hyperparameter tuning capabilities
- Include cross-validation and model selection
- Develop a web interface for predictions
- Add automated feature engineering

## License

This project is licensed under the MIT License - see the LICENSE file for details.