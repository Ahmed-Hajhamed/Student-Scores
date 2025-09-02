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
- **Multiple Models**: Supports Random Forest and Linear Regression
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

This will:
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

The models achieve the following performance metrics:

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Random Forest | ~1.08 | ~1.45 | ~0.67 |
| Linear Regression | ~1.15 | ~1.52 | ~0.62 |

## Visualizations

The project automatically generates several visualizations:

- **Data Distribution**: Histogram and box plot of exam scores
- **Correlation Matrix**: Heatmap showing feature relationships
- **Model Performance**: Actual vs predicted scatter plots
- **Residuals Analysis**: Error distribution and patterns
- **Feature Importance**: Most influential factors for predictions

## Module Documentation

### DataProcessor (`src/data_processor.py`)
- Handles data loading, cleaning, and preprocessing
- Creates train/test splits and preprocessing pipelines
- Provides data exploration utilities

### StudentPerformanceModel (`src/model.py`)
- Implements Random Forest and Linear Regression models
- Handles model training, evaluation, and persistence
- Provides feature importance analysis

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