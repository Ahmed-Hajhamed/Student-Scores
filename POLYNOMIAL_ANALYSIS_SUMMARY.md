# New Files Added for Polynomial Regression Analysis

## Summary of Changes

This document summarizes the new files and modifications made to implement and evaluate polynomial regression in the Student Performance Prediction project.

## New Files Created

### 1. `model_comparison.py`
**Purpose**: Comprehensive comparison of Linear, Polynomial (degree 2 & 3), and Random Forest models

**Features**:
- Trains and evaluates all four models
- Generates detailed comparison visualizations
- Creates performance summary tables
- Saves all models for future use

**Key Outputs**:
- Model performance metrics comparison
- Actual vs Predicted plots for all models
- Residuals analysis for all models
- Normalized performance heatmap

### 2. `test_regularized_polynomial.py`
**Purpose**: Tests polynomial regression with Ridge and Lasso regularization

**Features**:
- Tests degrees 2, 3, and 4 with different alpha values
- Compares Ridge vs Lasso regularization
- Identifies best regularized model
- Compares against baseline linear regression

**Key Findings**:
- Ridge with degree 2 and alpha=10.0 performed best among regularized models
- Still couldn't surpass baseline linear regression performance

### 3. `polynomial_regression_report.md`
**Purpose**: Detailed analysis report of polynomial regression experiments

**Contents**:
- Complete results summary
- Key findings and insights
- Technical analysis of overfitting
- Recommendations for future work
- Explanation of why polynomial regression struggled

## Modified Files

### 1. `src/model.py`
**Added Features**:
- `create_polynomial_model()` method for polynomial regression
- Support for PolynomialFeatures preprocessing
- Enhanced `_get_feature_names()` to handle polynomial features
- Import of PolynomialFeatures from sklearn

### 2. `README.md`
**Updated Sections**:
- Features list to include polynomial regression and regularization
- Usage section with new script commands
- Model Performance section with comprehensive comparison table
- Visualizations section with new plot types
- Module documentation updates

## Generated Visualizations

### New Visualization Files in `outputs/images/`:
1. `model_comparison.png` - Bar charts comparing all metrics
2. `actual_vs_predicted_comparison.png` - Side-by-side scatter plots
3. `residuals_comparison.png` - Residuals analysis for all models

### Existing Files (Enhanced):
- All original visualization files remain and are enhanced by the new comparison framework

## Saved Models

### New Model Files in `outputs/models/`:
1. `polynomial_regression_d2_model.pkl` - Degree 2 polynomial model
2. `polynomial_regression_d3_model.pkl` - Degree 3 polynomial model

### Existing Model Files (Updated):
- `linear_regression_model.pkl` - Updated with latest training
- `random_forest_model.pkl` - Updated with latest training

## Key Results

### Performance Ranking:
1. **Linear Regression**: R² = 0.7699, MAE = 0.45 ✅ **Best**
2. **Polynomial (d=2)**: R² = 0.7445, MAE = 0.66 
3. **Random Forest**: R² = 0.6415, MAE = 1.23
4. **Polynomial (d=3)**: R² = -0.3296, MAE = 2.77 ❌ **Overfitted**

### Regularization Results:
- **Best Regularized**: Ridge (d=2, α=10.0) with R² = 0.7493, MAE = 0.622
- **Still inferior** to baseline Linear Regression

## Conclusions

1. **Linear relationships dominate** the student performance dataset
2. **Polynomial regression introduces overfitting** without meaningful performance gains
3. **Regularization helps** but cannot overcome the fundamental linear nature of the data
4. **Simple linear regression remains the best choice** for this problem

## Usage Instructions

To reproduce these results:

```bash
# Run comprehensive model comparison
python model_comparison.py

# Test regularized polynomial regression
python test_regularized_polynomial.py

# Run original workflow (updated)
python main.py
```

---
*Analysis completed: September 2, 2025*
