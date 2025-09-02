# Polynomial Regression Analysis Report

## Overview
This report presents the results of implementing and comparing polynomial regression with existing linear regression and random forest models for student performance prediction.

## Models Compared
1. **Linear Regression** - Simple linear model
2. **Polynomial Regression (degree 2)** - Quadratic features
3. **Polynomial Regression (degree 3)** - Cubic features  
4. **Random Forest** - Ensemble tree-based model

## Results Summary

| Model | MAE | RMSE | R² Score | Rank |
|-------|-----|------|----------|------|
| Linear Regression | 0.45 | 1.80 | 0.7699 | 🥇 1st |
| Polynomial Regression (d=2) | 0.66 | 1.90 | 0.7445 | 🥈 2nd |
| Random Forest | 1.23 | 2.25 | 0.6415 | 🥉 3rd |
| Polynomial Regression (d=3) | 2.77 | 4.34 | -0.3296 | ❌ 4th |

## Key Findings

### 1. Linear Regression Performs Best
- **Highest R² score**: 0.7699 (explains ~77% of variance)
- **Lowest MAE**: 0.45 points average error
- **Lowest RMSE**: 1.80 points
- Simple and interpretable model

### 2. Polynomial Regression Analysis
- **Degree 2 (Quadratic)**: 
  - Performs reasonably well (R² = 0.7445)
  - Slightly worse than linear regression
  - Shows some overfitting compared to linear model
  
- **Degree 3 (Cubic)**:
  - **Severe overfitting** with negative R² score (-0.3296)
  - Very high error rates (MAE = 2.77, RMSE = 4.34)
  - Not suitable for this dataset

### 3. Model Complexity Analysis
- **Increasing polynomial degree led to overfitting**
- The dataset appears to have mostly linear relationships
- Higher-order polynomials captured noise rather than signal

### 4. Random Forest Performance
- Moderate performance (R² = 0.6415)
- Higher error rates than linear models
- May benefit from hyperparameter tuning

## Recommendations

### For Production Use
1. **Use Linear Regression** as the primary model
   - Best performance across all metrics
   - Simple, fast, and interpretable
   - Low risk of overfitting

### For Further Research
1. **Feature Engineering**: Focus on creating meaningful linear combinations
2. **Regularization**: Try Ridge/Lasso regression for polynomial features
3. **Cross-validation**: Implement k-fold CV for more robust evaluation
4. **Hyperparameter Tuning**: Optimize Random Forest parameters

## Technical Insights

### Why Polynomial Regression Struggled
1. **Limited Training Data**: 5,285 samples may not be enough for high-degree polynomials
2. **High Dimensionality**: Polynomial features create many new features, increasing overfitting risk
3. **Linear Relationships**: Student performance factors may have predominantly linear relationships

### Overfitting Evidence
- Polynomial degree 3 showed classic overfitting symptoms:
  - Negative R² score on test data
  - Extremely high error rates
  - Poor generalization capability

## Visualizations Generated
1. **Model Performance Comparison** - Bar charts of all metrics
2. **Actual vs Predicted Scatter Plots** - For each model
3. **Residuals Analysis** - Error distribution patterns
4. **Performance Heatmap** - Normalized comparison matrix

## Conclusion
Linear regression proves to be the optimal choice for this student performance prediction task. While polynomial regression can capture non-linear relationships, it introduced overfitting without providing meaningful performance improvements for this particular dataset. The results suggest that student performance factors have predominantly linear relationships that are well-captured by the simple linear model.

---
*Generated on: September 2, 2025*
*Models saved in: `/outputs/models/`*
*Visualizations saved in: `/outputs/images/`*
