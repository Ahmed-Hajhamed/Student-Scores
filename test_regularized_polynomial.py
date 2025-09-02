"""
Test polynomial regression with regularization to reduce overfitting.
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processor import DataProcessor


def test_regularized_polynomial():
    """Test polynomial regression with Ridge and Lasso regularization."""
    print("=== Regularized Polynomial Regression Test ===\n")
    
    # Load and prepare data
    data_processor = DataProcessor("data/StudentPerformanceFactors.csv")
    if not data_processor.load_data():
        print("Failed to load data. Exiting.")
        return
    
    X, y = data_processor.prepare_features()
    preprocessor = data_processor.create_preprocessor()
    X_train, X_test, y_train, y_test = data_processor.split_data(X, y)
    
    results = {}
    
    # Test different polynomial degrees with regularization
    degrees = [2, 3, 4]
    alphas = [0.1, 1.0, 10.0]
    
    for degree in degrees:
        print(f"\n--- Testing Polynomial Degree {degree} ---")
        
        for alpha in alphas:
            # Ridge Regression
            ridge_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('polynomial_features', PolynomialFeatures(degree=degree, include_bias=False)),
                ('regressor', Ridge(alpha=alpha))
            ])
            
            ridge_pipeline.fit(X_train, y_train)
            ridge_pred = ridge_pipeline.predict(X_test)
            ridge_r2 = r2_score(y_test, ridge_pred)
            ridge_mae = mean_absolute_error(y_test, ridge_pred)
            
            # Lasso Regression
            lasso_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('polynomial_features', PolynomialFeatures(degree=degree, include_bias=False)),
                ('regressor', Lasso(alpha=alpha, max_iter=2000))
            ])
            
            lasso_pipeline.fit(X_train, y_train)
            lasso_pred = lasso_pipeline.predict(X_test)
            lasso_r2 = r2_score(y_test, lasso_pred)
            lasso_mae = mean_absolute_error(y_test, lasso_pred)
            
            print(f"  Alpha {alpha:4.1f} - Ridge: R²={ridge_r2:.4f}, MAE={ridge_mae:.3f} | Lasso: R²={lasso_r2:.4f}, MAE={lasso_mae:.3f}")
            
            results[f"Ridge_d{degree}_a{alpha}"] = {'r2': ridge_r2, 'mae': ridge_mae}
            results[f"Lasso_d{degree}_a{alpha}"] = {'r2': lasso_r2, 'mae': lasso_mae}
    
    # Find best regularized model
    best_model = max(results.keys(), key=lambda k: results[k]['r2'])
    best_r2 = results[best_model]['r2']
    best_mae = results[best_model]['mae']
    
    print(f"\n=== BEST REGULARIZED MODEL ===")
    print(f"Model: {best_model}")
    print(f"R² Score: {best_r2:.4f}")
    print(f"MAE: {best_mae:.3f}")
    
    # Compare with baseline linear regression
    from sklearn.linear_model import LinearRegression
    
    baseline_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    baseline_pipeline.fit(X_train, y_train)
    baseline_pred = baseline_pipeline.predict(X_test)
    baseline_r2 = r2_score(y_test, baseline_pred)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    
    print(f"\nBaseline Linear Regression:")
    print(f"R² Score: {baseline_r2:.4f}")
    print(f"MAE: {baseline_mae:.3f}")
    
    if best_r2 > baseline_r2:
        print(f"\n✅ Regularized polynomial regression improved performance!")
        print(f"Improvement: +{(best_r2 - baseline_r2):.4f} R² points")
    else:
        print(f"\n❌ Baseline linear regression still performs better.")
        print(f"Difference: {(baseline_r2 - best_r2):.4f} R² points")


if __name__ == "__main__":
    test_regularized_polynomial()
