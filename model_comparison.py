"""
Comprehensive model comparison including Polynomial Regression.
This script compares Linear Regression, Polynomial Regression, and Random Forest models.
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processor import DataProcessor
from src.model import StudentPerformanceModel
from src.visualizer import Visualizer


def compare_models():
    """Compare Linear, Polynomial, and Random Forest models."""
    print("=== Comprehensive Model Comparison ===\n")
    
    # 1. Data Loading and Preprocessing
    print("1. Loading and preprocessing data...")
    data_processor = DataProcessor("data/StudentPerformanceFactors.csv")
    
    if not data_processor.load_data():
        print("Failed to load data. Exiting.")
        return
    
    # Prepare features and split data
    X, y = data_processor.prepare_features()
    preprocessor = data_processor.create_preprocessor()
    X_train, X_test, y_train, y_test = data_processor.split_data(X, y)
    
    # 2. Initialize models
    models = {}
    results = {}
    predictions = {}
    
    print("\n2. Training and evaluating models...")
    
    # Linear Regression
    print("\n--- Linear Regression ---")
    lr_model = StudentPerformanceModel(preprocessor)
    lr_model.create_linear_model()
    lr_model.train(X_train, y_train)
    lr_metrics = lr_model.evaluate(X_test, y_test)
    models['Linear Regression'] = lr_model
    results['Linear Regression'] = lr_metrics
    predictions['Linear Regression'] = lr_model.predict(X_test)
    
    # Polynomial Regression (degree 2)
    print("\n--- Polynomial Regression (degree 2) ---")
    poly2_model = StudentPerformanceModel(preprocessor)
    poly2_model.create_polynomial_model(degree=2)
    poly2_model.train(X_train, y_train)
    poly2_metrics = poly2_model.evaluate(X_test, y_test)
    models['Polynomial Regression (d=2)'] = poly2_model
    results['Polynomial Regression (d=2)'] = poly2_metrics
    predictions['Polynomial Regression (d=2)'] = poly2_model.predict(X_test)
    
    # Polynomial Regression (degree 3)
    print("\n--- Polynomial Regression (degree 3) ---")
    poly3_model = StudentPerformanceModel(preprocessor)
    poly3_model.create_polynomial_model(degree=3)
    poly3_model.train(X_train, y_train)
    poly3_metrics = poly3_model.evaluate(X_test, y_test)
    models['Polynomial Regression (d=3)'] = poly3_model
    results['Polynomial Regression (d=3)'] = poly3_metrics
    predictions['Polynomial Regression (d=3)'] = poly3_model.predict(X_test)
    
    # Random Forest
    print("\n--- Random Forest ---")
    rf_model = StudentPerformanceModel(preprocessor)
    rf_model.create_random_forest_model(n_estimators=100, max_depth=10)
    rf_model.train(X_train, y_train)
    rf_metrics = rf_model.evaluate(X_test, y_test)
    models['Random Forest'] = rf_model
    results['Random Forest'] = rf_metrics
    predictions['Random Forest'] = rf_model.predict(X_test)
    
    # 3. Create comparison DataFrame
    comparison_df = pd.DataFrame(results).T
    print("\n=== MODEL COMPARISON SUMMARY ===")
    print(comparison_df.round(4))
    
    # 4. Visualizations
    print("\n3. Creating comparison visualizations...")
    
    # Create comparison plots
    create_comparison_plots(y_test, predictions, results, comparison_df)
    
    # 5. Save models
    print("\n4. Saving models...")
    lr_model.save_model("outputs/models/linear_regression_model.pkl")
    poly2_model.save_model("outputs/models/polynomial_regression_d2_model.pkl")
    poly3_model.save_model("outputs/models/polynomial_regression_d3_model.pkl")
    rf_model.save_model("outputs/models/random_forest_model.pkl")
    
    # 6. Find best model
    best_model_name = comparison_df['r2'].idxmax()
    best_r2 = comparison_df.loc[best_model_name, 'r2']
    
    print(f"\n=== BEST MODEL ===")
    print(f"Best performing model: {best_model_name}")
    print(f"R² Score: {best_r2:.4f}")
    print(f"MAE: {comparison_df.loc[best_model_name, 'mae']:.2f}")
    print(f"RMSE: {comparison_df.loc[best_model_name, 'rmse']:.2f}")
    
    return models, results, comparison_df


def create_comparison_plots(y_test, predictions, results, comparison_df):
    """Create comprehensive comparison plots."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Model Performance Comparison Bar Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # R² Score comparison
    models = list(results.keys())
    r2_scores = [results[model]['r2'] for model in models]
    
    axes[0, 0].bar(range(len(models)), r2_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0, 0].set_title('R² Score Comparison', fontweight='bold')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_xticks(range(len(models)))
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(r2_scores):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # MAE comparison
    mae_scores = [results[model]['mae'] for model in models]
    axes[0, 1].bar(range(len(models)), mae_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0, 1].set_title('Mean Absolute Error Comparison', fontweight='bold')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_xticks(range(len(models)))
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(mae_scores):
        axes[0, 1].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # RMSE comparison
    rmse_scores = [results[model]['rmse'] for model in models]
    axes[1, 0].bar(range(len(models)), rmse_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[1, 0].set_title('Root Mean Squared Error Comparison', fontweight='bold')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_xticks(range(len(models)))
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(rmse_scores):
        axes[1, 0].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Combined metrics heatmap
    metrics_for_heatmap = comparison_df[['mae', 'rmse', 'r2']].copy()
    # Normalize metrics for better visualization (inverse for mae and rmse since lower is better)
    metrics_for_heatmap['mae_norm'] = 1 - (metrics_for_heatmap['mae'] / metrics_for_heatmap['mae'].max())
    metrics_for_heatmap['rmse_norm'] = 1 - (metrics_for_heatmap['rmse'] / metrics_for_heatmap['rmse'].max())
    metrics_for_heatmap['r2_norm'] = metrics_for_heatmap['r2'] / metrics_for_heatmap['r2'].max()
    
    heatmap_data = metrics_for_heatmap[['mae_norm', 'rmse_norm', 'r2_norm']].T
    heatmap_data.columns = models
    heatmap_data.index = ['MAE (inverted)', 'RMSE (inverted)', 'R² Score']
    
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0.5, 
                ax=axes[1, 1], cbar_kws={'label': 'Normalized Performance'})
    axes[1, 1].set_title('Normalized Performance Heatmap', fontweight='bold')
    axes[1, 1].set_ylabel('Metrics')
    
    plt.tight_layout()
    plt.savefig('outputs/images/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Actual vs Predicted for all models
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Actual vs Predicted Values Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, (model_name, pred) in enumerate(predictions.items()):
        row = i // 2
        col = i % 2
        
        axes[row, col].scatter(y_test, pred, alpha=0.6, color=colors[i], s=50)
        axes[row, col].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                           'r--', lw=2, label='Perfect Prediction')
        axes[row, col].set_xlabel('Actual Values')
        axes[row, col].set_ylabel('Predicted Values')
        axes[row, col].set_title(f'{model_name}\nR² = {results[model_name]["r2"]:.4f}', fontweight='bold')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/images/actual_vs_predicted_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Residuals Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Residuals Analysis Comparison', fontsize=16, fontweight='bold')
    
    for i, (model_name, pred) in enumerate(predictions.items()):
        row = i // 2
        col = i % 2
        
        residuals = y_test - pred
        
        axes[row, col].scatter(pred, residuals, alpha=0.6, color=colors[i], s=50)
        axes[row, col].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[row, col].set_xlabel('Predicted Values')
        axes[row, col].set_ylabel('Residuals')
        axes[row, col].set_title(f'{model_name} Residuals\nMAE = {results[model_name]["mae"]:.2f}', fontweight='bold')
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/images/residuals_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparison plots saved to outputs/images/")


if __name__ == "__main__":
    models, results, comparison_df = compare_models()
