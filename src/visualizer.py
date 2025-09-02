"""
Visualization utilities for student performance analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import os


class Visualizer:
    """Handle data visualization and plotting."""
    
    def __init__(self, save_path: str = "outputs/images"):
        """
        Initialize the visualizer.
        
        Args:
            save_path: Directory to save plots
        """
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_data_distribution(self, data: pd.DataFrame, target_column: str = 'Exam_Score') -> None:
        """
        Plot the distribution of the target variable.
        
        Args:
            data: Dataset
            target_column: Target variable column name
        """
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(data[target_column], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel(target_column)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {target_column}')
        
        plt.subplot(1, 2, 2)
        plt.boxplot(data[target_column])
        plt.ylabel(target_column)
        plt.title(f'Box Plot of {target_column}')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/score_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self, data: pd.DataFrame) -> None:
        """
        Plot correlation matrix of numeric features.
        
        Args:
            data: Dataset
        """
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = numeric_data.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.1, cbar_kws={"shrink": .5})
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_actual_vs_predicted(self, y_actual: pd.Series, y_predicted: np.ndarray, 
                                metrics: dict) -> None:
        """
        Plot actual vs predicted values.
        
        Args:
            y_actual: Actual values
            y_predicted: Predicted values
            metrics: Performance metrics dictionary
        """
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(y_actual, y_predicted, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(y_actual.min(), y_predicted.min())
        max_val = max(y_actual.max(), y_predicted.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Exam Scores')
        plt.ylabel('Predicted Exam Scores')
        plt.title('Actual vs Predicted Exam Scores')
        plt.legend()
        
        # Add performance metrics
        textstr = f"MAE: {metrics['mae']:.2f}\\nRMSE: {metrics['rmse']:.2f}\\nR²: {metrics['r2']:.4f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/actual_vs_predicted.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_residuals(self, y_actual: pd.Series, y_predicted: np.ndarray) -> None:
        """
        Plot residuals analysis.
        
        Args:
            y_actual: Actual values
            y_predicted: Predicted values
        """
        residuals = y_actual - y_predicted
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals vs Predicted
        axes[0].scatter(y_predicted, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted Values')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Residuals')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/residuals.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 10) -> None:
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with feature names and importance scores
            top_n: Number of top features to display
        """
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, v in enumerate(top_features['importance']):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_categorical_analysis(self, data: pd.DataFrame, categorical_cols: list, 
                                target_column: str = 'Exam_Score') -> None:
        """
        Plot analysis of categorical variables vs target.
        
        Args:
            data: Dataset
            categorical_cols: List of categorical column names
            target_column: Target variable column name
        """
        n_cols = min(3, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(categorical_cols):
            if i < len(axes):
                sns.boxplot(data=data, x=col, y=target_column, ax=axes[i])
                axes[i].set_title(f'{target_column} by {col}')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(len(categorical_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/categorical_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
