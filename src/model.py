"""
Machine learning models for student performance prediction.
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
import joblib
from typing import Tuple, Dict, Any
import os


class StudentPerformanceModel:
    """Machine learning model for predicting student performance."""
    
    def __init__(self, preprocessor: ColumnTransformer):
        """
        Initialize the model.
        
        Args:
            preprocessor: Preprocessing pipeline for features
        """
        self.preprocessor = preprocessor
        self.model = None
        self.pipeline = None
        self.is_trained = False
    
    def create_random_forest_model(self, **kwargs) -> Pipeline:
        """
        Create a Random Forest model pipeline.
        
        Args:
            **kwargs: Parameters for RandomForestRegressor
            
        Returns:
            Pipeline: Complete ML pipeline
        """
        # Default parameters
        rf_params = {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }
        rf_params.update(kwargs)
        
        self.model = RandomForestRegressor(**rf_params)
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', self.model)
        ])
        
        return self.pipeline
    
    def create_linear_model(self) -> Pipeline:
        """
        Create a Linear Regression model pipeline.
        
        Returns:
            Pipeline: Complete ML pipeline
        """
        self.model = LinearRegression()
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', self.model)
        ])
        
        return self.pipeline
    
    def create_polynomial_model(self, degree: int = 2, **kwargs) -> Pipeline:
        """
        Create a Polynomial Regression model pipeline.
        
        Args:
            degree: Degree of polynomial features (default: 2)
            **kwargs: Additional parameters for LinearRegression
            
        Returns:
            Pipeline: Complete ML pipeline with polynomial features
        """
        lr_params = {
            'fit_intercept': True
        }
        lr_params.update(kwargs)
        
        self.model = LinearRegression(**lr_params)
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('polynomial_features', PolynomialFeatures(degree=degree, include_bias=False)),
            ('regressor', self.model)
        ])
        
        return self.pipeline
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        if self.pipeline is None:
            raise ValueError("Model not created. Call create_random_forest_model() or create_linear_model() first.")
        
        print(f"Training {type(self.model).__name__} model...")
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        print("Training completed.")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.pipeline.predict(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dict: Performance metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        print(f"\n--- Model Performance ---")
        print(f"Mean Absolute Error (MAE): {metrics['mae']:.2f}")
        print(f"Mean Squared Error (MSE): {metrics['mse']:.2f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.2f}")
        print(f"R² Score: {metrics['r2']:.4f}")
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (for tree-based models).
        
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not support feature importance.")
        
        # Get feature names after preprocessing
        feature_names = self._get_feature_names()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def _get_feature_names(self) -> list:
        """Get feature names after preprocessing."""
        feature_names = []
        
        # Get transformer names and their corresponding features
        for name, transformer, features in self.preprocessor.transformers_:
            if name == 'cat' and hasattr(transformer, 'get_feature_names_out'):
                feature_names.extend(transformer.get_feature_names_out(features))
            else:
                feature_names.extend(features)
        
        # Handle polynomial features if they exist in the pipeline
        if hasattr(self.pipeline, 'named_steps') and 'polynomial_features' in self.pipeline.named_steps:
            poly_transformer = self.pipeline.named_steps['polynomial_features']
            if hasattr(poly_transformer, 'get_feature_names_out'):
                feature_names = poly_transformer.get_feature_names_out(feature_names)
        
        return feature_names
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(self.pipeline, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        self.pipeline = joblib.load(filepath)
        self.model = self.pipeline.named_steps['regressor']
        self.is_trained = True
        print(f"Model loaded from: {filepath}")
