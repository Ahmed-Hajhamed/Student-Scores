"""
Data loading and preprocessing utilities for student performance prediction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Tuple, Optional


class DataProcessor:
    """Handle data loading, cleaning, and preprocessing operations."""
    
    def __init__(self, data_path: str = "data/StudentPerformanceFactors.csv"):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to the CSV data file
        """
        self.data_path = data_path
        self.dataset = None
        self.numeric_features = None
        self.categorical_features = None
        self.preprocessor = None
    
    def load_data(self) -> bool:
        """
        Load the dataset from CSV file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.dataset = pd.read_csv(self.data_path)
            print(f"Data loaded successfully: {self.dataset.shape[0]} rows, {self.dataset.shape[1]} columns")
            return True
        except PermissionError:
            print("Permission denied. Try running the script as administrator or check file permissions.")
            return False
        except FileNotFoundError:
            print(f"File not found at {self.data_path}. Please verify the path is correct.")
            return False
    
    def explore_data(self) -> None:
        """Print basic information about the dataset."""
        if self.dataset is None:
            print("Dataset not loaded. Call load_data() first.")
            return
        
        print("\n--- Dataset Overview ---")
        print(f"Shape: {self.dataset.shape}")
        print(f"\nColumn names: {list(self.dataset.columns)}")
        print(f"\nData types:\n{self.dataset.dtypes}")
        print(f"\nMissing values:\n{self.dataset.isnull().sum()}")
        print(f"\nBasic statistics:\n{self.dataset.describe()}")
    
    def prepare_features(self, target_column: str = 'Exam_Score') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.
        
        Args:
            target_column: Name of the target column
            
        Returns:
            Tuple of (features, target)
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        X = self.dataset.drop(columns=[target_column])
        y = self.dataset[target_column]
        
        # Identify numeric and categorical columns
        self.numeric_features = X.select_dtypes(exclude=['object']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Features: {len(X.columns)} total ({len(self.numeric_features)} numeric, "
              f"{len(self.categorical_features)} categorical)")
        
        return X, y
    
    def create_preprocessor(self) -> ColumnTransformer:
        """
        Create a preprocessing pipeline for features.
        
        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        if self.numeric_features is None or self.categorical_features is None:
            raise ValueError("Features not identified. Call prepare_features() first.")
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), self.categorical_features)
            ]
        )
        
        return self.preprocessor
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Features
            y: Target variable
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
