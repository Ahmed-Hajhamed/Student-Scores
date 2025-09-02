"""
Example script showing how to use individual components of the student performance prediction system.
"""
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processor import DataProcessor
from src.model import StudentPerformanceModel
from src.visualizer import Visualizer


def example_basic_usage():
    """Example of basic usage with minimal code."""
    print("=== Basic Usage Example ===\\n")
    
    # 1. Load and prepare data
    processor = DataProcessor("data/StudentPerformanceFactors.csv")
    processor.load_data()
    X, y = processor.prepare_features()
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    
    # 2. Train a model
    model = StudentPerformanceModel(processor.create_preprocessor())
    model.create_random_forest_model()
    model.train(X_train, y_train)
    
    # 3. Evaluate the model
    metrics = model.evaluate(X_test, y_test)
    
    # 4. Make predictions
    predictions = model.predict(X_test)
    print(f"First 5 predictions: {predictions[:5]}")
    print(f"First 5 actual values: {y_test.iloc[:5].values}")


def example_custom_model():
    """Example of using custom model parameters."""
    print("\\n=== Custom Model Example ===\\n")
    
    # Load data
    processor = DataProcessor("data/StudentPerformanceFactors.csv")
    processor.load_data()
    X, y = processor.prepare_features()
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    
    # Create model with custom parameters
    model = StudentPerformanceModel(processor.create_preprocessor())
    model.create_random_forest_model(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    
    # Show feature importance
    importance = model.get_feature_importance()
    print("\\nTop 5 Most Important Features:")
    print(importance.head())


def example_visualization_only():
    """Example of using only the visualization components."""
    print("\\n=== Visualization Only Example ===\\n")
    
    # Load data
    processor = DataProcessor("data/StudentPerformanceFactors.csv")
    processor.load_data()
    
    # Create visualizer
    visualizer = Visualizer("outputs/examples")
    
    # Generate visualizations
    visualizer.plot_data_distribution(processor.dataset)
    visualizer.plot_correlation_matrix(processor.dataset)
    
    print("Visualizations saved to outputs/examples/")


def example_model_persistence():
    """Example of saving and loading models."""
    print("\\n=== Model Persistence Example ===\\n")
    
    # Train a model
    processor = DataProcessor("data/StudentPerformanceFactors.csv")
    processor.load_data()
    X, y = processor.prepare_features()
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    
    model = StudentPerformanceModel(processor.create_preprocessor())
    model.create_random_forest_model()
    model.train(X_train, y_train)
    
    # Save the model
    model.save_model("outputs/models/example_model.pkl")
    
    # Load the model (in a new instance)
    new_model = StudentPerformanceModel(processor.create_preprocessor())
    new_model.load_model("outputs/models/example_model.pkl")
    
    # Use the loaded model
    predictions = new_model.predict(X_test)
    print(f"Loaded model predictions (first 3): {predictions[:3]}")


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_custom_model()
    example_visualization_only()
    example_model_persistence()
    
    print("\\n=== All Examples Completed ===")
    print("Check the outputs folder for generated files.")
