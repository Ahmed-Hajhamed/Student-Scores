"""
Main script for student performance prediction.
This script demonstrates the complete workflow from data loading to model evaluation.
"""
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processor import DataProcessor
from src.model import StudentPerformanceModel
from src.visualizer import Visualizer


def main():
    """Main function to run the complete workflow."""
    print("=== Student Performance Prediction Workflow ===\\n")
    
    # 1. Data Loading and Preprocessing
    print("1. Loading and preprocessing data...")
    data_processor = DataProcessor("data/StudentPerformanceFactors.csv")
    
    if not data_processor.load_data():
        print("Failed to load data. Exiting.")
        return
    
    # Explore the data
    data_processor.explore_data()
    
    # Prepare features
    X, y = data_processor.prepare_features()
    
    # Create preprocessor
    preprocessor = data_processor.create_preprocessor()
    
    # Split data
    X_train, X_test, y_train, y_test = data_processor.split_data(X, y)
    
    # 2. Data Visualization
    print("\\n2. Creating data visualizations...")
    visualizer = Visualizer()
    
    # Plot data distribution
    visualizer.plot_data_distribution(data_processor.dataset)
    
    # Plot correlation matrix
    visualizer.plot_correlation_matrix(data_processor.dataset)
    
    # Plot categorical analysis
    if data_processor.categorical_features:
        visualizer.plot_categorical_analysis(
            data_processor.dataset, 
            data_processor.categorical_features
        )
    
    # 3. Model Training and Evaluation
    print("\\n3. Training and evaluating models...")
    
    # Random Forest Model
    print("\\n--- Random Forest Model ---")
    rf_model = StudentPerformanceModel(preprocessor)
    rf_model.create_random_forest_model(n_estimators=100, max_depth=10)
    rf_model.train(X_train, y_train)
    rf_metrics = rf_model.evaluate(X_test, y_test)
    
    # Get predictions for visualization
    rf_predictions = rf_model.predict(X_test)
    
    # 4. Model Visualization
    print("\\n4. Creating model performance visualizations...")
    
    # Plot actual vs predicted
    visualizer.plot_actual_vs_predicted(y_test, rf_predictions, rf_metrics)
    
    # Plot residuals
    visualizer.plot_residuals(y_test, rf_predictions)
    
    # Plot feature importance
    try:
        importance_df = rf_model.get_feature_importance()
        visualizer.plot_feature_importance(importance_df)
        print("\\nTop 10 Most Important Features:")
        print(importance_df.head(10))
    except Exception as e:
        print(f"Could not generate feature importance: {e}")
    
    # 5. Save the model
    print("\\n5. Saving the trained model...")
    rf_model.save_model("outputs/models/random_forest_model.pkl")
    
    # 6. Linear Regression Model (for comparison)
    print("\\n--- Linear Regression Model (for comparison) ---")
    lr_model = StudentPerformanceModel(preprocessor)
    lr_model.create_linear_model()
    lr_model.train(X_train, y_train)
    lr_metrics = lr_model.evaluate(X_test, y_test)
    
    # Save linear model too
    lr_model.save_model("outputs/models/linear_regression_model.pkl")
    
    # 7. Summary
    print("\\n=== SUMMARY ===")
    print(f"Random Forest - MAE: {rf_metrics['mae']:.2f}, R²: {rf_metrics['r2']:.4f}")
    print(f"Linear Regression - MAE: {lr_metrics['mae']:.2f}, R²: {lr_metrics['r2']:.4f}")
    
    if rf_metrics['r2'] > lr_metrics['r2']:
        print("\\nRandom Forest performs better!")
    else:
        print("\\nLinear Regression performs better!")
    
    print("\\nWorkflow completed successfully!")
    print("Check the 'outputs' folder for saved models and visualizations.")


if __name__ == "__main__":
    main()
