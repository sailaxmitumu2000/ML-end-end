
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
def evaluate_model(pipeline, X_test, y_test):
    print("\n🔮 Making predictions...")
    y_pred = pipeline.predict(X_test)
    print(f"✅ Predicted {len(y_pred)} samples")
    
    # Calculate metrics
    print("\n📈 Calculating metrics...")
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    print("EVALUATION RESULTS:")
    print("─" * 60)
    print(f"  📉 RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  📉 MAE  (Mean Absolute Error):     {mae:.4f}")
    print(f"  📈 R²   (R-Squared Score):         {r2:.4f}")
    print("─" * 60)
    
    print("\n💡 What these metrics mean:")
    print(f"  • RMSE: Average prediction error is ${rmse:.2f} (in $100,000s)")
    print(f"  • MAE:  Typical error is ${mae:.2f}")
    print(f"  • R²:   Model explains {r2*100:.1f}% of variance")
    
    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    from data import load_dataset, split_data
    from train import train_model
    
    print("\n🧪 TESTING EVALUATION MODULE\n")
    print("--- Step 1: Training a model ---")
    pipeline = train_model(
        model_name='random_forest',
        use_pca=False,
        use_poly=False,
        tune_hyperparams=False
    )
    
    # Load test data
    print("\n--- Step 2: Loading test data ---")
    X, y, features = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Evaluate
    print("\n--- Step 3: Evaluating model ---")
    metrics = evaluate_model(pipeline, X_test, y_test)
    print("\n✅ Evaluation test completed!")
    print(f"\n📊 Final Metrics: {metrics}")
    