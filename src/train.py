from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from data import load_dataset, split_data
from features import build_preprocessor
from models import get_model, get_params
import joblib
import os

def train_model(model_name='random_forest', use_pca=False, use_poly=False, 
                n_components=5, degree=2, tune_hyperparams=False):
    X, y, features = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor(
        use_pca=use_pca,
        use_poly=use_poly,
        n_components=n_components,
        degree=degree
    )
    model = get_model(model_name)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # FIT THE PIPELINE! This was missing
    print(f"\n🔧 Training {model_name}...")
    pipeline.fit(X_train, y_train)
    print("✅ Training completed!")

    return pipeline

if __name__ == "__main__":
    print("\n🧪 TESTING TRAINING MODULE\n")
    
    # Test 1: Train Random Forest without tuning
    print("\n--- TEST 1: Random Forest (No Tuning) ---")
    pipeline1 = train_model(
        model_name='random_forest',
        use_pca=False,
        use_poly=False,
        tune_hyperparams=False
    )
    
    print("\n✅ Training test completed!")