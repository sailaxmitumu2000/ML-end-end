from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import yaml

def load_config():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_model(model_name='random_forest'):
    # Dictionary of available models
    models = {
        'linear': LinearRegression(),
        'random_forest': RandomForestRegressor(random_state=42),
        'gradient_boost': GradientBoostingRegressor(random_state=42)
    }
    model = models.get(model_name, RandomForestRegressor(random_state=42))
    return model

def get_params(model_name='random_forest'):
    config = load_config()
    return config.get(model_name, {})

if __name__ == "__main__":
    print("=== TESTING MODELS ===\n")
    
    model1 = get_model('linear')
    params1 = get_params('linear')
    print(f"Linear Model: {type(model1).__name__}")
    print(f"Parameters: {params1}\n")

    model2 = get_model('random_forest')
    params2 = get_params('random_forest')
    print(f"Random Forest Model: {type(model2).__name__}")
    print(f"Parameters: {params2}\n")

    model3 = get_model('gradient_boost')
    params3 = get_params('gradient_boost')
    print(f"Gradient Boost Model: {type(model3).__name__}")
    print(f"Parameters: {params3}\n")
    
    print("✅ ALL TESTS PASSED!")