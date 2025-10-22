from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

def load_dataset():
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target
    feature_names = housing.feature_names
    return X, y, feature_names
def split_data(X,y,test_size=0.2,random_state=42):
    y_binned=pd.cut(y,bins=5,labels=False)
    X_train,X_test, y_train, y_test=train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y_binned)
    return X_train,X_test,y_train,y_test

if __name__ == "__main__":
    X, y, names = load_dataset()
    X_train,X_test, y_train, y_test=split_data(X,y)
    print("✅ Data loaded successfully!")
    print(f"Feature names: {names}")
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X-test: {X_test.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of y_test: {y_test.shape}")


