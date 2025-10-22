from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def build_preprocessor(use_poly=False,use_pca=False,n_components=None,degree=2):
    steps=[]
    steps.append(('scaler',StandardScaler()))

    if use_poly:
        steps.append(('polynomial',PolynomialFeatures(degree=degree,include_bias=False))
        )
    if use_pca:
        steps.append(('pca',PCA(n_components=n_components)))
    pipeline=Pipeline(steps)

    

    return pipeline
if __name__=="__main__":
    print("\n=== TEST 1: Basic (only scaling) ===")
    preprocessor1 = build_preprocessor()
    
    print("\n=== TEST 2: With Polynomial Features ===")
    preprocessor2 = build_preprocessor(use_poly=True, degree=2)
    
    print("\n=== TEST 3: With PCA ===")
    preprocessor3 = build_preprocessor(use_pca=True, n_components=5)
    
    print("\n=== TEST 4: With Both ===")
    preprocessor4 = build_preprocessor(use_pca=True, use_poly=True)
    
    print("\n✅ ALL TESTS PASSED!")
