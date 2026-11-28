"""Member 2: Random Forest. Edit this estimator; keep the shared split fixed."""
from sklearn.ensemble import RandomForestClassifier
from evaluation import cli

def build_model():
    # 100 trees; deterministic seed. Two workers avoid exhausting group laptops.
    return RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=2)

if __name__=='__main__':
    cli('random_forest',build_model())
