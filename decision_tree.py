"""Member 4: Decision Tree. Edit this estimator; keep the shared split fixed."""
from sklearn.tree import DecisionTreeClassifier
from evaluation import cli

def build_model():
    return DecisionTreeClassifier(random_state=42)

if __name__=='__main__':
    cli('decision_tree',build_model())
