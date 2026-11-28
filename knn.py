"""Member 3: K nearest neighbours. Scaling is handled by evaluation.py."""
from sklearn.neighbors import KNeighborsClassifier
from evaluation import cli

def build_model():
    return KNeighborsClassifier(n_neighbors=3,n_jobs=2)

if __name__=='__main__':
    cli('knn',build_model())
