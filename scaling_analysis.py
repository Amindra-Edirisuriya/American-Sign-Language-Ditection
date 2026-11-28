import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from common import ROOT, load_prepared

# Load prepared dataset
X,y,ids,t,g,k,tr,va,te,meta = load_prepared(ROOT/'prepared')

# Before scaling
before = pd.DataFrame(X[tr])

# After scaling
scaler = StandardScaler()
after = pd.DataFrame(scaler.fit_transform(X[tr]))

# Feature names
before.columns = after.columns = [
    'Feature_{}'.format(i + 1) for i in range(X.shape[1])
]

# Generate comparison table
summary = pd.DataFrame({
    'Before Mean': before.mean(),
    'Before Std': before.std(ddof=0),
    'After Mean': after.mean(),
    'After Std': after.std(ddof=0)
})

print(summary.round(4))
summary.to_csv('scaling_summary.csv')

# Generate graphs
for name, data in [('Before', before), ('After', after)]:
    plt.figure(figsize=(12, 5))
    plt.boxplot(data.iloc[:, :10].values)
    plt.title('Feature Distribution {} Scaling'.format(name))
    plt.xlabel('Feature')
    plt.ylabel('Value')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('{}_scaling.png'.format(name.lower()), dpi=300)
    plt.show()
