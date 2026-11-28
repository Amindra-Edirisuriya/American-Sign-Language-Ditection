"""Build connected groups BEFORE deduplication so removed rows cannot break links.
Only numeric vectors determine exact equality. Same-class timestamp links use <= gap_ms.
These are proxy groups, not verified signer/session identities.
"""
import numpy as np

class UnionFind:
    def __init__(self,n):
        self.parent=np.arange(n,dtype=np.int64)
    def find(self,a):
        while self.parent[a]!=a:
            self.parent[a]=self.parent[self.parent[a]]
            a=int(self.parent[a])
        return int(a)
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)

def make_groups(X,y,t,gap_ms=1000):
    if not isinstance(gap_ms,int) or gap_ms<0:raise ValueError('gap_ms must be a nonnegative integer')
    n=len(y)
    if X.shape!=(n,63) or t.shape!=(n,) or not np.isfinite(X).all():
        raise ValueError('Invalid sample arrays')
    uf=UnionFind(n);vectors={}
    canonical=np.ascontiguousarray(X,dtype=np.float64).copy()
    canonical[canonical==0]=0  # +0 and -0 represent the same coordinate.
    for i,row in enumerate(canonical):
        key=row.tobytes()
        if key in vectors:uf.union(i,vectors[key][0])
        vectors.setdefault(key,[]).append(i)
    temporal_edges=0
    for label in np.unique(y):
        idx=np.flatnonzero((y==label)&(t>=0))
        idx=idx[np.argsort(t[idx],kind='stable')]
        for a,b in zip(idx[:-1],idx[1:]):
            if int(t[b])-int(t[a])<=gap_ms:
                uf.union(a,b);temporal_edges+=1
    groups=np.array([uf.find(i) for i in range(n)],dtype=np.int64)
    keep=[];reason=np.full(n,'retained',dtype='U32')
    representative=np.arange(n,dtype=np.int64)
    conflicts=0;repeat_rows=0
    for idx in vectors.values():
        representative[idx]=idx[0]
        if len(set(y[idx]))>1:
            reason[idx]='conflicting_label_vector';conflicts+=len(idx)
        else:
            keep.append(idx[0]);reason[idx[1:]]='exact_duplicate_removed';repeat_rows+=len(idx)-1
    keep=np.asarray(sorted(keep),dtype=np.int64)
    stats=dict(original_samples=n,retained_samples=len(keep),exact_duplicate_rows_removed=repeat_rows,
               conflicting_label_rows_removed=conflicts,unique_numeric_vectors=len(vectors),
               same_class_temporal_edges=temporal_edges,retained_groups=len(np.unique(groups[keep])))
    return groups,keep,reason,representative,stats
