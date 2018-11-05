import numpy as np
import faiss

d = 64                           # dimension
nb = 100000                      # database size
nq = 1                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.










index = faiss.IndexFlatL2(d)
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

k = 4                           # we want to see 4 nearest neighbors
D, I = index.search(xb[:1], k)  # sanity check
#print('search vector:')
#print(xb[:1])
print('2.nd: I contains the nearest neighbors for each query vector:')
print(I)
print('D contains the squared L2 distances:')
print(D)
D, I = index.search(xb[392:393], k)  # sanity check
print('393.rd: I contains the nearest neighbors for each query vector:')
print(I)
print('D contains the squared L2 distances:')
print(D)

print('actual search:')
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries