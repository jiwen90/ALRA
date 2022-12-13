import scipy
import anndata
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

filename = "counts" # h5ad filename here

print("Reading data")

# format obs x var, UMI raw counts
# must be QC filtered (no cells with 0 expression)
data = anndata.read_h5ad(filename + ".h5ad")
genes = data.var.index.to_numpy()
data = scipy.sparse.csr_matrix.toarray(data.X).T
cells_threshold = 0

# remove genes with expression in less than threshold cells
if cells_threshold:
    print(f"Removing genes with expression in less than {cells_threshold} cells")
    counts = np.sum(data != 0, axis=1)
    to_keep = counts >= cells_threshold
    removed = genes.size - np.sum(to_keep)
    data = data[to_keep]
    genes = genes[to_keep]
    counts = counts[to_keep]
    print(f"Removed {removed} genes")

print("Performing log normalization")
counts = np.sum(data, axis=0) # counts per cell
data = np.log1p(data / counts * 1e4)

print("Approximating rank k")
K = 100
if K > min(data.shape):
    K = min(data.shape)
    print("Warning: For best performance, we recommend using ALRA on expression matrices larger than 100 by 100")

noise = K - 20

_, s, _ = randomized_svd(data, n_components=K, random_state=None, n_iter=2)

diffs = s[:-1] - s[1:]
mu = np.mean(diffs[noise-1:])
sigma = np.std(diffs[noise-1:])
num_of_sds = (diffs - mu) / sigma
thresh = 6
k = np.max(np.argwhere(num_of_sds > thresh)) + 1 # python is 0 indexed

print(f"Using rank {k}")

### compute rank k approximation

U, s, Vh = randomized_svd(data, n_components=k, random_state=None, n_iter=2)

print("Reconstructing matrix from SVD")
A_norm_rank_k = U @ np.diag(s) @ Vh

quantile_prob = 0.001
print(f"Find the {quantile_prob:.5f} quantile of each gene")

A_norm_rank_k_mins = np.abs(np.quantile(A_norm_rank_k, quantile_prob, axis=0))
print("Thresholding by the most negative value of each gene")

A_norm_rank_k[A_norm_rank_k <= A_norm_rank_k_mins] = 0
A_norm_rank_k_cor = A_norm_rank_k

print("Calculating std and mean")
sigma1 = np.std(A_norm_rank_k_cor, axis=1, ddof=1, where=(A_norm_rank_k_cor != 1))
sigma2 = np.std(data, axis=1, ddof=1, where=(data != 0))
mu1 = np.sum(A_norm_rank_k_cor, axis=1) / np.count_nonzero(A_norm_rank_k_cor, axis=1)
mu2 = np.sum(data, axis=1) / np.count_nonzero(data, axis=1)

# determine columns to scale (avoid divide by zero)
toscale = ~np.isnan(sigma1) & ~np.isnan(sigma2) & ~((sigma1 == 0) & (sigma2 == 0)) & ~(sigma1 == 0)
num_no_scale = np.sum(~toscale)

print(f"Scaling all except for {num_no_scale} columns")

sigma12 = sigma2 / sigma1

toadd = -1 * mu1 * sigma12 + mu2

A_norm_rank_k_temp = A_norm_rank_k_cor[toscale, :]
A_norm_rank_k_temp *= sigma12[toscale][:, None]
A_norm_rank_k_temp += toadd[toscale][:, None]

A_norm_rank_k_cor_sc = A_norm_rank_k_cor.copy()
A_norm_rank_k_cor_sc[toscale, :] = A_norm_rank_k_temp
A_norm_rank_k_cor_sc[A_norm_rank_k_cor == 0] = 0

lt0 = A_norm_rank_k_cor_sc < 0
A_norm_rank_k_cor_sc[lt0] = 0

per_neg = np.sum(lt0) / data.size
print(f"{per_neg:%} of the values became negative in the scaling process and were set to zero")

# recover original values erroneously set to 0
positions = (data > 0) & (A_norm_rank_k_cor_sc == 0)
A_norm_rank_k_cor_sc[positions] = data[positions]

original_nz = np.count_nonzero(data) / data.size
completed_nz = np.count_nonzero(A_norm_rank_k_cor_sc) / data.size

print(f"The matrix went from {original_nz:%} nonzero to {completed_nz:%} nonzero")

# R anndata cannot read sparse matrix
ann_obj = anndata.AnnData(X = A_norm_rank_k_cor_sc.T, var=pd.DataFrame(index=genes))
ann_obj.write_h5ad(filename + "_imputed.h5ad")

# sparse version with loom
#ann_obj = anndata.AnnData(X = scipy.sparse.csr_matrix(A_norm_rank_k_cor_sc.T), var=pd.DataFrame(index=genes))
#ann_obj.write_loom(filename + "_imputed.loom")
