import numpy as np

# compute inverse of a matrix in mod arithmetic
# matrix (np.ndarray)       matrix to find inverse of
# mod (int)                 modulus (default 26)
def modinv(matrix, mod=26):
    det = np.linalg.det(matrix)
    inv_det = pow(int(round(det)%mod), -1, mod) # find modular inverse of det
    # use numpy inverse but undo the nonmodular division by det step and mult by mod inv det instead
    inv_matrix = np.linalg.inv(matrix) * det * inv_det
    inv_matrix %= mod
    inv_matrix = np.round_(inv_matrix)
    return inv_matrix
