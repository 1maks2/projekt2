import numpy as np
import scipy as sp
from scipy.sparse.linalg import gmres

#SINGULARNE
A = np.array([[2, 8, 8, 9], [1, 6, 2, 0], [4, 0, 7, 2], [9, 9, 9, 4]])
singular_values = np.linalg.svd(A, compute_uv=False)
min_singular_value = min(singular_values)
print(f"Najmniejsza wartość singularna: {min_singular_value:.4f}")

#GMRES
A = np.array([[6, 6, 2, 8], [7, 9, 3, 2], [2, 9, 1, 9], [1, 9, 9, 1]])
b = np.array([1, 5, 6, 2])

tol = 10^(-10)
max_iter = 10

solution, exit_code = gmres(A, b, rtol=tol, restart=max_iter)

residuum_norm = np.linalg.norm(A @ solution - b)

print("Rozwiązanie:", solution)
print("Norma residuum:", residuum_norm)

#METODA QR
A = np.array([[6, 6, 2, 8], [7, 9, 3, 2], [2, 9, 1, 9], [1, 9, 9, 1]])
b = np.array([1, 5, 6, 2])
Q, R = np.linalg.qr(A)
b_q = Q.T @ b
solution = sp.linalg.solve_triangular(R, b_q)
print("Rozwiązanie:", solution)
