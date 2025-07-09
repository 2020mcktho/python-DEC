from fibre_solution import FibreSolution
import numpy as np

# Build the generalized eigenvalue problem
# We solve: A e = λ B e
# Note: the @ symbol does matrix multiplication

# Electric field wave solution
# curl(curl(E)) = star[d(star dE)] = star d star d E
# E is a 1-form
# therefore:
# d1 E -> 2-form
# star2 (2-form) -> dual 0-form
# d0 (dual 0-form) -> dual 1-form
# star1.T (dual 1-form) -> 1-form

fs = FibreSolution(core_radius=0.2)
fs.setup(epsilon_sc_index=1)

# A = fs.K[1].d.T @ Hodges[2][0] @ fs.K[1].d  # d1.T @ Hodge2_inv @ d1
# B = fs.Hodges[1][0]  # Hodge1

# Scalar Laplacian, matrices have dimensions of vertex number
A = fs.Hodges[0][1] @ fs.K[0].d.T @ fs.Hodges[1][0] @ fs.K[0].d  # Hodge0_inv @ d0.T @ Hodge1 @ d0
B = np.eye(fs.K[0].num_simplices)  # identity matrix, with same dimensions as A
print(fs.Hodges[1][0])

eigenvalues, eigenvectors = fs.solve_with_dirichlet(A, B, 0)
print(eigenvalues)  # gives omega (or beta) values

fs.plot_epsilon()
# plot_data_on_edges(mode=0)
fs.plot_data_on_vertices(mode=1)