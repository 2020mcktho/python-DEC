from fibre_solution import FibreSolution
import numpy as np
from scipy.sparse import diags

# Build the generalized eigenvalue problem
# We solve: A e = λ B e
# Note: the @ symbol does matrix multiplication

def EM_field():
    # Electric field wave solution, in the frequency domain
    # curl(curl(E)) = star[d(star dE)] = star d star d E
    # E is a 1-form
    # therefore:
    # d1 E -> 2-form
    # star2 (2-form) -> dual 0-form
    # d0 (dual 0-form) -> dual 1-form
    # star1.T (dual 1-form) -> 1-form

    fs = FibreSolution(n=20, core_radius=0.0)
    fs.setup(epsilon_sc_index=0)

    # Maxwell equation for E field  (combining grad(E) and grad(H))
    A = fs.K[1].d.T @ fs.Hodges[2][0] @ fs.K[1].d  # d1.T @ Hodge2_inv @ d1
    B = fs.Hodges[1][0]  # Hodge1

    eigenvalues, eigenvectors = fs.solve_with_dirichlet(A, B, 1)
    print(eigenvalues)  # gives omega (or beta) values

    fs.plot_n()
    fs.plot_data_on_edges(mode=0)


def ScalarLaplacianDirichlet():
    # Solving the Laplacian, with a 0-form field defined on the vertices
    # Using Dirichlet boundary conditions

    lambda0 = 1.55e-6
    k0 = 2*np.pi/lambda0

    fs = FibreSolution(n=10, core_radius=0.25, core_n=1.45, cladding_n=1.44)
    fs.setup(epsilon_sc_index=1, merge_type="average")

    A = fs.Hodges[0][1] @ fs.K[0].d.T @ fs.Hodges[1][0] @ fs.K[0].d  # Hodge0_inv @ d0.T @ Hodge1 @ d0
    B = np.eye(fs.K[0].num_simplices)  # identity matrix, with same dimensions as A

    eigenvalues, eigenvectors = fs.solve_with_dirichlet(A, B, 0, mode_number=6)
    n_eff = np.sqrt(k0 ** 2 / eigenvalues)
    print(eigenvalues, n_eff)  # gives omega (or beta) values

    fs.plot_n_shaded()
    for m in range(6):
        fs.plot_data_on_vertices_shaded(mode=m)

def ScalarLaplacianPML():
    # Solving the Laplacian, with a 0-form field defined on the vertices
    # Using a perfectly matched layer with increasing loss coefficient
    lambda0 = 1.55e-6
    k0 = 2*np.pi/lambda0

    fs = FibreSolution(n=20, core_radius=0.25, core_n=1.45, cladding_n=1.44)
    fs.setup(epsilon_sc_index=1, merge_type="max", use_pml=True)

    A = fs.Hodges[0][1] @ fs.K[0].d.T @ fs.Hodges[1][0] @ fs.K[0].d  # Hodge0_inv @ d0.T @ Hodge1 @ d0
    B = np.eye(fs.K[0].num_simplices)  # identity matrix, with same dimensions as A

    eigenvalues, eigenvectors = fs.solve(A, B, mode_number=6)
    n_eff = np.sqrt(k0 ** 2 / eigenvalues)
    print(eigenvalues, n_eff)  # gives omega (or beta) values

    fs.plot_n_shaded()
    for m in range(6):
        fs.plot_data_on_vertices_shaded(mode=m)

def Laplace_square():
    fs = FibreSolution(n=20, core_radius=0.0)
    fs.setup(epsilon_sc_index=0)

    # Scalar Laplacian, matrices have dimensions of vertex number
    A = fs.Hodges[0][1] @ fs.K[0].d.T @ fs.Hodges[1][0] @ fs.K[0].d  # Hodge0_inv @ d0.T @ Hodge1 @ d0
    B = np.eye(fs.K[0].num_simplices)  # identity matrix, with same dimensions as A

    eigenvalues, eigenvectors = fs.solve_with_dirichlet(A, B, 0)
    print(eigenvalues)  # gives omega (or beta) values

    fs.plot_n()
    fs.plot_data_on_vertices(mode=0)
