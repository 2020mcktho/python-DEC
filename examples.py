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

    core_diam = 1.25e-6
    use_core_radius = 0.25
    scale_factor = (use_core_radius * 2) / core_diam
    scale_factor = 1.

    lambda0 = 1.55e-6
    k0 = 2 * np.pi / lambda0

    fs = FibreSolution(mesh_size=0.05, core_radius=use_core_radius, core_n=3.5, cladding_n=1.0, buffer_size=0.5, max_imaginary_index=0.1)
    fs.setup(epsilon_sc_index=1, merge_type="average", use_pml=True)

    # Maxwell equation for E field  (combining grad(E) and grad(H))
    A = fs.K[1].d.T @ fs.Hodges[2][0] @ fs.K[1].d  # d1.T @ Hodge2_inv @ d1
    B = fs.Hodges[1][0]  # Hodge1

    eigenval_num = 6
    # want n_eff around 3 ish, so eigenvalues around (3 * scale_factor) ** 2
    search_val = (3 * scale_factor * k0) ** 2
    search_val = 1.
    eigenvalues, eigenvectors = fs.solve(A, B, mode_number=eigenval_num, search_near=search_val)

    n_eff = np.sqrt(eigenvalues / (k0 ** 2)) / scale_factor
    print(eigenvalues, n_eff)

    fs.plot_n_shaded()
    for m, eig in enumerate(eigenvalues):
        if fs.cladding_n < n_eff[m] < fs.core_n or n_eff[m].real / n_eff[m].imag > 100.:
            fs.plot_data_on_edges(mode=m)


def ScalarLaplacianDirichlet():
    # Solving the Laplacian, with a 0-form field defined on the vertices
    # Using Dirichlet boundary conditions

    lambda0 = 1.55e-6
    k0 = 2*np.pi/lambda0

    fs = FibreSolution(mesh_size=0.025, core_radius=0.45, core_n=1.45, cladding_n=1.44)
    fs.setup(epsilon_sc_index=1, merge_type="max", use_pml=True)

    A = fs.Hodges[0][1] @ fs.K[0].d.T @ fs.Hodges[1][0] @ fs.K[0].d  # Hodge0_inv @ d0.T @ Hodge1 @ d0
    B = np.eye(fs.K[0].num_simplices)  # identity matrix, with same dimensions as A

    eigenvalues, eigenvectors = fs.solve_with_dirichlet_core(A, B, 0, mode_number=10)
    n_eff = np.sqrt(k0 ** 2 / eigenvalues)
    print(eigenvalues, n_eff)  # gives omega (or beta) values

    fs.plot_n_shaded()
    # for m in range(6):
    #     fs.plot_data_on_vertices_shaded(mode=m)

    for m, eig in enumerate(eigenvalues):
        if fs.cladding_n < n_eff[m] < fs.core_n or True:
            fs.plot(m, ("vertices_shaded", "contours", "mesh"))

def ScalarLaplacianPML():
    # Solving the Laplacian, with a 0-form field defined on the vertices
    # Using a perfectly matched layer with increasing loss coefficient

    # Solve eigenproblem in img, with eigenvalues=k0^2

    # core is typically 1.25 microns in diameter
    # For a geometry scaled by L: eigenvalues are scaled by 1/L^2 due to the del squared operator
    core_diam = 1.25e-6
    use_core_radius = 0.25
    scale_factor = (use_core_radius * 2) / core_diam

    lambda0 = 1.55e-6
    k0 = 2*np.pi/lambda0

    fs = FibreSolution(mesh_size=0.05, core_radius=use_core_radius, core_n=3.5, cladding_n=1.0, max_imaginary_index=1.0)
    # Note: n=20 seems to be almost as good as n=80 for this fibre design

    fs.setup(epsilon_sc_index=1, merge_type="max", use_pml=True)

    A = fs.Hodges[0][1] @ fs.K[0].d.T @ fs.Hodges[1][0] @ fs.K[0].d  # Hodge0_inv @ d0.T @ Hodge1 @ d0
    B = np.eye(fs.K[0].num_simplices)  # identity matrix, with same dimensions as A

    eigenval_num = 7

    eigenvalues, eigenvectors = fs.solve(A, B, mode_number=eigenval_num, search_near=1.)
    eigenvalues *= (scale_factor ** 2)
    # scale the eigenvalues
    # the scaled eigenvalues represent the wavenumber squared (lambda = k0^2)
    # eigenvalues *= 1 / scale_factor ** 2
    # k0_vals = np.sqrt(eigenvalues) / scale_factor
    n_eff = np.sqrt(eigenvalues) / k0
    print(eigenvalues, n_eff)

    fs.plot_n_shaded()
    for m, eig in enumerate(eigenvalues):
        if fs.cladding_n < n_eff[m] < fs.core_n or True:
            fs.plot(m, ("vertices_shaded", "contours", "mesh"))
            # fs.plot_data_on_vertices_shaded(mode=m)
            # fs.plot_contours(mode=m)

def InhomogeneousWaveEqn():
    # RHS:
    # - d0 @ ( Hodge1_inv @ (  ) )
    # - d0 @ Hodge1_inv
    return

def Laplace_square():
    fs = FibreSolution(mesh_size=0.05, core_radius=0.25)
    fs.setup(epsilon_sc_index=0)

    # Scalar Laplacian, matrices have dimensions of vertex number
    A = fs.Hodges[0][1] @ fs.K[0].d.T @ fs.Hodges[1][0] @ fs.K[0].d  # Hodge0_inv @ d0.T @ Hodge1 @ d0
    B = np.eye(fs.K[0].num_simplices)  # identity matrix, with same dimensions as A

    eigenvalues, eigenvectors = fs.solve_with_dirichlet(A, B, 0, mode_number=6)
    print(eigenvalues)  # gives omega (or beta) values

    fs.plot_n()
    for mode in range(1):
        fs.plot_data_on_vertices(mode=mode)
