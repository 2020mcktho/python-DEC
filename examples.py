from fibre_solution import FibreSolution
import numpy as np
from scipy.sparse import diags, identity

# Build the generalized eigenvalue problem
# We solve: A e = λ B e
# Note: the @ symbol does matrix multiplication

epsilon_0 = 8.8541878188e-12  # Fm^-1


def calc_silica_epsilon_rel(lambda0: float):
    # use the Sellmeier equation to calculate epsilon from known coefficients for Silica
    # coefficients from Malitson, 1965

    # lambda0 is given in metres, then converted to micrometres
    lambda0_um = lambda0 / 1e-6

    B1, C1 = 0.6961663, 0.0684043 ** 2
    B2, C2 = 0.4079426, 0.1162414 ** 2
    B3, C3 = 0.8974794, 9.896161 ** 2

    lam2 = lambda0_um ** 2  # lambda squared

    n_squared = (1
          + (B1 * lam2) / (lam2 - C1)
          + (B2 * lam2) / (lam2 - C2)
          + (B3 * lam2) / (lam2 - C3))

    print(n_squared, lam2)

    return n_squared

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

    fs = FibreSolution(mesh_size=0.05, core_radius=use_core_radius, core_n=3.5, cladding_n=1.0, buffer_size=0.1, max_imaginary_index=0.1)
    fs.setup(epsilon_sc_index=1, merge_type="max", use_pml=True)

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
        if (fs.cladding_n < n_eff[m] < fs.core_n or n_eff[m].real / n_eff[m].imag > 100.) or True:
            fs.plot(m, ("shaded", "contours", "mesh"), simplex_type=1)
            # fs.plot_data_on_edges(mode=m)


def ScalarLaplacianDirichlet():
    # Solving the Laplacian, with a 0-form field defined on the vertices
    # Using Dirichlet boundary conditions

    # Note: a Dirichlet boundary implies a perfect electrical conductor, so will force an artificial node at the boundary

    lambda0 = 1.55e-6
    k0 = 2*np.pi/lambda0

    # create surrounding rods
    rod_layers, rods_density, layer_dist = 3, 0.2, 0.2  # rod_density is the distance between each rod, layer_dist is the distance between layers
    rods, rod_rad, rod_dist = [], 0.05, 0.3
    for layer in range(rod_layers):
        total_dist = rod_dist + layer_dist * layer
        # find the number of rods required to maintain the minimum rod density (round up)
        rod_num = int(np.ceil((2 * np.pi * total_dist) / rods_density))
        for i in range(rod_num):
            angle = 2 * np.pi * (i / rod_num)
            x, y = total_dist * np.cos(angle), total_dist * np.sin(angle)
            rods.append((np.array((x + 0.5, y + 0.5)), rod_rad))

    # calculate epsilon value (assuming Silica)
    ref_ind = np.sqrt(calc_silica_epsilon_rel(lambda0))
    print(ref_ind)

    fs = FibreSolution(mesh_size=0.02, core_radius=0.4, rods=rods, core_n=ref_ind, cladding_n=ref_ind, rod_n=1., buffer_size=0.1)
    fs.setup(epsilon_sc_index=1, merge_type="max", use_pml=True)

    A = fs.Hodges[0][1] @ fs.K[0].d.T @ fs.Hodges[1][0] @ fs.K[0].d  # Hodge0_inv @ d0.T @ Hodge1 @ d0
    B = identity(fs.K[0].num_simplices, format="csr")  # identity matrix, with same dimensions as A

    eigenvalues, eigenvectors = fs.solve_with_dirichlet_core(A, B, 0, mode_number=10)
    n_eff = np.sqrt(k0 ** 2 / eigenvalues)
    print(eigenvalues, n_eff)  # gives omega (or beta) values

    fs.plot_n_shaded()
    # for m in range(6):
    #     fs.plot_data_on_vertices_shaded(mode=m)

    real_imag_ratio_requirement = 1e-3

    for m, eig in enumerate(eigenvalues):
        # check that the effective refractive index is between the core and cladding indices
        if fs.cladding_n < n_eff[m] < fs.core_n or True:
            # check that the real part of the eigenvalue is much greater than the imaginary part
            if abs(eig.real / eig.imag) > real_imag_ratio_requirement:
                fs.plot(m, ("shaded",), simplex_type=0)

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
    B = identity(fs.K[0].num_simplices, format="csr")  # identity matrix, with same dimensions as A

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
    B = identity(fs.K[0].num_simplices, format="csr")  # identity matrix, with same dimensions as A

    eigenvalues, eigenvectors = fs.solve_with_dirichlet(A, B, 0, mode_number=6)
    print(eigenvalues)  # gives omega (or beta) values

    fs.plot_n()
    for mode in range(1):
        fs.plot_data_on_vertices(mode=mode)
