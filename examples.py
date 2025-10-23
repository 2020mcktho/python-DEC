from fibre_solution import FibreSolution
import numpy as np
from scipy.sparse import diags, identity
from scipy.special import jn_zeros, jv
import matplotlib.pyplot as plt

from analytical_solutions import solve_step_index, analytical_Laplace_square_plot, plot_bessel_functions, full_Bessel_solution
from my_generation_pydec import create_uniform_circular_fibre_mesh_delaunay, create_adaptive_circular_fibre_mesh_delaunay, simplicial_grid_2d, create_square_mesh_delaunay

# Build the generalized eigenvalue problem
# We solve: A e = λ B e
# Note: the @ symbol does matrix multiplication


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
    use_core_radius = 0.4
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
    search_val = 1.
    eigenvalues, eigenvectors = fs.solve(A, B, mode_number=eigenval_num, search_near=search_val)

    n_eff = np.sqrt(eigenvalues / (k0 ** 2)) / scale_factor
    print(eigenvalues, n_eff)

    fs.plot_n_shaded()
    for m, eig in enumerate(eigenvalues):
        if (fs.cladding_n < n_eff[m] < fs.core_n or n_eff[m].real / n_eff[m].imag > 100.) or True:
            fs.plot(m, ("shaded", "contours", "mesh"), simplex_type=1)
            # fs.plot_data_on_edges(mode=m)


def ScalarLaplacianDirichletRods():
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


def solid_core_setup(lambda0, core_diam, scale_factor, ref_ind):
    # create an air rod in the centre, such that the width of the silica ring is equal to the wavelength
    # rods = [(np.array((0.5, 0.5)), ((core_diam/2) - lambda0) * scale_factor, 1.)]
    rods = []

    core_ind = ref_ind
    clad_ind = 1.

    mesh_size = round(lambda0 / 8 * scale_factor, 3)
    mesh_size = 0.01
    print(f"Solid core: mesh size = {mesh_size}, index: {ref_ind}")

    k0 = 2 * np.pi / lambda0
    # eigval_guess = 1. + 1.j
    eigval_guess = (1.35 * k0 / scale_factor) ** 2

    return core_ind, clad_ind, mesh_size, rods, eigval_guess


def hollow_core_setup(lambda0, core_diam, scale_factor, ref_ind):
    # create an air rod in the centre, such that the width of the silica ring is equal to the wavelength
    # rods = [(np.array((0.5, 0.5)), ((core_diam/2) - lambda0) * scale_factor, 1.)]
    rods = [(np.array((0.5, 0.5)), (core_diam / 2 + lambda0 / 2) * scale_factor, ref_ind),
            (np.array((0.5, 0.5)), (core_diam / 2) * scale_factor, 1.)]
    # rods = []

    core_ind = 1.
    clad_ind = 1.

    mesh_size = round(lambda0 / 8 * scale_factor, 3)
    mesh_size = 0.02
    print(f"Hollow core: mesh size = {mesh_size}")

    k0 = 2 * np.pi / lambda0
    eigval_guess = (1. * k0 / scale_factor) ** 2 + .1j

    return core_ind, clad_ind, mesh_size, rods, eigval_guess


def ScalarLaplacianWavePMLBeta():
    # Solving the Laplacian, with a 0-form field defined on the vertices
    # Using Dirichlet boundary conditions
    # Eigenvalues can be interpreted as Beta values for propagation constant

    # Note: a Dirichlet boundary implies a perfect electrical conductor, so will force an artificial node at the boundary

    lambda0 = .5e-6
    k0 = 2*np.pi/lambda0

    core_diam = 1.5e-6
    use_core_radius = 0.2
    scale_factor = (use_core_radius * 2) / core_diam  # actual dims x scale factor = sim dims

    k0_scaled = k0 / scale_factor

    # calculate epsilon value (assuming Silica)
    ref_ind = np.sqrt(calc_silica_epsilon_rel(lambda0))

    # core_ind, clad_ind, mesh_size, rods, eigval_guess = hollow_core_setup(lambda0, core_diam, scale_factor, ref_ind)
    core_ind, clad_ind, mesh_size, rods, eigval_guess = solid_core_setup(lambda0, core_diam, scale_factor, ref_ind)

    mesh_size, max_mesh_size, mesh_change_dist = .01, .05, .1
    buffer_size, max_imag_index = .1, 1.e1

    simplex_type = 0  # scalar field defined on the vertices
    fs = FibreSolution(mesh_size=mesh_size, rods=rods, core_radius=use_core_radius, core_n=core_ind, cladding_n=clad_ind, buffer_size=buffer_size, max_imaginary_index=max_imag_index, mesh_generator=create_adaptive_circular_fibre_mesh_delaunay, generator_args=(max_mesh_size, mesh_change_dist))
    fs.setup(epsilon_sc_index=simplex_type, mu_sc_index=simplex_type+1, merge_type="max", use_pml=True, tolerance=mesh_size/2)

    A1 = fs.K[0].d.T @ fs.Hodges[1][0] @ fs.K[0].d  # d0.T @ mu * Hodge1 @ d0
    A2 = k0_scaled ** 2 * fs.Hodges[0][0]  # k0**2 * epsilon * Hodge0
    A = A1 + A2
    B = fs.Hodges[0][2]  # Hodge0, without core epsilon contributions (but including PML still)

    mode_num = 4
    eigenvalues, eigenvectors = fs.solve(A, B, mode_number=mode_num, search_near=eigval_guess)

    # eigenvalues are beta squared values
    beta_sim = np.sqrt(eigenvalues)
    beta = beta_sim * scale_factor
    n_eff = beta / k0
    print(beta, n_eff)  # gives omega (or beta) values

    fs.plot_n_shaded(show_mesh=True)
    # for m in range(6):
    #     fs.plot_data_on_vertices_shaded(mode=m)

    real_imag_ratio_requirement = 1e3

    use_abs_field = False

    used_modes = []
    for m, eig in enumerate(eigenvalues):
        # check that the effective refractive index is between the core and cladding indices
        if fs.cladding_n < n_eff[m] < fs.core_n or True:
            # check that the real part of the eigenvalue is much greater than the imaginary part
            if abs(eig.real / eig.imag) > real_imag_ratio_requirement:
                used_modes.append(m)
                fs.plot(m, ("mesh", "shaded"), simplex_type=simplex_type, absolute_field=use_abs_field)

    # plot the mode cross-sections on one graph
    fs.plot_radial_cross_sections(modes=used_modes, simplex_type=simplex_type, absolute_field=use_abs_field)


    # compare to the analytical solution
    solve_step_index(lambda0, core_diam, core_ind, clad_ind)


def ScalarLaplacianPMLSolid():
    # Solving the Laplacian, with a 0-form field defined on the vertices
    # Using a perfectly matched layer with increasing loss coefficient

    # Solve eigenproblem in img, with eigenvalues=k0^2

    # core is typically 1.25 microns in diameter
    # For a geometry scaled by L: eigenvalues are scaled by 1/L^2 due to the del squared operator
    core_diam = 1.25e-6
    use_core_radius = 0.3
    scale_factor = (use_core_radius * 2) / core_diam

    lambda0 = 1.55e-6
    k0 = 2*np.pi/lambda0

    # solid core setup
    fs = FibreSolution(mesh_size=0.02, core_radius=use_core_radius, core_n=3.5, cladding_n=1, max_imaginary_index=1.0)

    fs.setup(epsilon_sc_index=0, mu_sc_index=1, merge_type="max", use_pml=True)

    A = fs.Hodges[0][1] @ fs.K[0].d.T @ fs.Hodges[1][0] @ fs.K[0].d  # Hodge0_inv @ d0.T @ Hodge1 @ d0
    B = identity(fs.K[0].num_simplices, format="csr", dtype=complex)  # identity matrix, with same dimensions as A
    # B = None

    eigenval_num = 5

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


def find_nearest_index(array: np.ndarray, value: complex | float):
    idx = (np.abs(array - value)).argmin()
    return idx


def BesselCircularDrum(mode_num: int = 6, mesh_size: float = .01, max_mesh_size: float = .05, mesh_change_dist: float = .5, plot_mesh: bool = False, plot_modes: bool = False, plot_difference: bool = False, line_sample: bool = False, plot_rms_difference: bool = False):
    # equation: Laplacian * u = lambda * u
    # u = eigenvectors
    # lambda = eigenvalues

    # Laplacian = Hodge0_inv * d0_inv * Hodge1 * d0

    use_core_radius = 0.4
    eigval_guess = 1.+1.j
    core_ind = 1.
    clad_ind = core_ind - 1e-10

    buffer_size, max_imag_index = .15, 1.e2

    fs = FibreSolution(mesh_size=mesh_size, core_radius=use_core_radius, core_n=core_ind, cladding_n=clad_ind, buffer_size=buffer_size, max_imaginary_index=max_imag_index, mesh_generator=create_adaptive_circular_fibre_mesh_delaunay, generator_args=(max_mesh_size, mesh_change_dist))
    fs.setup(epsilon_sc_index=0, mu_sc_index=1, merge_type="max", use_pml=False, tolerance=mesh_size / 2)

    A = fs.Hodges[0][1] @ fs.K[0].d.T @ fs.Hodges[1][0] @ fs.K[0].d  # Hodge0_inv * d0.T @ mu * Hodge1 @ d0
    B = identity(fs.K[0].num_simplices, format="csr")  # identity matrix, with same dimensions as A

    print(fs.K[0].num_simplices)

    eigenvalues, eigenvectors = fs.solve_with_dirichlet_boundary(A, B, mode_number=mode_num, boundary_type=0, search_near=eigval_guess)

    # calculate the Bessel function zeros from the eigenvalues
    # Eigenvalues should be given by lambda=j_n,m / R
    #   lambda=eigenvalue
    #   j_n,m = jth zero of Bessel function J_n
    #   R = radius at which Dirichlet is implemented (core radius)
    bessel_zeros = np.sqrt(eigenvalues) * use_core_radius

    if plot_mesh:
        fs.plot_n_shaded(show_mesh=True)

    # use the ratio of real to imaginary index to filter the cladding modes (with high imaginary component of the index)
    real_imag_ratio_requirement = 1e5

    use_modes = []
    for m, eig in enumerate(eigenvalues):
        # check that the real part of the eigenvalue is much greater than the imaginary part
        if abs(eig.real / eig.imag) > real_imag_ratio_requirement:
            if plot_modes:
                fs.plot(m, ("mesh", "shaded",), simplex_type=0, absolute_field=True, real_field=True)
            use_modes.append(m)

    # Actual Bessel functions
    actual_zeros = []
    # create lists of the first M zeros of the first N Bessel functions
    N, M = 4, 4  # N = number of Bessel function, M = number of zeros for each
    # Higher M considers higher order azimuthal circular modes
    for n in range(N):  # loop through the Bessel function orders
        for m, val in enumerate(jn_zeros(n, M + 1)):  # loop through the zeros of this Bessel function
            actual_zeros.append((val, n, m))

    actual_zeros.sort(key=lambda x: x[0])
    actual_zeros = np.array(actual_zeros)
    zeros_transposed = actual_zeros.T

    # only scale the fields which have a maximum value of at least 1/100 of the highest value
    normalise_percentage = 1.e-1
    max_eigvec_mode = np.max(fs.eigvecs.ravel())
    scale_tolerance = max_eigvec_mode * normalise_percentage  # for modes with maximum value below this, it will not be normalised since it is approximately zero

    # calculate the values by which each function should be scaled
    corresponding_analytic_modes = []
    scale_values = []
    for mode in use_modes:
        # find the closest analytical zero
        mode_zero = bessel_zeros[mode]
        ind = find_nearest_index(zeros_transposed[0], mode_zero)
        zero_val, n, m = actual_zeros[ind]

        # find the vertex with the greatest mode (absolute) value
        mode_data = fs.eigvecs[:, mode]
        vertex_ind_max = np.argmax(np.abs(mode_data))
        vertex_max = fs.K.vertices[vertex_ind_max]
        # convert to r theta form
        r_max = np.sqrt(np.sum((vertex_max-.5)**2))
        theta_max = np.arctan2((vertex_max[1]-.5), (vertex_max[0]-.5))

        # analytical value
        analytical_val_cosine = full_Bessel_solution(r_max, theta_max, n, m, use_core_radius, normalise=True)
        analytical_val_sine = full_Bessel_solution(r_max, theta_max, n, m, use_core_radius, True, normalise=True)
        symmetry_options = [np.abs(analytical_val_cosine), np.abs(analytical_val_sine)]
        symmetry = np.argmax(symmetry_options)  # use the highest of the sin and cosine options
        # set the scale factor for the DEC mode to be the ratio of the analytical value to the DEC value
        scale_values.append(symmetry_options[symmetry] / np.abs(mode_data[vertex_ind_max]))

        # add this mode to the list of which analytic mode this one corresponds to
        corresponding_analytic_modes.append((ind, symmetry))

    corresponding_analytic_modes = np.array(corresponding_analytic_modes)

    analytic_bessel_zeros = actual_zeros[corresponding_analytic_modes.T[0]].T[0]
    print("calculated Bessel zeros:", bessel_zeros[use_modes], "\nanalytical Bessel zeros:", analytic_bessel_zeros)

    full_analytical_zero_data = np.array([[*actual_zeros[ind], symm] for ind, symm in corresponding_analytic_modes])

    # plot the mesh solutions analytically
    centres = fs.barycenter(0)
    for ind, (mode_ind, symmetry) in enumerate(corresponding_analytic_modes):
        if plot_modes:
            mode_val, n, m = actual_zeros[mode_ind]
            field = np.array([full_Bessel_solution(np.sqrt((x-.5)**2 + (y-.5)**2), np.arctan2((y-.5), (x-.5)), n, m, use_core_radius, normalise=True, sine=symmetry) for (x, y) in centres])
            field = np.abs(field) / np.max(np.abs(field))
            plt.tripcolor(*centres.T, field, shading='gouraud', cmap="cividis")
            plt.title(f"Mode {use_modes[ind]} ({mode_val}) Analytical Profile")
            plt.colorbar()
            plt.show()

    # compare the analytical and DEC eigenvalues
    zeros_percent_diff = np.abs(np.divide((bessel_zeros[use_modes] - analytic_bessel_zeros), analytic_bessel_zeros))
    print("eigenvalue % difference:", zeros_percent_diff)

    # plot the radial cross-section of the modes, returning the y-values of the lines
    if line_sample:
        field_line_values = fs.plot_radial_cross_sections(use_modes, simplex_type=0, scale_tolerance=scale_tolerance, positive_start=True, min_radius=0., max_radius=use_core_radius, absolute_field=True, scale_values=scale_values)
        bessel_field_line_values = plot_bessel_functions(full_analytical_zero_data, radius=use_core_radius, scale_to_1=False, normalise=True, absolute_field=True)

        # perform a mean square fit to find the difference between the scaled line plots
        rt_mean_squares = []
        for mode, (field_line, bessel_field_line) in enumerate(zip(field_line_values, bessel_field_line_values)):
            rt_mean_square_diff = np.sqrt(np.mean((field_line - bessel_field_line)**2))
            rt_mean_squares.append(rt_mean_square_diff)
        plt.plot(range(len(field_line_values)), rt_mean_squares)
        plt.xlabel("mode")
        plt.ylabel("root mean square difference")
        plt.title("rms difference from line sampling")
        plt.show()

    # integrate the square of both the DEC result and the corresponding analytic mode over the full surface to compare
    # if mode_integrals is approximately 1.+0.j then the eigenvectors are properly normalised
    mode_integrals = [np.conj(fs.eigvecs[:, mode]) @ B @ fs.eigvecs[:, mode] for mode in use_modes]

    # The Bessel functions can be normalised to integrate to 1 by dividing them by the square root of these values
    # analytical_integrals = [np.pi * use_core_radius ** 2 * (jv(n+1, zero_val)) ** 2 for zero_val, n, m in actual_zeros]
    # print("mode integrals:", mode_integrals, "\nanalytic integrals:", analytical_integrals)

    # plot the difference between the analytical solution and the DEC solution for each mode
    # also calculate the root-mean-square difference value for each mode plot
    rms_values = []
    for ind, (mode_ind, symmetry) in enumerate(corresponding_analytic_modes):
        mode_val, n, m = actual_zeros[mode_ind]
        field = np.array([full_Bessel_solution(np.sqrt((x - .5) ** 2 + (y - .5) ** 2), np.arctan2((y - .5), (x - .5)),
                                               n, m, use_core_radius, normalise=True, sine=symmetry) for (x, y) in
                          centres])

        # scale both the analytical and DEC fields
        scale_val = scale_values[ind]
        field_diff = np.abs(field) - (np.abs(fs.eigvecs[:, ind]) * scale_val)
        # weight this by the dual volumes, using the dual_volume attribute of the simplex
        # divide these values by the area of the drum to fix the scaling a bit
        weighted_field_diff = np.multiply(field_diff, fs.K[0].dual_volume) / (np.pi * use_core_radius ** 2)

        if plot_difference:
            plt.tripcolor(*centres.T, weighted_field_diff, shading='gouraud', cmap="cividis")
            plt.title(f"Mode {use_modes[ind]} ({mode_val}) Difference Profile")
            plt.colorbar()
            plt.show()

        # calculate the root-mean-square of the field differences
        rms_val = np.sqrt(np.mean(weighted_field_diff ** 2))
        rms_values.append(rms_val)

    # plot the rms difference values
    print("RMS eigenvector difference:", rms_values)
    if plot_rms_difference:
        plt.plot(range(len(corresponding_analytic_modes)), rms_values)
        plt.xlabel("mode")
        plt.ylabel("root mean square difference")
        plt.show()

    # return the percentage difference in the bessel zeros and return the rms difference values
    return use_modes, zeros_percent_diff, rms_values


def BesselMeshRun():
    mesh_sizes = [.005, .01, .02, .05, .1, .2]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1 will be a plot of the eigenvalue differences
    # ax2 will be a plot of the rms eigenvector differences

    for ind, mesh_size in enumerate(mesh_sizes[:-1]):
        # for ind2, max_mesh_size in enumerate(mesh_sizes[ind+1:]):
        max_mesh_size = mesh_sizes[ind + 1]
        # mesh_change_dist = .05
        mesh_change_dist = (mesh_size + max_mesh_size) / 2

        label = f"min: {mesh_size}, max: {max_mesh_size}, change: {mesh_change_dist}"

        print("simulating circular drum:\n", label)
        modes, zeros_percent_diff, rms_values = BesselCircularDrum(mesh_size=mesh_size, max_mesh_size=max_mesh_size, mesh_change_dist=mesh_change_dist)

        ax1.plot(modes, zeros_percent_diff, label=label)
        ax2.plot(modes, rms_values, label=label)

    ax1.set_xlabel("mode number")
    ax1.set_ylabel("percentage mode value difference")
    ax2.set_xlabel("mode number")
    ax2.set_ylabel("RMS eigenvector difference")

    plt.legend()

    plt.show()


def Laplace_square():
    fs = FibreSolution(mesh_size=0.02, core_radius=0.25, core_n=1., mesh_generator=simplicial_grid_2d)
    fs.setup(epsilon_sc_index=0)

    # Scalar Laplacian, matrices have dimensions of vertex number
    A = fs.Hodges[0][1] @ fs.K[0].d.T @ fs.Hodges[1][0] @ fs.K[0].d  # Hodge0_inv @ d0.T @ Hodge1 @ d0
    B = identity(fs.K[0].num_simplices, format="csr")  # identity matrix, with same dimensions as A

    eigenvalues, eigenvectors = fs.solve_with_dirichlet_boundary(A, B, 0, mode_number=20)
    print("simulated eigenvalues:", eigenvalues)  # gives omega (or beta) values

    fs.plot_n_shaded()
    for mode in [0, 1, 2, 3]:
        fs.plot(mode, ("mesh", "shaded",), simplex_type=0)

    modes = [0, 1, 2, 3]
    fs.plot_radial_cross_sections(modes, show_core_boundary=False, absolute_field=True)

    # analytical solution
    analytical_Laplace_square_plot(modes, absolute_field=True)
