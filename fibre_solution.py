import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse import diags
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.tri as tri
from pydec import simplicial_complex

# Step 1: Create a simple 2D mesh (square domain)
# In real use, you'd load a more realistic PCF mesh with air holes.
# from pydec.mesh.generation import simplicial_grid_2d
from my_generation_pydec import simplicial_grid_2d, create_uniform_circular_fibre_mesh_delaunay, create_fibre_mesh_delaunay


epsilon_0 = 8.8541878188e-12  # Fm^-1
mu_0 = 1.25663706127e-6  # NA^-2

class FibreSolution:
    def __init__(self, sc: simplicial_complex = None, mesh_size: float = 0.05, dimension: float = 1., scale_factor: float = 1., core_radius: float = 0.4, rods: tuple[tuple[np.ndarray, float, complex]] = (), core_n: complex = 3.5, cladding_n: complex = 1., buffer_size: float = 0.05, max_imaginary_index: float = .1, colour_map="cividis", mesh_generator=create_uniform_circular_fibre_mesh_delaunay, generator_args: tuple = ()):
        # if no simplicial complex mesh provided, generate a square mesh instead, using n divisions per side
        if sc is None:
            # vertices, triangles = simplicial_grid_2d(mesh_size)
            # vertices, triangles = create_circular_fibre_mesh_delaunay(mesh_size, core_radius)
            vertices, triangles = mesh_generator(mesh_size, core_radius, *generator_args)

            # vertices, triangles = create_fibre_mesh_delaunay(mesh_size, core_radius)
            # vertices, triangles = create_fibre_mesh(mesh_size, core_radius)
            # Create simplicial complex
            self.K = simplicial_complex(vertices, triangles)

        self.mesh_size = mesh_size

        # Example: circle in the center (high-index core)
        self.core_radius = core_radius
        self.rods = rods
        self.core_n = core_n
        self.cladding_n = cladding_n

        self.buffer_size = buffer_size

        self.Hodges = [[self.K[0].star.astype(complex), self.K[0].star_inv.astype(complex), self.K[0].star.astype(complex), self.K[0].star_inv.astype(complex)],
                       [self.K[1].star.astype(complex), self.K[1].star_inv.astype(complex), self.K[1].star.astype(complex), self.K[1].star_inv.astype(complex)],
                       [self.K[2].star.astype(complex), self.K[2].star_inv.astype(complex), self.K[2].star.astype(complex), self.K[2].star_inv.astype(complex)]]
        self.n_vals = np.array([1], dtype=complex)
        self.n_mat = diags(self.n_vals)
        self.n_mat_inv = diags(self.n_vals)
        self.eigvals, self.eigvecs = np.array([]), np.array([])
        self.use_pml = False
        self.max_imaginary_index = max_imaginary_index
        self.setup()

        self.cmap = colour_map

    def barycenter(self, sc_index: int = 1):
        return np.average(self.K.vertices[self.K[sc_index].simplices], axis=1)

    def get_ext_and_int_edge_ind(self):
        boundary_edges = self.K.boundary()
        boundary_edges.sort()
        boundary_ind = list(np.sort([self.K[1].simplex_to_index[e]
                                     for e in boundary_edges]))
        internal_edges = set(self.K[1].simplex_to_index.keys()) - set(boundary_edges)
        internal_ind = list(np.sort([self.K[1].simplex_to_index[e]
                                     for e in internal_edges]))
        return boundary_ind, internal_ind

    def get_ext_and_int_vertex_ind(self):
        boundary_edges = self.K.boundary()
        boundary_edges.sort()
        boundary_ind = set()
        for edge in boundary_edges:
            boundary_ind.add(edge[0])
            boundary_ind.add(edge[1])
        internal_ind = set(range(len(self.K.vertices))) - boundary_ind
        return list(boundary_ind), list(internal_ind)

    def get_core_boundary_edge_indices(self):
        edges = self.K[1].simplices

        # find the edges that move from a core vertex to a cladding vertex
        boundary_edges = []
        for edge_ind, e in enumerate(edges):
            v1, v2 = self.K.vertices[e[0]], self.K.vertices[e[1]]

            # loop through the rods and check those boundaries as well
            in1 = np.sqrt((v1[0] - 0.5) ** 2 + (v1[1] - 0.5) ** 2) < self.core_radius
            in2 = np.sqrt((v2[0] - 0.5) ** 2 + (v2[1] - 0.5) ** 2) < self.core_radius
            if in1 != in2:
                boundary_edges.append(edge_ind)

        return boundary_edges

    def get_refractive_index_edge_boundary_indices(self):
        edges = self.K[1].simplices

        # find the edges that move from a core vertex to a cladding vertex
        boundary_edges = []
        for edge_ind, e in enumerate(edges):
            v1, v2 = self.K.vertices[e[0]], self.K.vertices[e[1]]
            # find the refractive index of both end points
            eps1, eps2 = self.n_vals[e[0]], self.n_vals[e[1]]
            # if the real parts of these indices are different, it is a boundary between the sections
            if eps1.real != eps2.real:
                boundary_edges.append(edge_ind)

        return boundary_edges

    def get_edge_vertex_indices(self, edge_indices):
        vert_ind = []
        # loop through the edge indices
        for ind in edge_indices:
            # find the vertices of the edge
            v1, v2 = self.K[1].simplices[ind]
            # add new vertices to the index list
            if v1 not in vert_ind:
                vert_ind.append(v1)
            if v2 not in vert_ind:
                vert_ind.append(v2)
        return vert_ind

    def create_n_geometry(self, tolerance: float | None = None):
        # create a matrix of refractive index values at each vertex
        # these will be averaged to find the values for the edges or faces
        self.n_vals = np.ones(len(self.K.vertices), dtype=complex) * self.cladding_n

        # if the tolerance is not set, use half the step size instead
        if tolerance is None:
            tolerance = self.mesh_size / 2

        barycenter_points = self.barycenter(0)
        barycenter_points = self.K.vertices
        x, y = barycenter_points.T
        r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        self.n_vals[r <= self.core_radius + tolerance] = self.core_n

        for (rx, ry), rod_radius, rod_n in self.rods:
            r2 = np.sqrt((x - rx) ** 2 + (y - ry) ** 2)
            self.n_vals[r2 <= rod_radius + tolerance] = rod_n

    def create_n_matrix(self, sc_index: int = 1, merge_type: str = "average"):
        if merge_type == "average":  # use the average of the surrounding points
            self.n_mat = diags(np.average(self.n_vals[self.K[sc_index].simplices], axis=1))
            self.n_mat_inv = diags(1 / np.average(self.n_vals[self.K[sc_index].simplices], axis=1))
        else:  # use the max value instead
            self.n_mat = diags(np.max(self.n_vals[self.K[sc_index].simplices], axis=1))
            self.n_mat_inv = diags(1 / np.max(self.n_vals[self.K[sc_index].simplices], axis=1))

    def apply_n_matrix(self, sc_index: int = 0, pml_only: bool = False):
        # if this should only affect the pml matrices, modify Hodges [sc_index][2 and 3]
        # if this should only affect the epsilon matrices, modify Hodges[sc_index][0 and 1]
        if pml_only:
            self.Hodges[sc_index][2] = self.K[sc_index].star @ (self.n_mat_inv ** 2)
            self.Hodges[sc_index][3] = self.K[sc_index].star_inv @ (self.n_mat ** 2)
        else:
            self.Hodges[sc_index][0] = self.K[sc_index].star @ (self.n_mat_inv ** 2)
            self.Hodges[sc_index][1] = self.K[sc_index].star_inv @ (self.n_mat ** 2)

    def apply_mu(self, sc_index: int = 1):
        use_mu = 1  # don't use mu_0 here?
        self.Hodges[sc_index][0] = (self.K[sc_index].star * 1 / use_mu).astype(complex)
        self.Hodges[sc_index][1] = (self.K[sc_index].star_inv * use_mu).astype(complex)

    def create_pml_vertex_buffer(self):
        # max_imaginary_index controls the decay speed inside the buffer

        # (.5 - buffer_size) gives the radius at which the buffer starts, and it stretches all the way until the corner
        buffer_start = .5 - self.buffer_size
        buffer_size = (.5 * np.sqrt(2) - buffer_start)

        def calc_buffer_index_percentage(r):
            percentage = (r - buffer_start) / buffer_size
            return percentage ** 2

        # find the points within the buffer zone at the edge (for the perfectly matched layer)
        x, y = self.K.vertices.T
        dist_to_edge = np.min([x, 1. - x, y, 1. - y], axis=0)
        dist_from_centre = np.sqrt((x - .5) ** 2 + (y - .5) ** 2)
        in_buffer = dist_from_centre > (.5 - self.buffer_size)
        # Add the complex loss part to the refractive index inside the Perfectly Matched Layer
        absorption = np.zeros_like(self.n_vals)
        absorption[in_buffer] = 1j * (calc_buffer_index_percentage(dist_from_centre[in_buffer]) * self.max_imaginary_index)

        self.n_vals += absorption  # add the complex part to the vertices inside the buffer

    def setup(self, epsilon_sc_index: int = 0, mu_sc_index: int = 1, merge_type: str = "average", use_pml: bool = False, tolerance: float | None = None):
        # Set up the fibre geometry and different sections of refractive index
        # Also set up the perfectly matched layer, if it is being used

        self.use_pml = use_pml

        self.create_n_geometry(tolerance)
        if self.use_pml:
            self.create_pml_vertex_buffer()
            self.apply_n_matrix(epsilon_sc_index, True)
        self.create_n_matrix(epsilon_sc_index, merge_type)

        self.apply_n_matrix(epsilon_sc_index)
        self.apply_mu(mu_sc_index)

    def solve_with_dirichlet_core(self, A: np.ndarray, B: np.ndarray, boundary_type: int = 1, mode_number: int = -1, eigval_pref: str = "LM", search_near: complex = 1., real_matrix: bool = True):
        # identify the edges going from core to cladding vertices
        indices1 = self.get_core_boundary_edge_indices()
        indices = self.get_refractive_index_edge_boundary_indices()
        print(len(indices1), len(indices))
        """
        for e in edges:
            A[e, :] = 0
            A[e, e] = 1
            B[e] = 0
        """

        # create a mask to filter out the vertices (or edges) with the Dirichlet condition applied
        dimension = self.K[boundary_type].num_simplices  # number of points in the mesh of this type
        mask = np.ones(dimension, dtype=bool)  # 1=no dirichlet, 0=dirichlet

        # if using vertices rather than edges, make a list of all the edge end points
        if boundary_type == 0:
            indices = self.get_edge_vertex_indices(indices)

        # set the mask to be False where the dirichlet boundaries are active
        mask[indices] = False
        free = np.nonzero(mask)[0]  # find the indices of the non-dirichlet values

        # only take the rows and columns of the matrices corresponding to indices without Dirichlet boundaries
        A_reduced = A[free, :][:, free]
        B_reduced = B[free, :][:, free]

        # if the mode number is -1, get all the modes
        if mode_number == -1:
            print(A_reduced.shape)
            mode_number = A_reduced.shape[0] - 2

        # Solve the reduced eigenproblem: A e = λ B e
        # if a perfectly matched layer is being used, this will always require the complex solver
        if real_matrix and not self.use_pml:  # when the matrix is known to be symmetric or Hermitian
            eigvals, eigvecs_reduced = spla.eigsh(A_reduced, k=mode_number, M=B_reduced, sigma=search_near,
                                                  which=eigval_pref)
        else:  # when the matrix may be complex
            eigvals, eigvecs_reduced = spla.eigs(A_reduced.astype(complex), k=mode_number,
                                                 M=B_reduced.astype(complex), sigma=search_near, which=eigval_pref)

        # Sort the eigenpairs (sometimes eigsh returns unordered)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs_reduced = eigvecs_reduced[:, idx]

        # Reconstruct full-length eigenvectors with zeros on boundary
        eigvecs_full = np.zeros((dimension, mode_number), dtype=complex)
        eigvecs_full[free, :] = eigvecs_reduced

        # store the eigenvalues and eigenvectors
        self.eigvals, self.eigvecs = eigvals, eigvecs_full
        return self.eigvals, self.eigvecs

    def solve_with_dirichlet_boundary(self, A: np.ndarray, B: np.ndarray, boundary_type: int = 1, mode_number: int = -1, eigval_pref: str = "LM", search_near: complex = 1., real_matrix: bool = True):
        # Find boundary and internal edges
        if boundary_type == 1:  # edges on the boundary
            boundary_indices, internal_indices = self.get_ext_and_int_edge_ind()
        else:  # vertices on the boundary
            boundary_indices, internal_indices = self.get_ext_and_int_vertex_ind()

        # Restrict A and B to interior edge DoFs
        A_reduced = A[internal_indices, :][:, internal_indices]
        B_reduced = B[internal_indices, :][:, internal_indices]

        # np.ix_() function docs
        # https://numpy.org/doc/stable/reference/generated/numpy.ix_.html
        # A_boundaries = A[]

        # if the mode number is -1, get all the modes
        if mode_number == -1:
            mode_number = A_reduced.shape[0]

        # Solve the reduced eigenproblem: A e = λ B e
        # if a perfectly matched layer is being used, this will always require the complex solver
        if real_matrix and not self.use_pml:  # when the matrix is known to be symmetric or Hermitian
            eigvals, eigvecs_reduced = spla.eigsh(A_reduced, k=mode_number, M=B_reduced, sigma=search_near, which=eigval_pref)
        else:  # when the matrix may be complex
            eigvals, eigvecs_reduced = spla.eigs(A_reduced.astype(complex), k=mode_number, M=B_reduced.astype(complex), sigma=search_near, which=eigval_pref)

        # Sort the eigenpairs (sometimes eigsh returns unordered)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs_reduced = eigvecs_reduced[:, idx]

        # Reconstruct full-length eigenvectors with zeros on boundary
        dimension = self.K[boundary_type].num_simplices  # number of points in the mesh of this type
        eigvecs_full = np.zeros((dimension, mode_number), dtype=complex)
        eigvecs_full[internal_indices, :] = eigvecs_reduced

        self.eigvals, self.eigvecs = eigvals, eigvecs_full
        return self.eigvals, self.eigvecs

    def solve(self, A: np.ndarray, B: np.ndarray | None = None, mode_number: int = -1, eigval_pref: str = "LM", real_matrix: bool = True, search_near: complex = 1.+0.j):
        # if the mode number is -1, get all the modes
        if mode_number == -1:
            mode_number = A.shape[0]

        # Solve the eigenproblem: A e = λ B e
        if real_matrix and not self.use_pml:  # when the matrix is known to be symmetric or Hermitian
            eigvals, eigvecs = spla.eigsh(A, k=mode_number, M=B, sigma=search_near, which=eigval_pref)
        else:  # when the matrix may be complex
            eigvals, eigvecs= spla.eigs(A, k=mode_number, M=B, sigma=search_near, which=eigval_pref)

        # Sort the eigenpairs (sometimes eigsh returns unordered)
        # also, normlise the eigenvectors before storing them
        idx = np.argsort(eigvals)
        self.eigvals = eigvals[idx]
        self.eigvecs = eigvecs[:, idx]

        return self.eigvals, self.eigvecs

    def plot_n_shaded(self, show_mesh: bool = False):
        mode = np.abs(self.n_vals)  # shade using the epsilon values
        plt.tripcolor(*self.K.vertices.T, mode, shading='gouraud', cmap="viridis")

        if show_mesh:
            self.plot_mesh()

        plt.title("Refractive Index Profile")
        plt.colorbar()
        plt.show()

    def plot_n_boundary(self):
        boundary_edges = self.get_refractive_index_edge_boundary_indices()
        boundary_points = self.get_edge_vertex_indices(boundary_edges)


    def plot_n(self):
        # plotting
        fig, ax = plt.subplots()

        vals = np.abs(self.n_vals)
        # fill colour the triangles based on their epsilon value
        min_epsilon = np.min(vals)
        max_epsilon = np.max(vals)
        if min_epsilon == max_epsilon:
            min_epsilon -= 0.01
        cmap = plt.get_cmap("inferno")  # colour map
        norm = mcolors.BoundaryNorm((min_epsilon, max_epsilon), cmap.N, clip=True)  # normalisation
        for ind, k_vertices in enumerate(self.K[2].simplices):
            if len(vals) == len(self.K.vertices):  # epsilon defined on the vertices
                epsilon_val = np.average(vals[k_vertices])
            else:  # epsilon defined on the faces
                epsilon_val = vals[ind]
            col = cmap(norm(epsilon_val))
            trianglex, triangley = self.K.vertices[k_vertices].T
            plt.fill(trianglex, triangley, color=col, edgecolor="black", linewidth=2)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)  # add colorbar
        cbar.set_ticks(np.linspace(min_epsilon, max_epsilon, 6))
        plt.title("Refractive Index")
        plt.show()

    def plot_data_on_edges(self, mode: int = 0):
        # Plotting edge field magnitudes

        abs_field = np.abs(self.eigvecs[:, mode])

        edge_centres = self.barycenter(1)

        plt.scatter(*edge_centres.T, c=abs_field, cmap=self.cmap, s=10)
        plt.colorbar()

        """
        from scipy.interpolate import RBFInterpolator
    
        edge_centres_x, edge_centres_y = edge_centres.T
        plt.scatter(edge_centres_x, edge_centres_y, c=abs_field, cmap='plasma', s=10)
        plt.colorbar(label="|E| (arbitrary units)")
    
        # interpolation to create a smoother image
        rbf = RBFInterpolator(edge_centres, abs_field)
        def interp_func(pos):
            return rbf(pos)
    
        x = np.linspace(0., 1., 10)
        y = np.linspace(0., 1., 10)
        xy_axis_points = np.meshgrid(x, y)
        coord_points = np.array(xy_axis_points).reshape(2, -1).T
        print(coord_points)
        interp_func(coord_points)
    
        # attempt 2 at interpolation
        xobs = edge_centres
        yobs = np.log(abs_field)  # use the log of the value
        xgrid = np.mgrid[0:1:50j, 0:1:50j]
        xflat = xgrid.reshape(2, -1).T
        yflat = RBFInterpolator(xobs, yobs)(xflat)
        ygrid = yflat.reshape(50, 50)
        fig, ax = plt.subplots()
        ax.pcolormesh(*xgrid, ygrid, shading='gouraud')
        p = ax.scatter(*xobs.T, c=yobs, s=50, ec='k')
        fig.colorbar(p)
        plt.title("Mode Profile (Magnitude of E-field)")
        plt.axis('equal')
        plt.show()
        """

    def plot_data_shaded(self, mode: int = 0, simplex_type: int = 0, absolute_field: bool = False, real_field: bool = False):
        centres = self.barycenter(simplex_type)

        if real_field:
            field = self.eigvecs[:, mode].real
        else:
            field = self.eigvecs[:, mode]
        if absolute_field:
            field = np.abs(field)

        # normalise
        # field /= np.max(abs_field)

        plt.tripcolor(*centres.T, field, shading='gouraud', cmap=self.cmap)
        plt.title(f"Mode {mode} ({self.eigvals[mode]}) Profile")
        plt.colorbar()

    def plot_cross_section(self, mode: int, simplex_type: int, absolute_field: bool = False):
        centres = self.barycenter(simplex_type)
        abs_field = np.abs(self.eigvecs[:, mode])

        if absolute_field:
            field = abs_field
        else:
            field = self.eigvecs[:, mode]

        # normalise
        field /= np.max(abs_field)

        triang = tri.Triangulation(*centres.T)

        x_line = np.linspace(0., 1., 100)
        y_line = 0 * (x_line) + .5
        interp = tri.LinearTriInterpolator(triang, field)
        field_line = interp(x_line, y_line)

        plt.plot(x_line, field_line)
        plt.vlines([0.5 - self.core_radius, 0.5 + self.core_radius], min(field_line), max(field_line), color="black", linestyles='dashed', label="core boundary")
        plt.show()

    def plot_radial_cross_sections(self, modes: list, simplex_type: int = 0, absolute_field: bool = False, positive_start: bool = False, show_core_boundary: bool = True, scale_tolerance: float = 1e-3, min_radius: float = 0., max_radius: float = .5, scale_values: tuple = ()):
        centres = self.barycenter(simplex_type)

        # sample the eigenvectors from 0->core_radius
        x_line = np.linspace(.5 - max_radius, .5 - min_radius, 1000)
        y_line = 0 * (x_line) + .5

        min_y, max_y = 1., -1.
        for ind, mode in enumerate(modes):
            field = self.eigvecs[:, mode]

            if absolute_field:  # use the absolute values of the field
                field = np.abs(field)

            triang = tri.Triangulation(*centres.T)
            interp = tri.LinearTriInterpolator(triang, field)
            field_line = interp(x_line, y_line)

            # scale the field line to peak at 1, if its maximum value is greater than the normalise_max value
            abs_max = np.max(np.abs(field_line))
            if abs_max > scale_tolerance:
                # if an entry in the scale values exists for this, use it to scale the line
                if len(scale_values) >= ind:
                    field_line *= scale_values[ind]
                else:
                    field_line /= abs_max

            if positive_start:  # make the field positive near the centre (r=0)
                # sample the field at the first point that isn't zero (so the last point, so that it is closest to the 0 radius)
                # multiply by the sign of this value to make it positive
                field_line *= np.sign(field_line[-2])

            # reverse the y-values on the plot so that is plots from the centre outwards
            plt.plot(.5 - x_line, field_line, label=f"mode {mode}")

            # update the min and max y values
            min_y = min(np.min(field_line), min_y)
            max_y = max(np.max(field_line), max_y)

        if show_core_boundary:
            plt.vlines([self.core_radius], min_y, max_y, color="black", linestyles='dashed', label="core boundary")

        plt.legend()
        plt.show()

    def plot_data_on_vertices_shaded(self, mode: int = 0):
        abs_field = np.abs(self.eigvecs[:, mode])
        # print(self.K.vertices.shape, abs_field.shape, self.K[0].simplices)
        plt.tripcolor(*self.K.vertices.T, abs_field, shading='gouraud', cmap=self.cmap)
        plt.title(f"Mode {mode} ({self.eigvals[mode]}) Profile")
        plt.colorbar()

    def plot_data_on_edges_shaded(self, mode: int = 0):
        abs_field = np.abs(self.eigvecs[:, mode])
        edge_centres = self.barycenter(1)
        # print(self.K.vertices.shape, abs_field.shape, self.K[0].simplices)
        plt.tripcolor(*edge_centres.T, abs_field, shading='gouraud', cmap=self.cmap)
        plt.title(f"Mode {mode} ({self.eigvals[mode]}) Profile")
        plt.colorbar()

    def plot_data_on_vertices(self, mode: int = 0):
        # Plotting field magnitudes on the vertices
        abs_field = np.abs(self.eigvecs[:, mode])
        plt.scatter(*self.K.vertices.T, c=abs_field, cmap=self.cmap, s=10)

        plt.colorbar()

    def plot_streamlines(self, mode: int = 0):
        abs_field = np.abs(self.eigvecs[:, mode])

        # Define grid limits and resolution
        x_min, y_min = self.K.vertices.min(axis=0)
        x_max, y_max = self.K.vertices.max(axis=0)

        grid_x, grid_y = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )

        # Interpolate vector components to grid

        U = griddata(self.K.vertices, abs_field[:, 0], (grid_x, grid_y), method='cubic', fill_value=0)
        V = griddata(self.K.vertices, abs_field[:, 1], (grid_x, grid_y), method='cubic', fill_value=0)

        # Plot streamlines

        plt.figure(figsize=(8, 8))
        plt.streamplot(grid_x, grid_y, U, V, density=1.5, linewidth=1, arrowsize=1, arrowstyle='->')

    def plot_contours(self, mode: int = 0, simplex_type: int = 0):
        abs_field = np.abs(self.eigvecs[:, mode])
        centres = self.barycenter(simplex_type)

        # Interpolate scalar field onto a regular grid
        x_min, y_min = centres.min(axis=0)
        x_max, y_max = centres.max(axis=0)

        grid_x, grid_y = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300)
        )

        abs_grid = griddata(centres, abs_field, (grid_x, grid_y), method='cubic', fill_value=0)

        # Plot contour lines (lines of constant field magnitude)

        contours = plt.contour(grid_x, grid_y, abs_grid, levels=15, cmap=self.cmap)
        # plt.clabel(contours, inline=True, fontsize=8)

    def plot_mesh(self):
        # Create triangulation
        triang = tri.Triangulation(*self.K.vertices.T, self.K[2].simplices)

        # Plot
        plt.triplot(triang, color="black", linewidth=0.5)

    def plot(self, mode: int = 0, plot_types: tuple = ("vertices",), simplex_type: int = 0, absolute_field: bool = True, real_field: bool = False):
        if "shaded" in plot_types:
            self.plot_data_shaded(mode, simplex_type, absolute_field, real_field)
        if "vertices_shaded" in plot_types:
            self.plot_data_on_vertices_shaded(mode)
        if "edges_shaded" in plot_types:
            self.plot_data_on_edges_shaded(mode)
        if "contours" in plot_types:
            self.plot_contours(mode, simplex_type)
        if "mesh" in plot_types:
            self.plot_mesh()

        eigval = self.eigvals[mode]
        plt.title(f"Mode {mode} ({eigval})")
        plt.axis('equal')
        plt.show()

        if "cross_section" in plot_types:
            self.plot_cross_section(mode, simplex_type, absolute_field)

