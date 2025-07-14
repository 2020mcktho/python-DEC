import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse import diags
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pydec import simplicial_complex

# Step 1: Create a simple 2D mesh (square domain)
# In real use, you'd load a more realistic PCF mesh with air holes.
# from pydec.mesh.generation import simplicial_grid_2d
from my_generation_pydec import simplicial_grid_2d


class FibreSolution:
    def __init__(self, sc: simplicial_complex = None, n: int = 20, core_radius: float = 0.4, core_n: complex = 3., cladding_n: complex = 1., buffer_size: float = 0.1):
        # if no simplicial complex mesh provided, generate a square mesh instead, using n divisions per side
        if sc is None:
            vertices, triangles = simplicial_grid_2d(n)
            # Create simplicial complex
            self.K = simplicial_complex(vertices, triangles)

        # Example: circle in the center (high-index core)
        self.core_radius = core_radius
        self.core_n = core_n
        self.cladding_n = cladding_n

        self.buffer_size = buffer_size

        self.Hodges = [[self.K[0].star, self.K[0].star_inv],
                       [self.K[1].star, self.K[1].star_inv],
                       [self.K[2].star, self.K[2].star_inv]]
        self.n_vals = np.array([1], dtype=complex)
        self.n_mat = diags(self.n_vals)
        self.n_mat_inv = diags(self.n_vals)
        self.eigvals, self.eigvecs = np.array([]), np.array([])
        self.use_pml = False
        self.setup()

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

    def create_n_geometry(self):
        # create a matrix of refractive index values at each vertex
        # these will be averaged to find the values for the edges or faces
        self.n_vals = np.ones(len(self.K.vertices), dtype=complex) * self.cladding_n

        barycenter_points = self.barycenter(0)
        x, y = barycenter_points.T
        r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        self.n_vals[r < self.core_radius] = self.core_n

    def create_n_matrix(self, sc_index: int = 1, merge_type: str = "average"):
        if merge_type == "average":  # use the average of the surrounding points
            self.n_mat = diags(np.average(self.n_vals[self.K[sc_index].simplices], axis=1))
            self.n_mat_inv = diags(1 / np.average(self.n_vals[self.K[sc_index].simplices], axis=1))
        else:  # use the max value instead
            self.n_mat = diags(np.max(self.n_vals[self.K[sc_index].simplices], axis=1))
            self.n_mat_inv = diags(1 / np.max(self.n_vals[self.K[sc_index].simplices], axis=1))

    def apply_n_matrix(self, sc_index: int = 1):
        self.Hodges[sc_index][0] = self.K[sc_index].star @ (self.n_mat_inv ** 2)
        self.Hodges[sc_index][1] = self.K[sc_index].star_inv @ (self.n_mat ** 2)

    def create_pml_vertex_buffer(self, max_imaginary_index: float = .1):
        # max_imaginary_index controls the decay speed inside the buffer

        # find the points within the buffer zone at the edge (for the perfectly matched layer)
        x, y = self.K.vertices.T
        dist_to_edge = np.min([x, 1. - x, y, 1. - y], axis=0)
        in_buffer = dist_to_edge < self.buffer_size
        # Add the complex loss part to the refractive index inside the Perfectly Matched Layer
        absorption = np.zeros_like(self.n_vals)
        absorption[in_buffer] = 1j * (((1 - dist_to_edge[in_buffer] / self.buffer_size) ** 2) * max_imaginary_index)

        self.n_vals += absorption  # add the complex part to the vertices inside the buffer

    def setup(self, epsilon_sc_index: int = 0, merge_type: str = "average", use_pml: bool = False):
        # Set up the fibre geometry and different sections of refractive index
        # Also set up the perfectly matched layer, if it is being used

        self.use_pml = use_pml

        self.create_n_geometry()
        if self.use_pml:
            self.create_pml_vertex_buffer()
        self.create_n_matrix(epsilon_sc_index, merge_type)
        self.apply_n_matrix(epsilon_sc_index)

    def solve_with_dirichlet(self, A: np.ndarray, B: np.ndarray, boundary_type: int = 1, mode_number: int = 1, eigval_pref: str = "LM", real_matrix: bool = True):
        # Find boundary and internal edges
        if boundary_type == 1:  # edges on the boundary
            boundary_indices, internal_indices = self.get_ext_and_int_edge_ind()
        else:  # vertices on the boundary
            boundary_indices, internal_indices = self.get_ext_and_int_vertex_ind()

        # Restrict A and B to interior edge DoFs
        A_reduced = A[internal_indices, :][:, internal_indices]
        B_reduced = B[internal_indices, :][:, internal_indices]

        # Solve the reduced eigenproblem: A e = λ B e
        # if a perfectly matched layer is being used, this will always require the complex solver
        if real_matrix and not self.use_pml:  # when the matrix is known to be symmetric or Hermitian
            eigvals, eigvecs_reduced = spla.eigsh(A_reduced, k=mode_number, M=B_reduced, sigma=1.0, which=eigval_pref)
        else:  # when the matrix may be complex
            eigvals, eigvecs_reduced = spla.eigs(A_reduced, k=mode_number, M=B_reduced, sigma=1.0, which=eigval_pref)

        # Sort the eigenpairs (sometimes eigsh returns unordered)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs_reduced = eigvecs_reduced[:, idx]

        # Reconstruct full-length eigenvectors with zeros on boundary
        dimension = self.K[boundary_type].num_simplices  # number of points in the mesh of this type
        eigvecs_full = np.zeros((dimension, mode_number))
        eigvecs_full[internal_indices, :] = eigvecs_reduced

        self.eigvals, self.eigvecs = eigvals, eigvecs_full
        return self.eigvals, self.eigvecs

    def solve(self, A: np.ndarray, B: np.ndarray | None = None, mode_number: int = 1, eigval_pref: str = "LM", real_matrix: bool = True):
        # Solve the reduced eigenproblem: A e = λ B e
        if real_matrix and not self.use_pml:  # when the matrix is known to be symmetric or Hermitian
            eigvals, eigvecs = spla.eigsh(A, k=mode_number, M=B, sigma=1.0, which=eigval_pref)
        else:  # when the matrix may be complex
            eigvals, eigvecs= spla.eigs(A, k=mode_number, M=B, sigma=1.0, which=eigval_pref)

        # Sort the eigenpairs (sometimes eigsh returns unordered)
        idx = np.argsort(eigvals)
        self.eigvals = eigvals[idx]
        self.eigvecs = eigvecs[:, idx]

        return self.eigvals, self.eigvecs

    def plot_n_shaded(self):
        mode = np.abs(self.n_vals)  # shade using the epsilon values
        plt.tripcolor(*self.K.vertices.T, mode, shading='gouraud')
        plt.title("Fundamental Mode Profile")
        plt.colorbar()
        plt.show()

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

        plt.scatter(*edge_centres.T, c=abs_field, cmap='viridis', s=10)
        plt.title("First Eigenmode on Unit Square (Dirichlet BC)")
        plt.axis('equal')
        plt.colorbar()
        plt.show()

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

    def plot_data_on_vertices_shaded(self, mode: int = 0):
        abs_field = np.abs(self.eigvecs[:, mode])
        plt.tripcolor(*self.K.vertices.T, abs_field, shading='gouraud')
        plt.title("Fundamental Mode Profile")
        plt.colorbar()
        plt.show()

    def plot_data_on_vertices(self, mode: int = 0):
        # Plotting field magnitudes on the vertices
        abs_field = np.abs(self.eigvecs[:, mode])
        plt.scatter(*self.K.vertices.T, c=abs_field, cmap='viridis', s=10)
        plt.title("First Eigenmode on Unit Square (Dirichlet BC)")
        plt.axis('equal')
        plt.colorbar()
        plt.show()
