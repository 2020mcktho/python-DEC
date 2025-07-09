import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pydec import simplicial_complex

# Step 1: Create a simple 2D mesh (square domain)
# In real use, you'd load a more realistic PCF mesh with air holes.
# from pydec.mesh.generation import simplicial_grid_2d
from my_generation_pydec import simplicial_grid_2d


class FibreSolution:
    def __init__(self, sc: simplicial_complex = None, n: int = 20, core_radius: float = 0.4, core_eps: float = 3., cladding_eps: float = 1.):
        # if no simplicial complex mesh provided, generate a square mesh instead, using n divisions per side
        if sc is None:
            vertices, triangles = simplicial_grid_2d(n)
            # Create simplicial complex
            self.K = simplicial_complex(vertices, triangles)

        # Example: circle in the center (high-index core)
        self.core_radius = core_radius
        self.core_eps = core_eps
        self.cladding_eps = cladding_eps

        self.Hodges = [[self.K[0].star, self.K[0].star_inv],
                       [self.K[1].star, self.K[1].star_inv],
                       [self.K[2].star, self.K[2].star_inv]]
        self.epsilon = np.array([1])
        self.eigvals, self.eigvecs = np.array([]), np.array([])
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

    def apply_epsilon(self, sc_index: int = 1):
        # create a matrix of epsilon values at each vertex
        # these will be averaged to find the epsilon values for the edges or faces
        self.epsilon = np.ones(len(self.K.vertices)) * self.cladding_eps

        barycenter_points = self.barycenter(0)
        x, y = barycenter_points.T
        r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        self.epsilon[r < self.core_radius] = self.core_eps

        edge_epsilons = np.average(self.epsilon[self.K[sc_index].simplices], axis=1)
        self.Hodges[sc_index][0] @= np.diagflat(edge_epsilons)
        self.Hodges[sc_index][1] @= np.diagflat(1 / edge_epsilons)
        # for ind, vertices in enumerate(self.K[sc_index].simplices):
        #     avg_epsilon = np.average(self.epsilon[vertices])
        #     self.Hodges[sc_index][0][ind, ind] *= avg_epsilon
        #     self.Hodges[sc_index][1][ind, ind] /= avg_epsilon

        # modify the Hodge stars
        # self.Hodges[sc_index][0] = self.Hodges[sc_index][0] @ np.diagflat(self.epsilon)
        # self.Hodges[sc_index][1] = self.Hodges[sc_index][1] @ np.diagflat(self.epsilon_inv)

    def setup(self, epsilon_sc_index: int = 2):
        # Step 2: Define permittivity (epsilon) distribution

        N0 = self.K[0].num_simplices  # number of vertices
        N1 = self.K[1].num_simplices  # number of edges
        N2 = self.K[2].num_simplices  # number of faces

        self.apply_epsilon(epsilon_sc_index)

    def solve_with_dirichlet(self, A: np.ndarray, B: np.ndarray, boundary_type: int = 1):
        # Find boundary and internal edges
        if boundary_type == 1:  # edges on the boundary
            boundary_indices, internal_indices = self.get_ext_and_int_edge_ind()
        else:  # vertices on the boundary
            boundary_indices, internal_indices = self.get_ext_and_int_vertex_ind()

        # Restrict A and B to interior edge DoFs
        A_reduced = A[internal_indices, :][:, internal_indices]
        B_reduced = B[internal_indices, :][:, internal_indices]

        # Solve the reduced eigenproblem: A e = λ B e
        k = 6  # Number of modes
        eigvals, eigvecs_reduced = spla.eigsh(A_reduced, k=k, M=B_reduced, sigma=1.0, which='LM')

        # Optional: sort the eigenpairs (sometimes eigsh returns unordered)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs_reduced = eigvecs_reduced[:, idx]

        # Reconstruct full-length eigenvectors with zeros on boundary
        dimension = self.K[boundary_type].num_simplices  # number of points in the mesh of this type
        eigvecs_full = np.zeros((dimension, k))
        eigvecs_full[internal_indices, :] = eigvecs_reduced

        self.eigvals, self.eigvecs = eigvals, eigvecs_full
        return self.eigvals, self.eigvecs

    def plot_epsilon(self):
        # plotting
        fig, ax = plt.subplots()

        # plot the triangle edges
        # plt.triplot(vertices[:,0], vertices[:,1], triangles)

        # fill colour the triangles based on their epsilon value
        min_epsilon = np.min(self.epsilon)
        max_epsilon = np.max(self.epsilon)
        if min_epsilon == max_epsilon:
            min_epsilon -= 0.01
        cmap = plt.get_cmap("inferno")  # colour map
        norm = mcolors.BoundaryNorm((min_epsilon, max_epsilon), cmap.N, clip=True)  # normalisation
        for ind, k_vertices in enumerate(self.K[2].simplices):
            if len(self.epsilon) == len(self.K.vertices):  # epsilon defined on the vertices
                epsilon_val = np.average(self.epsilon[k_vertices])
            else:  # epsilon defined on the faces
                epsilon_val = self.epsilon[ind]
            col = cmap(norm(epsilon_val))
            trianglex, triangley = self.K.vertices[k_vertices].T
            plt.fill(trianglex, triangley, color=col, edgecolor="black", linewidth=2)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)  # add colorbar
        cbar.set_ticks(np.linspace(min_epsilon, max_epsilon, 6))
        plt.title("Permittivity value")
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


    def plot_data_on_vertices(self, mode: int = 0):
        # Plotting field magnitudes on the vertices
        abs_field = np.abs(self.eigvecs[:, mode])
        plt.scatter(*self.K.vertices.T, c=abs_field, cmap='viridis', s=10)
        plt.title("First Eigenmode on Unit Square (Dirichlet BC)")
        plt.axis('equal')
        plt.colorbar()
        plt.show()
