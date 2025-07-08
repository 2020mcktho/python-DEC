import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from fontTools.varLib.interpolatableHelpers import min_cost_perfect_bipartite_matching
from pydec import simplicial_complex

# Step 1: Create a simple 2D mesh (square domain)
# In real use, you'd load a more realistic PCF mesh with air holes.
# from pydec.mesh.generation import simplicial_grid_2d
from my_generation_pydec import simplicial_grid_2d

def barycenter(sc: simplicial_complex, sc_index: int = 1):
    return np.average(sc.vertices[sc[sc_index].simplices], axis=1)

# Generate a square mesh with (n x n) subdivisions
n = 40
vertices, triangles = simplicial_grid_2d(n)

# Create simplicial complex
K = simplicial_complex(vertices, triangles)

# Step 2: Get DEC operators
# d0 = exterior_derivative(K, 0)  # 0-form to 1-form (grad)
# d1 = exterior_derivative(K, 1)  # 1-form to 2-form (curl)
d0 = K[0].d  # 0-form to 1-form
d1 = K[1].d  # 0-form to 1-form


# Step 3: Define permittivity (epsilon) distribution
# Example: circle in the center (high-index core)
core_radius = 0.4
core_eps, cladding_eps = 2.5, 1.

N1 = K[1].num_simplices
N2 = K[2].num_simplices
epsilon = np.ones(N2) * cladding_eps
epsilon_inv = np.ones(N2) / cladding_eps  # background cladding with epsilon=2.1
barycenter_points = barycenter(K, 2)
x, y = barycenter_points.T
r = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
epsilon[r < core_radius] = core_eps
epsilon_inv[r < core_radius] = 1 / core_eps  # core region with epsilon=2.5

# Step 4: Construct Hodge stars
# Hodge1 = hodge_star(K, 1, dual_type='barycentric', values=1.0)     # edge Hodge star (μ assumed 1)
# Hodge2 = hodge_star(K, 2, dual_type='barycentric', values=epsilon) # face Hodge star for ε
# edge Hodge star, assuming mu=1.0, square matrix with dimension of edge number
Hodge1 = K[1].star
Hodge1_inv = K[1].star_inv  # assume that mu=1.
# face Hodge star, using epsilon values calculated above, square matrix with dimension of vertex number
Hodge2 = K[2].star @ np.diagflat(epsilon)
Hodge2_inv = K[2].star_inv @ np.diagflat(epsilon_inv)
# Note: since the matrices are both diagonal, this essentially multiplies the elements together

# Step 5: Build the generalized eigenvalue problem
# We solve: A e = λ B e
# Note: the @ symbol does matrix multiplication
# A = d1.T @ spla.inv(Hodge2) @ d1  # could also use K[1].star_inv to get the inverse Hodge star
A = d1.T @ Hodge2_inv @ d1
B = Hodge1

# Step 6: Solve eigenvalue problem (λ = ω^2)
k = 6  # Number of modes
eigvals, eigvecs = spla.eigsh(A, k=k, M=B, sigma=1.5, which='LM')

print(eigvals)  # gives omega (or beta) values

# Step 7: Visualize the first mode
# Convert eigenvector back to edge fields

def plot_epsilon():
    # plotting
    fig, ax = plt.subplots()

    # plot the triangle edges
    # plt.triplot(vertices[:,0], vertices[:,1], triangles)

    # fill colour the triangles based on their epsilon value
    min_epsilon = np.min(epsilon)
    max_epsilon = np.max(epsilon)
    if min_epsilon == max_epsilon:
        min_epsilon -= 0.01
    cmap = plt.get_cmap("inferno")  # colour map
    norm = mcolors.BoundaryNorm((min_epsilon, max_epsilon), cmap.N, clip=True)  # normalisation
    for ind, k_vertices in enumerate(K[2].simplices):
        col = cmap(norm(epsilon[ind]))
        trianglex, triangley = K.vertices[k_vertices].T
        plt.fill(trianglex, triangley, color=col, edgecolor="black", linewidth=2)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)  # add colorbar
    cbar.set_ticks(np.linspace(min_epsilon, max_epsilon, 6))
    plt.title("Permittivity value")
    plt.show()

def plot_data(field_data):
    # Plotting edge field magnitudes
    from scipy.interpolate import RBFInterpolator

    abs_field = np.abs(field_data)

    edge_centres = barycenter(K, 1)

    """
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
    """

    xobs = edge_centres
    yobs = abs_field
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

plot_epsilon()
plot_data(eigvecs[:, 0])  # 0=first mode, 1=second mode, etc.
