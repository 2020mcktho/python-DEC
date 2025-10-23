from numpy import array, zeros, resize, arange, ravel, concatenate, matrix, transpose, int32, cos, sin, pi, ceil, sqrt, clip, random
import pygmsh
from scipy.spatial import Delaunay


def simplicial_grid_2d(mesh_size, core_radius):
    """
    Create an NxN 2d grid in the unit square

    The number of vertices along each axis is (N+1) for a total of (N+1)x(N+1) vertices

    A tuple (vertices,indices) of arrays is returned
    """
    n = int(1. / mesh_size)
    vertices = zeros(((n + 1) ** 2, 2))
    vertices[:, 0] = ravel(resize(arange(n + 1), (n + 1, n + 1)))
    vertices[:, 1] = ravel(transpose(resize(arange(n + 1), (n + 1, n + 1))))
    vertices /= n

    indices = zeros((2 * (n ** 2), 3), int32)

    t1 = transpose(concatenate((matrix(arange(n)), matrix(arange(1, n + 1)), matrix(arange(n + 2, 2 * n + 2))), axis=0))
    t2 = transpose(
        concatenate((matrix(arange(n)), matrix(arange(n + 2, 2 * n + 2)), matrix(arange(n + 1, 2 * n + 1))), axis=0))
    first_row = concatenate((t1, t2))

    for i in range(n):
        indices[(2 * n * i):(2 * n * (i + 1)), :] = first_row + i * (n + 1)

    return (vertices, indices)


def create_square_mesh_delaunay(mesh_size, core_radius):
    mesh_points = [(.5, .5)]  # start with the centre point in the list

    # create points around the boundary box
    for x in arange(0., 1. + mesh_size, mesh_size):
        for y in arange(0., 1. + mesh_size, mesh_size):
            mesh_points.append((x, y))

    mesh_points = array(mesh_points)
    tri = Delaunay(mesh_points)
    simplices = tri.simplices

    return mesh_points, simplices


def create_fibre_mesh(mesh_size, core_radius):
    with pygmsh.geo.Geometry() as geom:
        # geom.add_box(0., 1., 0., 1., 0., 1., mesh_size)

        R = 1.0  # unit circle
        geom.add_circle([0.0, 0.0, 0.0], R, mesh_size=mesh_size)
        mesh = geom.generate_mesh()

    vertex_list = mesh.points[:, :2]  # Strip z, keep [x, y]
    triangle_list = mesh.cells_dict["triangle"]

    print("First few vertices:", vertex_list[:5])
    print("First few triangles:", triangle_list[:5])

    return vertex_list, triangle_list


def create_fibre_mesh_delaunay(mesh_size, core_radius):
    mesh_points = []

    # create a regular square mesh from 0-1
    for x in arange(0., 1. + mesh_size, mesh_size):
        for y in arange(0., 1. + mesh_size, mesh_size):
            mesh_points.append((x, y))

    # create points on the surface of the core circle
    centre_circle = (.5, .5)
    circle_circ = 2 * pi * core_radius
    n_circle_points = int(ceil(circle_circ / mesh_size))
    for i in range(n_circle_points):
        theta = 2 * pi * i / n_circle_points
        x = centre_circle[0] + core_radius * cos(theta)
        y = centre_circle[1] + core_radius * sin(theta)
        if (x, y) not in mesh_points:
            mesh_points.append((x, y))

    mesh_points = array(mesh_points)
    tri = Delaunay(mesh_points)
    simplices = tri.simplices

    return mesh_points, simplices


def create_uniform_circular_fibre_mesh_delaunay(mesh_size, core_radius):
    mesh_points = [(.5, .5)]  # start with the centre point in the list

    # create points around the boundary box
    for x in arange(0., 1. + mesh_size, mesh_size):
        for y in arange(0., 1. + mesh_size, mesh_size):
            if x == 0 or x == 1 or y == 0 or y == 1:
                mesh_points.append((x, y))

    # create circles going outwards, with one circle on the core boundary

    # create points on the surface of the core circle
    centre_circle = (.5, .5)
    # diagonal distance to corner = sqrt(.5**2 + .5**2) = .5sqrt(2)
    mesh_size *= 2
    for circle_radius in arange(mesh_size, .5 * sqrt(2), mesh_size):
        circle_circ = 2 * pi * circle_radius
        n_circle_points = int(ceil(circle_circ / mesh_size))
        for i in range(n_circle_points):
            theta = 2 * pi * i / n_circle_points
            x = centre_circle[0] + circle_radius * cos(theta)
            y = centre_circle[1] + circle_radius * sin(theta)
            if 0 <= x <= 1 and 0 <= y <= 1:
                mesh_points.append((x, y))

    mesh_points = array(mesh_points)
    tri = Delaunay(mesh_points)
    simplices = tri.simplices

    return mesh_points, simplices


def create_adaptive_circular_fibre_mesh_delaunay(mesh_size, core_radius, max_mesh_size=.1, mesh_boundary_change_dist=.1, use_central_point: bool = True):
    mesh_points = []

    # start with the centre point in the list
    if use_central_point:
        mesh_points.append((.5, .5))

    # create circles going inwards, starting on the core boundary
    # mesh_size determines the minimum mesh size
    # create the circle radii to use, and then calculate mesh size for that layer

    def mesh_size_calc(radius):
        percentage = abs(radius - core_radius) / mesh_boundary_change_dist
        return clip(mesh_size + percentage * (max_mesh_size - mesh_size), mesh_size, max_mesh_size)

    circle_radius = core_radius
    centre_circle = (.5, .5)

    def add_circle_points(current_radius, current_use_mesh_size):
        circle_circ = 2 * pi * current_radius
        n_circle_points = int(ceil(circle_circ / current_use_mesh_size))
        for i in range(n_circle_points):
            theta = 2 * pi * i / n_circle_points
            x = centre_circle[0] + circle_radius * cos(theta)
            y = centre_circle[1] + circle_radius * sin(theta)
            if 0 <= x <= 1 and 0 <= y <= 1:
                mesh_points.append((x, y))

    while circle_radius > 0:
        # calculate mesh size based on the distance to the core radius
        use_mesh_size = mesh_size_calc(circle_radius)
        # add points
        add_circle_points(circle_radius, use_mesh_size)
        # add to the overall radius covered
        circle_radius -= use_mesh_size

    # add a circle of points on the radius distance, using the minimum mesh size
    add_circle_points(circle_radius, mesh_size)

    mesh_points = array(mesh_points)
    tri = Delaunay(mesh_points)
    simplices = tri.simplices

    return mesh_points, simplices


def disk_mesh(mesh_size, core_radius):
    with pygmsh.occ.Geometry() as geom:
        R = 1.0  # unit circle
        circle1 = geom.add_disk([0.0, 0.0, 0.0], R, mesh_size=mesh_size)
        # circle2 = geom.add_disk([0.0, 0.0, 0.0], R/2, mesh_size=mesh_size)
        geom.add_point([0.0, 0.0, 0.0], mesh_size=mesh_size / 2)
        # geom.boolean_union([circle1, circle2])
        geom.synchronize()
        mesh = geom.generate_mesh(dim=2)

    # `mesh.points` is (N, 3): x, y, z
    # `mesh.cells_dict["triangle"]` is (M, 3): triangle vertex indices

    vertex_list = mesh.points[:, :2]  # Strip z, keep [x, y]
    triangle_list = mesh.cells_dict["triangle"]

    print("First few vertices:", vertex_list[:5])
    print("First few triangles:", triangle_list[:5])

    return vertex_list, triangle_list

def circular_geom_mesh(mesh_size, core_radius):
    with pygmsh.geo.Geometry() as geom:
        # Fiber parameters (in microns or your length unit)
        r_core = core_radius / 2
        r_clad = 1. - core_radius

        # Mesh sizes
        h_core = mesh_size / 2
        h_clad = mesh_size

        center = geom.add_point([0, 0, 0], mesh_size=h_core)

        # Outer boundary points (to define circle)
        boundary_pts = []
        n_boundary_pts = 100
        for i in range(n_boundary_pts):
            theta = 2 * pi * i / n_boundary_pts
            x = r_clad * cos(theta)
            y = r_clad * sin(theta)
            pt = geom.add_point([x, y, 0], mesh_size=h_clad)
            boundary_pts.append(pt)

        # Create circle arcs to close the outer boundary
        circle_lines = []
        for i in range(n_boundary_pts):
            start = boundary_pts[i]
            end = boundary_pts[(i + 1) % n_boundary_pts]
            circle_lines.append(geom.add_line(start, end))
        outer_loop = geom.add_curve_loop(circle_lines)
        outer_surface = geom.add_plane_surface(outer_loop)
        """
        # Define a Distance field to center for mesh size grading
        dist_field = geom.add_distance_field([center])

        # Define Threshold field to specify mesh sizes by distance
        threshold = geom.add_threshold_field(
            dist_field,
            dist_min=0.0,
            dist_max=r_core,
            size_min=h_core,
            size_max=h_clad,
        )

        # Set the background mesh field (enables graded mesh)
        geom.set_background_mesh(threshold)"""

        geom.synchronize()
        mesh = geom.generate_mesh()

    vertex_list = mesh.points[:, :2]  # Strip z, keep [x, y]
    triangle_list = mesh.cells_dict["triangle"]

    return vertex_list, triangle_list


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # points, triangles = circular_geom_mesh(0.1, 0.25)
    # points, triangles = create_fibre_mesh(0.1, .25)
    # points, triangles = create_fibre_mesh_delaunay(0.1, .25)
    # points, triangles = create_uniform_circular_fibre_mesh_delaunay(0.05, .25)
    points, triangles = create_adaptive_circular_fibre_mesh_delaunay(.01, .2, .05, .1)

    plt.triplot(*points.T, triangles, linewidth=0.3, marker="o", markersize=1)
    plt.gca().set_aspect("equal")
    plt.title("Triangulated mesh")
    plt.show()