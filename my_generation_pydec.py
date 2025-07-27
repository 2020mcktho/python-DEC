from numpy import zeros, resize, arange, ravel, concatenate, matrix, transpose, int32, cos, sin, pi
import pygmsh


def simplicial_grid_2d(mesh_size):
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


def disk_mesh(mesh_size):
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
    points, triangles = create_fibre_mesh(0.1, .25)

    plt.triplot(*points.T, triangles, linewidth=0.3)
    plt.gca().set_aspect("equal")
    plt.title("Fiber-Optimized Mesh with Radial Grading")
    plt.show()