from numpy import zeros, resize, arange, ravel, concatenate, matrix, transpose, int32


def simplicial_grid_2d(n):
    """
    Create an NxN 2d grid in the unit square

    The number of vertices along each axis is (N+1) for a total of (N+1)x(N+1) vertices

    A tuple (vertices,indices) of arrays is returned
    """
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