"""
trainingdata - Functions useful for creating and manipulating training data.
"""


from itertools import repeat


def create_training_grid(n):
    """Create a grid of training data. The input n is a vector containing the
    numbers of evenly-spaced data points to use in each dimension. For example,
    for an (x, y, z) grid, with n = [3, 4, 5], we will get a grid with 3 points
    along the x-axis, 4 points along the y-axis, and 5 points along the z-axis,
    for a total of 3*4*5 = 60 points. The points along each dimension
    are evenly spaced in the range [0, 1]. When there is m=1 dimension, a list
    is returned. When m>1, a list of lists is returned. """

    # Determine the number of dimensions in the result.
    m = len(n)

    # Handle 1-D and (n>1)-D cases differently.
    if m == 1:
        X = [i/(n[0] - 1) for i in range(n[0])]
    else:
        # Compute the evenly-spaced points along each dimension.
        x = [[i/(nn - 1) for i in range(nn)] for nn in n]

        # Assemble all possible point combinations.
        X = []
        p1 = None
        p2 = 1
        for j in range(m - 1):
            p1 = prod(n[j + 1:])
            XX = [xx for item in x[j] for xx in repeat(item, p1)]*p2
            X.append(XX)
            p2 *= n[j]
        X.append(x[-1]*p2)
        X = list(zip(*X))

    # Return the list of training points.
    return X

def prod(n):
    """Compute the product of the elements of a list."""
    p = 1
    for nn in n:
        p *= nn
    return p


if __name__ == '__main__':

    # Point counts along 1, 2, 3, 4 dimensions.
    n1 = [3]
    n2 = [3, 4]
    n3 = [3, 4, 5]
    n4 = [3, 4, 5, 6]

    # Reference values for tests
    p1_ref, p2_ref, p3_ref, p4_ref = (3, 12, 60, 360)

    print('Testing point counts.')
    assert prod(n1) == p1_ref
    assert prod(n2) == p2_ref
    assert prod(n3) == p3_ref
    assert prod(n4) == p4_ref

    print('Testing grid creatiion.')
    X1 = create_training_grid(n1)
    assert len(X1) == prod(n1)
    X2 = create_training_grid(n2)
    assert len(X2) == prod(n2)
    X3 = create_training_grid(n3)
    assert len(X3) == prod(n3)
    X4 = create_training_grid(n4)
    assert len(X4) == prod(n4)
