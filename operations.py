import numpy as np


def display_matrix(matrix, size):      # function for display a matrix from 2D array
    for row in range(size):
        for column in range(size):
            if round(matrix[row][column], 2) - int(matrix[row][column]) == 0:
                print("%9.0d" % matrix[row][column], end='')

            else:
                print("%9.3f" % matrix[row][column], end='')
        print("\n")


def det_nonzero_column(matrix, column, start_row, size):   # this function detect of existence of a nonzero
    for row in range(start_row, size):                       # element in specific column to change rows
        if matrix[row][column] != 0:
            return row
    else:
        return -1


def LU_factorization(identity_matrix, matrix_A, size, pivot_row, pivot_column):   # this function factorize A to L and U matrix
    for row in range(pivot_row + 1, size):
        if matrix_A[row][pivot_column] != 0:
            scale = (-1) * (matrix_A[row][pivot_column] / matrix_A[pivot_row][pivot_column])
            for column in range(size):
                matrix_A[row][column] = (scale * matrix_A[pivot_row][column]) + matrix_A[row][column]
                identity_matrix[row][column] = (scale * identity_matrix[pivot_row][column]) + identity_matrix[row][column]
    return matrix_A, identity_matrix


pivotPositions = []    # list of pivot positions


def row_replacements_forward(augmented_matrix_Lb, size, pivot_row, pivot_column):   # this function make 0
    for row in range(pivot_row + 1, size):                             # elements that below the pivot positions
        if augmented_matrix_Lb[row][pivot_column] != 0:
            scale = (-1) * (augmented_matrix_Lb[row][pivot_column] / augmented_matrix_Lb[pivot_row][pivot_column])
            for column in range(size+1):
                augmented_matrix_Lb[row][column] = (scale * augmented_matrix_Lb[pivot_row][column]) + augmented_matrix_Lb[row][column]
    return augmented_matrix_Lb


def solve_Lb_equation(L_matrix, b, size):
    complete_row = 0
    augmented_matrix_Lb = np.column_stack((L_matrix, np.array(b)))  # create augmented matrix of [L | b] by using column stack
    for column in range(size+1):
        pivot_r_position = det_nonzero_column(augmented_matrix_Lb, column, complete_row, size)
        if pivot_r_position == -1:
            continue
        augmented_matrix_Lb = row_replacements_forward(augmented_matrix_Lb, size, complete_row, column)
        complete_row += 1
    y = np.array(augmented_matrix_Lb[:, -1])
    return y


def scale_pivots(matrix, size, pivot_row, pivot_column):  # this function make 1 pivot positions
    scale = matrix[pivot_row][pivot_column]
    if scale != 1:
        for column in range(size + 1):
            matrix[pivot_row][column] = matrix[pivot_row][column] / scale
    return matrix


def row_replacements_backward(augmented_matrix_Uy, size, pivot_row, pivot_column):      # this function make 0
    for row in range(pivot_row):                                           # elements that above the pivot positions
        scale = (-1) * augmented_matrix_Uy[row][pivot_column]
        for column in range(size+1):
            augmented_matrix_Uy[row][column] = (scale * augmented_matrix_Uy[pivot_row][column]) + augmented_matrix_Uy[row][column]
    return augmented_matrix_Uy


def solve_Ux_equation(U_matrix, y, size):
    augmented_matrix_Uy = np.column_stack((U_matrix, np.array(y)))  # create augmented matrix of [U | y] by using column stack

    for pivot_row, pivot_column in pivotPositions:  # this section make 1 pivot positions
        augmented_matrix_Uy = scale_pivots(augmented_matrix_Uy, size, pivot_row, pivot_column)

    for pivot_row, pivot_column in reversed(pivotPositions):  # this section make 0 elements that above the pivot positions
        augmented_matrix_Uy = row_replacements_backward(augmented_matrix_Uy, size, pivot_row, pivot_column)
    x = np.array(augmented_matrix_Uy[:, -1])
    return x

