import numpy as np
import operations as op

print("Please enter size of your Matrix : ")  # first enter size of n

n = int(input())
entries = []    # array for elements of matrix

for row in range(n):
    entries = entries + list(map(float, input().split()))     # input elements of each line
matrix_A = np.array(entries).reshape(n, n)      # create n * n array with numpy library by using reshape func
identity_matrix = np.identity(n)

complete_row = 0       # number of rows that find pivot in them
for column in range(n):

    pivot_r_position = op.det_nonzero_column(matrix_A, column, complete_row, n)  # find the first none_zero
    if pivot_r_position == -1:  # elements in specific column
        continue
    op.pivotPositions.append(tuple([complete_row, column]))
    temp_LU = op.LU_factorization(identity_matrix, matrix_A, n, complete_row, column)

    complete_row += 1

U_matrix = temp_LU[0]    # U factor of A
L_matrix = np.linalg.inv(temp_LU[1])     # L factor of A

print("\nU Factor : \n")
op.display_matrix(U_matrix, n)
print("\n\nL Factor : \n")
op.display_matrix(L_matrix, n)

identity_matrix = np.identity(n)
A_inverse = []

for column in range(n):
    vector_b = identity_matrix[column, :]
    y = op.solve_Lb_equation(L_matrix, vector_b, n)
    x = op.solve_Ux_equation(U_matrix, y, n)
    A_inverse.append(x)

A_inverse = np.transpose(np.array(A_inverse))

print("\nA Inverse : \n")
op.display_matrix(A_inverse, n)
