import numpy as np

TEST_INPUT_PART_B1 = np.array([1, 2, 3, 4])
TEST_INPUT_PART_B2 = np.array([0.1, 0.2, 0.3])
TEST_OUTPUT_PART_B = np.array([[1.1, 1.2, 1.3],
                               [2.1, 2.2, 2.3],
                               [3.1, 3.2, 3.3],
                               [4.1, 4.2, 4.3]])


def check_answer(predicted, actual):
    try:
        assert np.allclose(predicted, actual), "INCORRECT"
        print("CORRECT")
    except:
        print("INCORRECT")


def part_b(x, y):
    # TODO: Given a vector x (size (N,)), and an vector y (size (M,))
    # return an NxM matrix A (size (N,M)) where C[i,j] = x[i] + y[j]
    return x[:,np.newaxis] + y


check_answer(part_b(TEST_INPUT_PART_B1, TEST_INPUT_PART_B2), TEST_OUTPUT_PART_B)
