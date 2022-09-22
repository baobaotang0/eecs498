import numpy as np
TEST_INPUT_PART_A = np.array([[0.37454012, 0.95071431, 0.73199394],
 [0.59865848, 0.15601864, 0.15599452],
 [0.05808361, 0.86617615, 0.60111501],
 [0.70807258, 0.02058449, 0.96990985],
 [0.83244264, 0.21233911, 0.18182497]])
TEST_OUTPUT_PART_A = np.array([2.05724837, 0.91067164, 1.52537477, 1.69856692, 1.22660672])

def check_answer(predicted, actual):
    try:
        assert np.allclose(predicted, actual), "INCORRECT"
        print("CORRECT")
    except:
        print("INCORRECT")

def part_a(A):
    x_norm = np.linalg.norm(A, ord=1, axis=1)
    print(x_norm)
    # TODO: given matrix A of size (N,M) return a vector
    # of size (N,) that consists of the l1-norm of each row
    return x_norm

check_answer(part_a(TEST_INPUT_PART_A), TEST_OUTPUT_PART_A)