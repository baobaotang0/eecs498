import numpy as np


TEST_INPUT_PART_D0a = np.array([[0.63352971, 0.53577468, 0.09028977, 0.8353025],
 [0.32078006, 0.18651851, 0.04077514, 0.59089294],
 [0.67756436, 0.01658783, 0.51209306, 0.22649578],
 [0.64517279, 0.17436643, 0.69093774, 0.38673535],
 [0.93672999, 0.13752094, 0.34106635, 0.11347352]])
TEST_INPUT_PART_D0b = np.array([[0.92469362],
 [0.87733935],
 [0.25794163],
 [0.65998405]])
TEST_OUTPUT_PART_D0 = np.array([2, 0.719627281044947])

def check_answer(predicted, actual):
    try:
        assert np.allclose(predicted, actual), "INCORRECT"
        print("CORRECT")
    except:
        print("INCORRECT")

def part_d(A, x):
    # TODO: given M-D-dimensional vectors as
    cos_alpha = (A@x).flatten()/(np.linalg.norm(A, axis=1)*np.linalg.norm(x, axis=0))
    n = np.argmin(x, axis=0)[0]
    return [n, cos_alpha[n]]

check_answer(part_d(TEST_INPUT_PART_D0a, TEST_INPUT_PART_D0b), TEST_OUTPUT_PART_D0)