import numpy as np


TEST_INPUT_PART_C1 = np.array(
[[0.00552212, 0.81546143, 0.70685734],
 [0.72900717, 0.77127035, 0.07404465],
 [0.35846573, 0.11586906, 0.86310343],
 [0.62329813, 0.33089802, 0.06355835]]
)
TEST_INPUT_PART_C2 = np.array(
[[0.31098232, 0.32518332, 0.72960618],
 [0.63755747, 0.88721274, 0.47221493],
 [0.11959425, 0.71324479, 0.76078505],
 [0.5612772,  0.77096718, 0.4937956 ],
 [0.52273283, 0.42754102, 0.02541913],
 [0.10789143, 0.03142919, 0.63641041]]
)
TEST_OUTPUT_PART_C = np.array([2, 1, 0, 1, 3, 2])



def check_answer(predicted, actual):
    try:
        assert np.allclose(predicted, actual), "INCORRECT"
        print("CORRECT")
    except:
        print("INCORRECT")

def part_c(A, B):
    # TODO: fix the buggy solution below
    x = ((np.linalg.norm(A, axis=1)**2)[:,np.newaxis] + (np.linalg.norm(B, axis=1)**2) - 2 * np.matmul(A, B.T))
    return np.argmin(x, axis=0)

check_answer(part_c(TEST_INPUT_PART_C1, TEST_INPUT_PART_C2), TEST_OUTPUT_PART_C)