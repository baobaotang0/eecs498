import copy
import numpy as np

# Problem 1 (i)
def f_1(x, y, z):
    """
    Computes the forward and backward pass through the computational graph 
    of (i)

    Inputs:
    - x, y, z: Python floats

    Returns a tuple of:
    - L: The output of the graph
    - grads: A tuple (grad_x, grad_y, grad_z)
    giving the derivative of the output L with respect to each input.
    """
    L = None
    grad_x = None
    grad_y = None
    grad_z = None
    ###########################################################################
    # TODO: Implement the forward pass for the computational graph for (i) and#
    # store the output of this graph as L                                     #
    ###########################################################################
    y1 = copy.copy(y)
    y2 = copy.copy(y)
    p = x + y1
    q = y2 + z
    L = p * q
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################



    ###########################################################################
    # TODO: Implement the backward pass for the computational graph for (i)   #
    # Store the gradients for each input                                      #
    ###########################################################################
    grad_L = 1
    grad_p = grad_L * q
    grad_q = grad_L * p
    grad_x = grad_p * 1
    grad_y1 = grad_p * 1
    grad_z = grad_q * 1
    grad_y2 = grad_q * 1
    grad_y = grad_y1 + grad_y2
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    grads = (grad_x, grad_y, grad_z)
    return L, grads



# Problem 1 (iii)
def f_3(x0, x1, w0, w1, w2):
    """
    Computes the forward and backward pass through the computational graph 
    of (iii)

    Inputs:
    - x0, x1, w0, w1, w2: Python floats

    Returns a tuple of:
    - L: The output of the graph
    - grads: A tuple (grad_x0, grad_x1, grad_w0, grad_w1, grad_w2)
    giving the derivative of the output L with respect to each input.
    """
    L = None
    grad_x0 = None
    grad_x1 = None
    grad_w0 = None
    grad_w1 = None
    grad_w2 = None
    ###########################################################################
    # TODO: Implement the forward pass for the computational graph for (iii)  #
    # and store the output of this graph as L                                 #
    ###########################################################################
    a = x0 * w0
    b = x1 * w1
    c = a + b + w2
    d = c * -1
    e = np.exp(d)
    f = e + 1
    L = 1 / f
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################



    ###########################################################################
    # TODO: Implement the backward pass for the computational graph for (iii) #
    # Store the gradients for each input                                      #
    ###########################################################################
    grad_L = 1
    grad_f = -1 / (f ** 2) * grad_L
    grad_e = grad_f
    grad_d = np.exp(d) * grad_e
    grad_c = - grad_d
    grad_w2 = grad_c
    grad_a = grad_c
    grad_b = grad_c
    grad_w0 = grad_a * x0
    grad_w1 = grad_b * x1
    grad_x0 = grad_a * w0
    grad_x1 = grad_b * w1
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    grads = (grad_x0, grad_x1, grad_w0, grad_w1, grad_w2)
    return L, grads


# Problem 1 (iv)
def f_4(x, y, z):
    """
    Computes the forward and backward pass through the computational graph 
    of (iv)

    Inputs:
    - x, y, z: Python floats

    Returns a tuple of:
    - L: The output of the graph
    - grads: A tuple (grad_x, grad_y, grad_z)
    giving the derivative of the output L with respect to each input.
    """
    L = None
    grad_x = None
    grad_y = None
    grad_z = None
    ###########################################################################
    # TODO: Implement the forward pass for the computational graph for (iv)   #
    # and store the output of this graph as L                                 #
    ###########################################################################
    a = x * -1
    b = np.exp(y)
    c = np.exp(z)
    p = b * c
    e = c / p
    d = b + p
    m = a - d
    n = m / e
    L = n ** 2
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################



    ###########################################################################
    # TODO: Implement the backward pass for the computational graph for (iv)  #
    # Store the gradients for each input                                      #
    ###########################################################################
    grad_L = 1
    grad_n = 2 * n * grad_L
    grad_m = grad_n / e
    grad_e = grad_n * (-1 / e ** 2)
    grad_d = grad_m * -1
    grad_p = grad_d + grad_e * (-1 / p ** 2)
    grad_a = grad_m
    grad_b = grad_p * c
    grad_c = grad_p * b
    grad_x = grad_a * -1
    grad_y = grad_b * np.exp(y)
    grad_z = grad_c * np.exp(z)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    grads = (grad_x, grad_y, grad_z)
    return L, grads