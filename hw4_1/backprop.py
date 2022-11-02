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

    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################



    ###########################################################################
    # TODO: Implement the backward pass for the computational graph for (i)   #
    # Store the gradients for each input                                      #
    ###########################################################################

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

    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################



    ###########################################################################
    # TODO: Implement the backward pass for the computational graph for (iii) #
    # Store the gradients for each input                                      #
    ###########################################################################

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

    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################



    ###########################################################################
    # TODO: Implement the backward pass for the computational graph for (iv)  #
    # Store the gradients for each input                                      #
    ###########################################################################

    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    grads = (grad_x, grad_y, grad_z)
    return L, grads