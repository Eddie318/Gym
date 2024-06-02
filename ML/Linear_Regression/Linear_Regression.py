import numpy as np

def cost_MSE(x, y, w, b):
        """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    total_cost = 0
    m = x.shape[0]
    for i in range(m):
        f_wb = w * x[i] + b
        sq = (f_wb - y[i]) ** 2
        total_cost += sq
    total_cost = total_cost * (1 / (2 * m))

    return total_cost

def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    m = x.shape[0]
    dJ_dw = 0
    dJ_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dJ_dw_i = (f_wb - y[i]) * x[i]
        dJ_db_i = (f_wb - y[i])
        dJ_dw += dJ_dw_i
        dJ_db += dJ_db_i
    dJ_dw = dJ_dw / m
    dJ_db = dJ_db / m

    return dJ_dw, dJ_db 

def gradient_decent(alpha, iter, x, y, w_in, b_in, cost_function, gradient_function):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      """
    w = copy.deepcopy(w_in) # avoid modifying global w_in
    b = b_in
    w = w_in

    for i in range(iter):
         dJ_dw, dJ_db = gradient_function(x,y,w,b)
         w = w - alpha * dJ_dw
         b = b - alpha * dJ_db
    return w, b

