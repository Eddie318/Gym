import math, copy
import numpy as np
import pandas as pd


def cost_MSE(x, y, w, b):
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,n)): Data, m examples with n features 
      y (ndarray (m,)): target values
      w (ndarry (n,)): multivariate
      b (scalar)    : model parameter
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """

    total_cost = 0
    m = x.shape[0]
    for i in range(m):
        #f_wb = w * x[i] + b
        f_wb = np.dot(x[i], w) + b
        sq = (f_wb - y[i]) ** 2
        total_cost += sq
    total_cost = total_cost * (1 / (2 * m))

    return total_cost

def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): multivariate
      b (scalar)    : model parameter 
    Returns
      dj_dw (ndarray (n,)): The gradients of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    m,n = x.shape[0]
    dJ_dw = n
    dJ_db = 0
    for i in range(m):
        #f_wb = w * x[i] + b
        err = (np.dot(x[i], w) + b)- y[i]
        for j in range(n):
          dJ_dw[j] += err * x[i,j]
        dJ_db += err
    dJ_dw = dJ_dw / m
    dJ_db = dJ_db / m

    return dJ_dw, dJ_db 

def gradient_decent(alpha, iter, x, y, w_in, b_in, cost_function, gradient_function):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,n))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in (ndarray (n,))  : initial values of model parameters
      b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (ndarray (n,)): Updated value of parameter after running gradient descent
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

# Load the dataset
file_path = r'C:\Users\Administrator\Gym\ML\Linear_Regression\pokemon.csv'
df = pd.read_csv(file_path)

'''
# Convert to numpy arrays
Mileage = df['Mileage'].to_numpy()
Price = df['Price'].to_numpy()

# Scale the data
Mileage_mean = np.mean(Mileage)
Mileage_std = np.std(Mileage)
Price_mean = np.mean(Price)
Price_std = np.std(Price)

Mileage_scaled = (Mileage - Mileage_mean) / Mileage_std
Price_scaled = (Price - Price_mean) / Price_std

# Gradient Descent
learning_rate = 1.0e-3
iterations = 100000
w_init = 0
b_init = 0


w, b = gradient_decent(learning_rate, iterations, Mileage_scaled, Price_scaled, w_init, b_init, cost_MSE, compute_gradient)
print(f"w: {w}, b: {b}")
# Unscale the parameters
w_unscaled = w * Price_std / Mileage_std
b_unscaled = Price_std * b + Price_mean - w_unscaled * Mileage_mean

print(f"Unscaled w: {w_unscaled}, Unscaled b: {b_unscaled}")
# Test prediction
test_mileage = 3300
test_mileage_scaled = (test_mileage - Mileage_mean) / Mileage_std
test_price_scaled = w * test_mileage_scaled + b
test_price = test_price_scaled * Price_std + Price_mean
print(f"Predicted price for mileage {test_mileage}: {test_price}") 


'''

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Check for missing values
print(df.isnull().sum())


