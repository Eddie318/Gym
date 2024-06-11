import math, copy
import numpy as np
import pandas as pd


def cost_MSE(X, y, w, b):
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
    m = X.shape[0]
    for i in range(m):
        #f_wb = w * x[i] + b
        f_wb = np.dot(X[i], w) + b
        sq = (f_wb - y[i]) ** 2
        total_cost += sq
    total_cost = total_cost * (1 / (2 * m))

    return total_cost

def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): multivariate
      b (scalar)    : model parameter 
    Returns
      dj_dw (ndarray (n,)): The gradients of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    m,n = X.shape
    dJ_dw = np.zeros((n,))
    dJ_db = 0
    for i in range(m):
        #f_wb = w * x[i] + b
        err = (np.dot(X[i], w) + b)- y[i]
        for j in range(n):
          dJ_dw[j] = dJ_dw[j] + err * X[i,j]
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
#file_path = r'C:\Users\Administrator\Gym\ML\Linear_Regression\pokemon.csv'
file_path = r'/Users/mujiaxuan/Desktop/moujiaxuan/practice/Gym/ML/Linear_Regression/pokemon.csv'
df = pd.read_csv(file_path)

# Select features and target variable
# Replace 'CombatPower' with the actual name of the target column in your dataset
features = df[['HP', 'Attack', 'Defense', 'Speed', 'Sp. Atk', 'Sp. Def']]
target = df['Combat Power']  # Replace 'CombatPower' with the correct column name

# Convert to numpy arrays
X = features.to_numpy()
y = target.to_numpy()

test = np.array([45,49,65,45,49,65])
test = (test - np.mean(X, axis=0)) / np.std(X, axis=0)

# Normalize the features (optional but recommended for gradient descent)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Initialize parameters
w_init = np.zeros(X.shape[1])
b_init = 0
learning_rate = 1.0e-2
iterations = 100000

# Perform gradient descent
w, b = gradient_decent(learning_rate, iterations, X, y, w_init, b_init, cost_MSE, compute_gradient)

# Print the resulting parameters
print("Weights:", w)
print("Bias:", b)

#m,_ = X.shape
#for i in range(m):
#    print(f"prediction: {np.dot(X[i], w) + b:0.2f}, target value: {y[i]}")

print(test)
print(f"prediction: {np.dot(test, w) + b:0.2f}, target value: ..")
