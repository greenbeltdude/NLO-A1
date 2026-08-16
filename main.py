import pandas as pd
import numpy as np
import matplotlib as plt


'''
This set of functions takes a csv file and has methods that convert it to usable data,
and maximizes the log likelihood to predict the relation between the 
first column values and the data values of the other column variables.
'''
def load_data(path, seed):
    '''
    Take a csv file with delimiter ; and make a pandas dataframe.
    Then runs Algorithm 1.
    
    Params:
    path - name of the file/path to the file
    seed - natural number that determine the random number
    
    Returns matrix X and vector y of observations.
    '''
    # Gets dataframe and convert to matrix without headers
    df = pd.read_csv(path, sep=';')
    data_matrix = np.array(df)
    
    # Makes Zero Vector of Same number of rows as data_matrix
    X = np.array([])
    
    # Add numeric columns of data_matrix
    for j in range(1,data_matrix.shape[1]):
        if isinstance(data_matrix[0,j],float):
            if len(X) == 0:
                X = data_matrix[:,j]
            else:
                X = np.vstack([X, data_matrix[:,j]])
    
        else:
            # Parse the Unique Strings and Alphabetically Order Them
            unique_strings = []
            for i in range(0, data_matrix.shape[0]):
                if not(data_matrix[i,j] in unique_strings):
                    unique_strings.append(data_matrix[i,j])
            unique_strings.sort()
            
            # Construct Columns Corresponding to Each String
            for str_i in range(1,len(unique_strings)):
                str = unique_strings[str_i]
                str_list = []
                for i in range(0, data_matrix.shape[0]):
                    if str == data_matrix[i,j]:
                        str_list.append(1)
                    else:
                        str_list.append(0)
                    
                str_col = np.transpose(np.array(str_list))
            
                # Concatenate the Columns to X or Initialize X
                if len(X) == 0:
                    X = str_col
                else:
                    X = np.vstack([X, str_col])                    
                    
    # Correcting for Numpy Behavior
    X = np.transpose(X)
    
    # Constructing y vector
    y_list = []

    for type in data_matrix[:,0]:
        if type == 'poisonous':
            y_list.append(1)
        else:
            y_list.append(0)
    
            
    y = np.transpose(np.array(y_list))
    
    # Permuting rows of y and X
    rng = np.random.default_rng(seed)
    permute = rng.permutation(len(y))
    X = X[permute]
    y = y[permute]
    
    return X, y


def func_llh(theta, X, y):
    '''
    Gets the Log Likelihood given parameter theta.
    
    Params:
    theta - parameter vector
    X - observations of non-class values
    y - class observations
    
    Returns the Log Likelihood value.
    '''
    w_0 = theta[0]
    w = theta[1: len(theta)]
    
    # For each observation get qi then update li
    f = 0
    for i in range(0, X.shape[0]):
        y_i = y[i]
        x_i = X[i,:]
        
        log_li = y_i * (w_0 + w.T @ x_i) - np.log(1 + np.exp(w_0 + w.T @ x_i))
        f += log_li
    
    return f


def mini_batch_grad_llh(theta, X, y, batch_bounds):
    '''
    Computes the gradient of the log likelihood for a mini-batch at a given theta.
    
    Params:
    theta - parameter vector
    X - non-class observations
    y - class observations
    batch_bounds - the bounds for which the batch is decided
    
    Returns the mini-batch gradient.
    '''
    
    w_0 = theta[0]
    w = theta[1: len(theta)]
    
    # Define List of Chosen Observations
    B = list(range(batch_bounds[0],batch_bounds[1]))
    card_B = len(B)
    
    min_b_grad = 0
    for i in B:
        x_i = X[i,:]
        
        # Make the Extended Row with 1 as the first element
        ext_x = np.hstack([np.array([1]), x_i])
        scalar_term = y[i] - (1/(1 + np.exp(-w_0 - w.T @ x_i)))
        grad_i =  X.shape[0] * scalar_term * ext_x
        min_b_grad += (1/card_B) * grad_i
    
    return min_b_grad      
        

def next_batch_bounds(batch_bounds, num_obs):
    '''
    Updates the Batch Bounds According to Given Piecewise.
    
    Params:
    batch_bounds - bounds with which the batch is decided.
    num_obs - the number of observations
    
    Returns the new Batch Bounds.
    '''
    b_start = batch_bounds[0]
    b_end = batch_bounds[1]
    
    if 2 * b_end - b_start < num_obs:
        return b_end, 2 * b_end - b_start
    
    else:
        return b_end - num_obs, 2 * b_end - b_start - num_obs
        
        
def adam(theta0, func, mini_batch_grad, num_obs, batch_size, \
    alpha, beta1, beta2, epsilon, delta):
    '''
    Runs Adaptive Moment Estimation Minimization
    using an arbitrary mini-batch gradient and function.
    
    Params:
    theta0 - starting parameter
    func - the function to minimize
    mini_batch_grad - the gradient of the function to minimize
    num_obs - number of observations
    batch_size - the value of the starting b_end and the size of the batches
    alpha - constant used when moving points
    beta1 - first decay constant
    beta2 - second decay constant
    epsilon - constant
    delta - termination condition for change in function value
    
    Returns a list of thetas used in the iterations.
    '''
    
    batch_bounds = [0, min(num_obs, batch_size)]
    theta_k = theta0
    m_k = np.transpose(np.zeros(len(theta0)))
    v_k = np.transpose(np.zeros(len(theta0)))
    
    theta_list = [theta0]
    previous_function_value = np.inf
    k = 1
    while True:
        # Compute Gradient g_k and Update m_k and v_k
        g_k = mini_batch_grad(theta_k, batch_bounds)
        m_k = beta1 * m_k + (1 - beta1) * g_k
        v_k = beta2 * v_k + (1 - beta2) * (g_k * g_k) 
        
        # Update v_hat_k and m_hat_k
        m_hat_k = m_k / (1 - (beta1**k))
        v_hat_k = v_k / (1 - (beta2**k))
        v_hat_k = v_hat_k.astype(np.float64)
        
        # Update theta_k
        theta_k = theta_k - alpha * m_hat_k / (v_hat_k ** 0.5 + epsilon)
        theta_list.append(theta_k)
        
        # Update Bounds
        prev_batch_bounds = batch_bounds
        batch_bounds = next_batch_bounds(batch_bounds, num_obs)
        
        # Checks Termination Condition and Returns Theta List if True
        if batch_bounds[0] < prev_batch_bounds[1]:
            function_value = func(theta_k)
            
            if previous_function_value - function_value < delta: # Assuming Minimization
                return theta_list
            
            previous_function_value = function_value
        
        k += 1


def logit(X, y):
    '''
    Runs Adam to minimize the negative log likelihood function and returns optimal theta.
    
    Params:
    X - observations of non-class variables
    y - class observations
    
    Returns the final theta of the adam returned theta list
    '''
    def mini_batch_grad(theta, batch_bounds):
        return -mini_batch_grad_llh(theta, X, y, batch_bounds) # Accounting for Maximization
    
    def func(theta):
        return -func_llh(theta, X, y) # Accounting for Maximization
    
    batch_size = 128
    num_obs = X.shape[0]
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 10**-8
    delta = 10 **-8
    theta0 = np.transpose(np.zeros(X.shape[1] + 1))
    
    
    return adam(theta0, func, mini_batch_grad, num_obs, batch_size, \
    alpha, beta1, beta2, epsilon, delta)[-1]
                
            
def logistic_prediction(X, theta):
    '''
    Uses the estimated maximized theta given by Adam as well as the logistic function
    to predict the whether they are poisonous or not.
    
    Params:
    X - observations of non-class variables
    theta - the given parameter (typically the one returned by logit)
    
    Returns the predicted classes of the observations.
    '''
    w_0 = theta[0]
    w = theta[1: len(theta)]
    
    y_hat_list = []
    for i in range(0, X.shape[0]):
        x_i = X[i,:]
        if 1/(1 + np.exp(-w_0 - w.T @ x_i)) >= 0.5:
            y_hat_list.append(1)
        else:
            y_hat_list.append(0)
    
    return np.transpose(np.array(y_hat_list))


def accuracy(y, yhat):
    '''
    Calculates the accuracy of the predictions given the vector of actual values.
    
    Params:
    y - actual observations of the class variable
    yhat - predicted class variable values
    
    Returns the accuracy of the predications.
    '''
    total_predictions = len(y)
    num_correct = 0
    
    for i in range(0, len(y)):
        if y[i] == yhat[i]:
            num_correct += 1
            
    return np.float64(num_correct / total_predictions)
            


