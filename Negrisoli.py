
"""
@author: Riccardo Negrisoli
   # Matricola : 1104591
"""
#%%
print ('PROBLEM SET 4')
#%%
print ('EX 1')
print ('point G')
import numpy as np

# Constants
alpha = 0.33
beta = 0.98
k_0 = 1  # Initial capital
num_periods = 100

# Initialize arrays for capital and consumption
k = np.zeros(num_periods + 1)
c = np.zeros(num_periods + 1)
k[0] = k_0
# Adjust initial consumption to a more reasonable fraction of initial capital output
c[0] = min(k_0**alpha * 0.75, k_0**alpha)

# Simulate the system
for t in range(num_periods):
    k[t+1] = k[t]**alpha - c[t]
    # Ensure that capital does not become negative or zero
    if k[t+1] <= 0:
        k[t+1] = 0.1  # Set a lower bound to keep capital positive
    c[t+1] = beta * alpha * k[t+1]**(alpha - 1) * c[t]
    # Prevent consumption from becoming non-physical
    if c[t+1] > k[t+1]**alpha or c[t+1] <= 0:
        c[t+1] = k[t+1]**alpha * 0.75

# Calculate eigenvalues of the state transition matrix A
A = np.array([[alpha, -1], [alpha**2 - alpha, 2 - alpha]])
eigenvalues = np.linalg.eigvals(A)

# Display results
print("Capital over time:", k)
print("Consumption over time:", c)
print("Eigenvalues of the matrix A:", eigenvalues)
#%%
print ('EX 2')
print ('point B')

from scipy.optimize import fsolve

# Parameters
alpha = 0.33
beta = 0.98
k_star = (1 / (beta * alpha)) ** (1 / (alpha - 1))
c_star = k_star**alpha - k_star

# Function to find the roots
def equations(eta_kk):
    return eta_kk - alpha + (c_star / k_star) * (alpha - 1) * eta_kk / (1 - eta_kk)

# Initial guess
eta_kk_guess = 0.5

# Solve for eta_kk
eta_kk = fsolve(equations, eta_kk_guess)[0]

# Calculate eta_ck
eta_ck = (alpha - 1) * eta_kk / (1 - eta_kk)

print(f"eta_kk: {eta_kk}")
print(f"eta_ck: {eta_ck}")


#%% 
print ('EX 3')
# The library Linearsolve gave me problems hence 
#I'm using alternative libraries

from numpy.linalg import inv

# Finding steady state values for capital (k) and consumption (c)
def steady_state(values):
    k, c = values
    eq1 = k**alpha - k - c  # Resource constraint
    eq2 = c - beta * (alpha * k**(alpha-1)) * (k**alpha - k)  # Euler equation
    return [eq1, eq2]

initial_guess = [1, 0.3]
k_steady, c_steady = fsolve(steady_state, initial_guess)

# Define the log-linearization around the steady state
def log_linear_system(k, c, k_steady, c_steady):
    # Linearize around the steady state
    y = k_steady**alpha
    log_k = np.log(k/k_steady)
    log_c = np.log(c/c_steady)
    log_y = np.log(y/c_steady)
    
    # Matrix form A*X_t = B*X_t+1
    # X_t = [log_k, log_c], X_t+1 = [log_k_plus, log_c_plus]
    A = np.array([[1, 0],
                  [-beta * alpha * k_steady**(alpha-1), 1]])
    B = np.array([[alpha, -1],
                  [0, beta * alpha * k_steady**(alpha - 1) * (alpha - 1) * k_steady**(alpha - 2) * y]])
    
    return A, B

A, B = log_linear_system(k_steady, c_steady, k_steady, c_steady)

# Solve for the policy function using matrix operations
P = inv(A) @ B

print("Policy and Transition Function Matrix P:")
print(P)
#%%
print ('EX 4')
print ('point E')

from scipy.linalg import schur

# Parameters
alpha = 0.3
beta = 0.98
delta = 0.2
phi = 0.5

# Define the steady state equations
def steady_state(values):
    k, c, n = values
    eq1 = c + k - (1 - delta) * k - k**alpha * n**(1 - alpha)  # Resource constraint
    eq2 = 1 / c - beta * (1 / c) * (alpha * k**(alpha - 1) * n**(1 - alpha) + 1 - delta)  # Euler equation
    eq3 = phi * c / (1 - n) - (1 - alpha) * k**alpha * n**(-alpha)  # Labor-leisure trade-off
    return [eq1, eq2, eq3]

# Initial guess for k, c, n
initial_guess = [1, 0.5, 0.3]
k_steady, c_steady, n_steady = fsolve(steady_state, initial_guess)

# Linearize around the steady state
def log_linear_system(k_steady, c_steady, n_steady):
    # Steady state values
    y_steady = k_steady**alpha * n_steady**(1 - alpha)
    w_steady = (1 - alpha) * y_steady / n_steady
    r_steady = alpha * y_steady / k_steady

    # Define the Jacobian matrices for the system
    A = np.array([[1, -1 / c_steady, 0],
                  [-1, beta * r_steady / c_steady, 0],
                  [0, phi / (1 - n_steady), -phi * c_steady / (1 - n_steady)**2]])

    B = np.array([[alpha * k_steady**(alpha - 1) * n_steady**(1 - alpha), 0, (1 - alpha) * k_steady**alpha * n_steady**(-alpha)],
                  [0, 1 / c_steady, 0],
                  [0, 0, 1]])

    return A, B

A, B = log_linear_system(k_steady, c_steady, n_steady)

# Schur decomposition to find the unique stable steady state
T, Z = schur(A - B @ np.eye(len(A)))
eigenvalues = np.diag(T)

# Print results
print("Steady state values:")
print(f"k_steady: {k_steady}")
print(f"c_steady: {c_steady}")
print(f"n_steady: {n_steady}")
print("\nEigenvalues of the system (should be less than one in magnitude for stability):")
print(eigenvalues)
#%%
print ('Point F')

from scipy.linalg import solve_discrete_lyapunov
# Define the steady state equations
def steady_state(values):
    k, c, n = values
    eq1 = c + k - (1 - delta) * k - k**alpha * n**(1 - alpha)  # Resource constraint
    eq2 = 1 / c - beta * (1 / c) * (alpha * k**(alpha - 1) * n**(1 - alpha) + 1 - delta)  # Euler equation
    eq3 = phi * c / (1 - n) - (1 - alpha) * k**alpha * n**(-alpha)  # Labor-leisure trade-off
    return [eq1, eq2, eq3]

# Initial guess for k, c, n
initial_guess = [1, 0.5, 0.3]
k_steady, c_steady, n_steady = fsolve(steady_state, initial_guess)

# Linearize around the steady state
def log_linear_system(k_steady, c_steady, n_steady):
    # Steady state values
    y_steady = k_steady**alpha * n_steady**(1 - alpha)
    w_steady = (1 - alpha) * y_steady / n_steady
    r_steady = alpha * y_steady / k_steady

    # Define the Jacobian matrices for the system
    A = np.array([[1, -1 / c_steady, 0],
                  [-1, beta * r_steady / c_steady, 0],
                  [0, phi / (1 - n_steady), -phi * c_steady / (1 - n_steady)**2]])

    B = np.array([[alpha * k_steady**(alpha - 1) * n_steady**(1 - alpha), 0, (1 - alpha) * k_steady**alpha * n_steady**(-alpha)],
                  [0, 1 / c_steady, 0],
                  [0, 0, 1]])

    return A, B

A, B = log_linear_system(k_steady, c_steady, n_steady)

# Solve for the policy function using matrix operations
P = np.linalg.solve(A, B)

print("Steady state values:")
print(f"k_steady: {k_steady}")
print(f"c_steady: {c_steady}")
print(f"n_steady: {n_steady}")
print("\nPolicy and Transition Function Matrix P:")
print(P)

# Transition function in the form of X_{t+1} = T * X_t
T = np.linalg.inv(A) @ B

print("\nTransition Function Matrix T:")
print(T)

#%%
print ('EX 5')

z = 1  # Productivity, constant at 1
n = 1  # Constant labor supply

# Steady state calculation
def steady_state(vars):
    k, c = vars
    if k <= 0:  # Prevent taking a logarithm or power of a non-positive number
        return (1e10, 1e10)  # Return a large error if k is non-positive
    if c <= 0:  # Prevent zero or negative consumption
        return (1e10, 1e10)  # Return a large error if c is non-positive

    # Steady state equations derived from the model
    ss1 = c - (z * k**alpha * n**(1 - alpha) + (1 - delta) * k - k)
    ss2 = beta * c - c  # Simplistic steady state consumption condition
    return (ss1, ss2)

# Initial guesses for k and c
initial_guesses = [1, 1]  # More reasonable initial guesses
k_ss, c_ss = fsolve(steady_state, initial_guesses)

# Output steady state values
print("Steady state capital:", k_ss)
print("Steady state consumption:", c_ss)
print (" Clearly this steady state is unlikely correct")


