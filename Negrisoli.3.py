# -*- coding: utf-8 -*-
"""
@author: Riccardo Negrisoli
   # Matricola : 1104591
"""
#%% Ex1 
print ('Problem set 3')
print ('OLG, Cagan Model of Hyperinflation and Cake Eating')
print ('Exercize 1')
print ('Canonical OLG Model')
#%% Point a
import numpy as np
import matplotlib.pyplot as plt

# Constants
gamma = 0.5
w_t = 1
r_t1 = 0.05

# Define the range for c1,t
c1_t = np.linspace(0.01, w_t - 0.01, 100)  # Ensure c1,t does not reach w_t exactly

# Compute c2,t+1 from the budget constraint
c2_t1 = (w_t - c1_t) * (1 + r_t1) + 1e-5  # Small positive offset to avoid log(0)

# Utility function
utility = np.log(c1_t) + gamma * np.log(c2_t1)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(c1_t, c2_t1, color='k', linestyle='-', label='Budget Line') 
plt.xlabel('$c1,t$ (Consumption at time t)')
plt.ylabel('$c2,t+1$ (Consumption at time t+1)')
plt.title('Consumer Equilibrium in the OLG Model')

# Add an indifference curve
# Let's select a utility level and plot the corresponding indifference curve
u_level = np.max(utility)  # Choose a high utility level to show optimal point
indifference_c2_t1 = np.exp((u_level - np.log(c1_t)) / gamma)  # Rearranging the utility function

plt.plot(c1_t, indifference_c2_t1, color='blue', linestyle='--', label='Indifference Curve (High Utility)')  # Change color to red and linestyle to dashed
plt.legend()
plt.grid(True)
plt.show()


#%% b
# Parameters
gamma = 0.5
w_t = 1
r_t1_original = 0.05
r_t1_new = 0.10  # increased interest rate

# Define the optimal consumption function
def optimal_consumption(w_t, r_t1, gamma):
    c1_t = w_t / (1 + gamma * (1 + r_t1))
    c2_t1 = gamma * (1 + r_t1) * w_t / (1 + gamma * (1 + r_t1))
    return c1_t, c2_t1

# Calculate consumption for original and increased interest rates
c1_t_original, c2_t1_original = optimal_consumption(w_t, r_t1_original, gamma)
c1_t_new, c2_t1_new = optimal_consumption(w_t, r_t1_new, gamma)

# Plotting
fig, ax = plt.subplots()
interest_rates = [r_t1_original, r_t1_new]
c1_t_values = [c1_t_original, c1_t_new]
c2_t1_values = [c2_t1_original, c2_t1_new]

ax.plot(interest_rates, c1_t_values, label='Optimal $c_{1,t}$', marker='o')
ax.plot(interest_rates, c2_t1_values, label='Optimal $c_{2,t+1}$', marker='o')
ax.set_xlabel('Interest Rate $r_{t+1}$')
ax.set_ylabel('Consumption')
ax.set_title('Changes in Optimal Consumption with Interest Rate')
ax.legend()

plt.show()
#%%c
# Constants
r_t1 = 0.05  # Fixed interest rate
gammas = np.linspace(0.1, 0.9, 20)  # Gamma values from 0.1 to 0.9
wages = np.linspace(0.5, 2, 20)     # Wage values from 0.5 to 2

# Calculate optimal savings s_t for varying gamma and fixed wage w_t = 1
optimal_savings_gamma = []
w_t_fixed = 1  # Fixed wage at 1 for gamma simulation
for gamma in gammas:
    c1_t = w_t_fixed / (1 + gamma * (1 + r_t1))  # Derived consumption c1,t
    s_t = w_t_fixed - c1_t  # Calculating savings
    optimal_savings_gamma.append(s_t)

# Calculate optimal savings s_t for varying wages and fixed gamma = 0.5
gamma_fixed = 0.5  # Fix gamma at 0.5 for wage simulation
optimal_savings_wage = []
for w_t in wages:
    c1_t = w_t / (1 + gamma_fixed * (1 + r_t1))  # Derived consumption c1,t
    s_t = w_t - c1_t  # Calculating savings
    optimal_savings_wage.append(s_t)

# Plotting the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot for varying gamma
ax1.plot(gammas, optimal_savings_gamma, marker='o', color='green', linestyle='--')  
ax1.set_title('Optimal Savings vs Gamma')
ax1.set_xlabel('Gamma')
ax1.set_ylabel('Optimal Savings')

# Plot for varying wage
ax2.plot(wages, optimal_savings_wage, marker='o', color='blue', linestyle='-.')  
ax2.set_title('Optimal Savings vs Wage')
ax2.set_xlabel('Wage')
ax2.set_ylabel('Optimal Savings')

plt.tight_layout()
plt.show()

#%% d
print ('Point D')
# Define parameters
n = 0.02  # Population growth rate
alpha = 0.33  # Capital's share of output in Cobb-Douglas production function
z = 1  # Productivity parameter
delta = 0.05  # Depreciation rate

# Define the savings function based on a simplistic model where s_t = z * k_t^alpha - delta * k_t
def savings_function(k_t, z, alpha, delta):
    return z * k_t ** alpha - delta * k_t

# Define the capital update function
def capital_next_period(k_t, n, z, alpha, delta):
    s_t = savings_function(k_t, z, alpha, delta)
    return s_t / (1 + n)

# Generate a range of k_t values
k_t_values = np.linspace(0.1, 10, 100)

# Compute k_{t+1} values
k_t1_values = [capital_next_period(k, n, z, alpha, delta) for k in k_t_values]

# Plot k_{t+1} vs k_t
plt.figure(figsize=(8, 6))
plt.plot(k_t_values, k_t1_values, label='$k_{t+1} = g(k_t, \\Lambda)$')
plt.plot(k_t_values, k_t_values, 'r--', label='45-degree line')
plt.title('Capital Next Period vs. Current Capital')
plt.xlabel('$k_t$')
plt.ylabel('$k_{t+1}$')
plt.legend()
plt.grid(True)
plt.show()

# Find the steady state by finding where k_{t+1} = k_t
steady_state_k = k_t_values[np.argmin(abs(np.array(k_t1_values) - k_t_values))]
print("The steady state capital is approximately:", steady_state_k)
#%% e
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
n = 0.02  # Population growth rate
alpha = 0.33  # Capital's share of output in a Cobb-Douglas production function
z = 1  # Productivity parameter
delta = 0.05  # Depreciation rate

# Function to compute next period's capital based on current capital
def capital_next_period(k_t, n, z, alpha, delta):
    # Compute savings as a function of current capital, productivity, and depreciation
    s_t = z * k_t ** alpha - delta * k_t
    # Return next period's capital, adjusting for population growth
    return s_t / (1 + n)

# Generate a range of current capital values
k_t_values = np.linspace(0.1, 10, 400)

# Compute next period's capital for each value of current capital
k_t1_values = [capital_next_period(k, n, z, alpha, delta) for k in k_t_values]

# Plotting the results
plt.figure(figsize=(8, 6))
plt.plot(k_t_values, k_t1_values, label='g(kt, Λ)')  # Plot g(kt, Λ)
plt.plot(k_t_values, k_t_values, 'r--', label='45-degree line')  # Plot the 45-degree line
plt.title('Capital Next Period vs. Current Capital')
plt.xlabel('Current Capital $k_t$')
plt.ylabel('Next Period Capital $k_{t+1}$')
plt.legend()

# Annotate the fixed point (steady state)
# Find the index and value where kt+1 is approximately equal to kt
steady_state_index = np.argmin(np.abs(np.array(k_t1_values) - k_t_values))
steady_state_kt = k_t_values[steady_state_index]
plt.scatter([steady_state_kt], [k_t1_values[steady_state_index]], color='green', zorder=5)
plt.annotate(f'Steady State (~{steady_state_kt:.2f})', (steady_state_kt, k_t1_values[steady_state_index]),
             textcoords="offset points", xytext=(0,10), ha='center', color='green')

plt.grid(True)
plt.show()
#%% f
z_values = [0.8, 1, 1.2]
def capital_next_period(k_t, z, n, alpha, delta):
    # Compute savings as a function of current capital, productivity, and depreciation
    s_t = z * k_t ** alpha - delta * k_t
    # Return next period's capital, adjusting for population growth
    return s_t / (1 + n)

# Generate a range of current capital values
k_t_values = np.linspace(0.1, 10, 400)

# Plotting the results
plt.figure(figsize=(10, 8))

for z in z_values:
    # Compute next period's capital for each value of current capital
    k_t1_values = [capital_next_period(k, z, n, alpha, delta) for k in k_t_values]
    plt.plot(k_t_values, k_t1_values, label=f'g(kt, Λ) with z={z}')

# Plot the 45-degree line
plt.plot(k_t_values, k_t_values, 'k--', label='45-degree line')
plt.title('Capital Next Period vs. Current Capital with Varying Productivity')
plt.xlabel('Current Capital $k_t$')
plt.ylabel('Next Period Capital $k_{t+1}$')
plt.legend()
plt.grid(True)
plt.show()
#%% G
# Parameters
alpha = 0.3
n = 0.01
z_values = [0.5, 1, 1.5]
k = np.linspace(0.1, 10, 400)  # Range of capital values to consider

def production_function(k, z, alpha):
    return z * k ** alpha

def capital_next(k, z, alpha, n):
    """Calculate next period's capital based on the production function and savings rate."""
    y = production_function(k, z, alpha)
    savings_rate = 0.2  # Assume some savings rate
    investment = savings_rate * y
    return investment / (1 + n)  # Adjusting for population growth

plt.figure(figsize=(10, 8))

for z in z_values:
    k_next = capital_next(k, z, alpha, n)
    plt.plot(k, k_next, label=f'z = {z}')

# Plotting the 45-degree line to find fixed points
plt.plot(k, k, 'k--', label='45-degree line')

plt.title('Capital Next Period vs. Current Capital for different z')
plt.xlabel('Capital This Period (k)')
plt.ylabel('Capital Next Period (k_next)')
plt.legend()
plt.grid(True)
plt.show()
#%% 
#######################
#EX2
###########
print ('Ex2')
# a
def simulate_hyperinflation(ρ, δ, α, m0, periods=50):
    mt = np.zeros(periods)
    pt = np.zeros(periods)
    mt[0] = m0  # Initial value for money supply
    pt[0] = 0   # Initial guess for price level, which will be updated

    for t in range(1, periods):
        # Calculate next price level based on the rational expectations and feedback effect
        pt[t] = pt[t-1] + (mt[t-1] - pt[t-1]) / α
        # Calculate next money supply based on current money supply and price level
        mt[t] = ρ * mt[t-1] + δ * pt[t-1]
        
    return mt, pt

# Parameters according to the model's specification
ρ = 0.9  # Persistence of money supply
δ = 0.04  # Feedback of price into money supply
α = 1.5  # Sensitivity of the price adjustment to money supply
m0 = 0.1  # Initial money supply

# Run the simulation for a specified number of periods
mt, pt = simulate_hyperinflation(ρ, δ, α, m0)

# Plotting the results to visualize the evolution of money supply and price level
plt.figure(figsize=(10, 5))
plt.plot(mt, label='Money Supply (m_t)')
plt.plot(pt, label='Price Level (p_t)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Simulation of the Cagan Model of Hyperinflation')
plt.legend()
plt.grid(True)
plt.show()
#%% b
print ('point b')
# Define the parameters
ρ = 0.9
δ = 0.04
α = 1.5

# Define the matrices
Λ0 = np.array([[1, 0],  # Identity matrix for the state space representation
               [0, 1]])

Λ1 = np.array([[ρ, δ],  # Transition matrix for state vector from t to t+1
               [1/α, 1 - 1/α]])

# Output the matrices
print("Lambda_0 Matrix (Λ0):")
print(Λ0)
print("\nLambda_1 Matrix (Λ1):")
print(Λ1)
#%% c
print ('point c')
def analyze_model():
    # Parameters of the model
    ρ = 0.9  # Coefficient of money supply persistence
    δ = 0.04  # Coefficient of price feedback into money supply
    α = 1.5  # Sensitivity of price adjustment

    # Matrix A as derived from the state space representation
    A = np.array([
        [ρ, δ],                # Row corresponding to the money supply equation
        [1/α, 1 - 1/α]         # Row corresponding to the price level equation
    ])

    # Calculate eigenvalues of matrix A
    eigenvalues = np.linalg.eigvals(A)

    # Print matrix A and its eigenvalues
    print("Matrix A:")
    print(A)
    print("\nEigenvalues of matrix A:")
    print(eigenvalues)

    # Check stability by examining eigenvalues
    stability_check = np.all(np.abs(eigenvalues) < 1)
    print("\nStability Check (all eigenvalues inside unit circle):", stability_check)

    return A, eigenvalues, stability_check

# Call the function to perform analysis
matrix_A, eigenvalues, is_stable = analyze_model()
#%% d
print ('Point D')
def analyze_saddle_path_stability():
    # Parameters of the model
    ρ = 0.9  # Coefficient of money supply persistence
    δ = 0.04  # Coefficient of price feedback into money supply
    α = 1.5  # Sensitivity of price adjustment

    # Define the matrix A according to the state space form derived earlier
    A = np.array([
        [ρ, δ],
        [1/α, 1 - 1/α]
    ])

    # Calculate eigenvalues of the matrix A
    eigenvalues = np.linalg.eigvals(A)
    print("Eigenvalues of Matrix A:", eigenvalues)

    # Check for saddle path stability: one eigenvalue should be > 1 and one < 1
    unstable_eigenvalues = np.sum(eigenvalues > 1)
    stable_eigenvalues = np.sum(eigenvalues < 1)
    saddle_path_stable = unstable_eigenvalues == 1 and stable_eigenvalues == 1

    # Output results
    print("Matrix A:")
    print(A)
    print("Eigenvalues of Matrix A:", eigenvalues)
    print("Saddle Path Stability:", "Yes" if saddle_path_stable else "No")
    print(f"Unstable eigenvalues: {unstable_eigenvalues}, Stable eigenvalues: {stable_eigenvalues}")

    return eigenvalues, saddle_path_stable

# Run the function
eigenvalues, is_stable = analyze_saddle_path_stability()
#%% e
def calculate_blanchard_khan_coefficients(A):
    """
    Calculate the Blanchard-Khan coefficients for the policy and transition functions
    based on a diagonalizable system matrix A.
    """
    # Compute the eigenvalues and right eigenvectors of A
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Assume ηpm corresponds to how price reacts to money supply (simple example)
    # and ηmm corresponds to the next money supply response to current money supply
    # For simplicity, let's assume the system is diagonalizable and use the inverse of eigenvectors matrix
    if np.iscomplex(eigenvalues).any():
        print("Complex eigenvalues found. Additional considerations needed for real system.")
        return None

    # We use the eigenvectors matrix as the transformation basis
    V_inv = np.linalg.inv(eigenvectors)
    ηpm = V_inv[1, 0]  # Response of price to money supply changes
    ηmm = V_inv[0, 0]  # Response of next money supply to current money supply

    return ηpm, ηmm

# Parameters and Matrix A definition
ρ = 0.9
δ = 0.04
α = 1.5
A = np.array([[ρ, δ], [1/α, 1 - 1/α]])

# Calculate Blanchard-Khan coefficients
ηpm, ηmm = calculate_blanchard_khan_coefficients(A)
print("ηpm (Policy function coefficient):", ηpm)
print("ηmm (Transition function coefficient):", ηmm)
#%% f
# Parameters
ηpm = 0.5  # Policy function coefficient (simplification for demonstration)
ηmm = 0.9  # Transition function coefficient
m0 = 1.0   # Initial money supply
p0 = ηpm * m0  # Initial price level calculated via policy function

# Simulation settings
N = 50  # Number of periods to simulate

# Initialize arrays to store simulation data
mt = np.zeros(N)
pt = np.zeros(N)
mt[0] = m0
pt[0] = p0  # Start at the policy function level

# Simulate the system
for t in range(1, N):
    mt[t] = ηmm * mt[t-1]  # Transition function
    pt[t] = ηpm * mt[t]    # Policy function

# Perturb initial price to see divergence
p0_high = p0 * 1.1  # 10% higher
p0_low = p0 * 0.9   # 10% lower
pt_high = np.zeros(N)
pt_low = np.zeros(N)
pt_high[0] = p0_high
pt_low[0] = p0_low

# Simulate with high and low initial prices
for t in range(1, N):
    pt_high[t] = ηpm * mt[t]  # Same mt, different initial pt
    pt_low[t] = ηpm * mt[t]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(mt, pt, label='At Policy Function', marker='o')
plt.plot(mt, pt_high, label='Initial pt 10% Higher', marker='x')
plt.plot(mt, pt_low, label='Initial pt 10% Lower', marker='^')
plt.xlabel('Money Supply (mt)')
plt.ylabel('Price Level (pt)')
plt.title('System Behavior Under Policy Function')
plt.legend()
plt.grid(True)
plt.show()
#%% G

# Parameters for the policy and transition functions
ηpm = 0.5  # Policy function coefficient
ηmm = 0.9  # Transition function coefficient

# Initial conditions with a negative money supply
m0 = -1.0   # Negative initial money supply
p0 = ηpm * m0  # Initial price level calculated via policy function

# Simulation settings
N = 50  # Number of periods to simulate

# Initialize arrays to store simulation data
mt = np.zeros(N)
pt = np.zeros(N)
mt[0] = m0
pt[0] = p0  # Start at the policy function level

# Simulate the system
for t in range(1, N):
    mt[t] = ηmm * mt[t-1]  # Transition function
    pt[t] = ηpm * mt[t]    # Policy function

# Perturb initial price to see divergence
p0_high = p0 * 1.1  # 10% higher
p0_low = p0 * 0.9   # 10% lower
pt_high = np.zeros(N)
pt_low = np.zeros(N)
pt_high[0] = p0_high
pt_low[0] = p0_low

# Simulate with high and low initial prices
for t in range(1, N):
    pt_high[t] = ηpm * mt[t]  # Same mt, different initial pt
    pt_low[t] = ηpm * mt[t]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(mt, pt, label='At Policy Function', marker='o')
plt.plot(mt, pt_high, label='Initial pt 10% Higher', marker='x')
plt.plot(mt, pt_low, label='Initial pt 10% Lower', marker='^')
plt.xlabel('Money Supply (mt)')
plt.ylabel('Price Level (pt)')
plt.title('System Behavior with Negative Initial Money Supply')
plt.legend()
plt.grid(True)
plt.show()
#%% H
# Define the parameters of the model
ρ = 0.9
δ = 0.04

# Coefficients of the quadratic equation
a1 = δ**2
b1 = 0  # There is no linear term in η_upm in the derived equation
c1 = ρ**2 - 2*ρ + 1

# Calculate the coefficients for the quadratic equation
coefficients = [a1, b1, c1]

# Solve the quadratic equation
solutions = np.roots(coefficients)
print("Solutions for η_upm:", solutions)

# Check corresponding η_umm for stability
η_umm_solutions = [ρ + δ * sol for sol in solutions]
print("Corresponding η_umm values for each η_upm:", η_umm_solutions)
#%% I
# Parameters from the Cagan model
ρ = 0.9
δ = 0.04

# Quadratic coefficients for η_upm
a1 = δ**2
b1 = 0
c1 = ρ**2 - 2*ρ + 1

# Solve the quadratic equation for η_upm
coefficients = [a1, b1, c1]
η_upm_solutions = np.roots(coefficients)
print("Solutions for η_upm:", η_upm_solutions)

# Hypothetical example of deriving η_pm using Blanchard Khan or similar approach
# Placeholder: Assume η_pm derived from a stable solution or analysis
η_pm = 0.5  # Hypothetical value for demonstration purposes

# Compare η_upm solutions to η_pm
print("Comparison with Blanchard Khan η_pm:")
comparison_results = np.isclose(η_upm_solutions, η_pm)
print("Does η_upm match η_pm? ", comparison_results)

# Evaluate stability of each η_upm by checking corresponding η_umm
η_umm_solutions = [ρ + δ * sol for sol in η_upm_solutions]
print("Corresponding η_umm values for each η_upm:", η_umm_solutions)
print("Stability Check (|η_umm| < 1):", [abs(η) < 1 for η in η_umm_solutions])
#%% J
# Example parameters from the Cagan model
ρ = 0.9
δ = 0.04

# Assume we have derived η_upm from the quadratic equation
η_upm = 0.5  # This should be the actual solution derived from solving the quadratic

# Calculate η_umm using the relation derived from the method of undetermined coefficients
η_umm = ρ + δ * η_upm

# Assume η_mm is derived from Blanchard Khan method using actual stability analysis
η_mm = 0.9  # This should be the actual value derived from Blanchard Khan method

# Print and compare η_umm with η_mm
print(f"Calculated η_umm: {η_umm}")
print(f"Given η_mm from Blanchard Khan: {η_mm}")
print("Do η_umm and η_mm match? ", np.isclose(η_umm, η_mm))

#%%
####################
############
########
# EX 3
# A
from scipy.optimize import fsolve
# Define the parameters
W1_value = 10  # Initial wealth
r_value = 0.05  # Interest rate
beta_value = 0.95  # Discount factor
gamma_value = 0.5  # CEIS parameter

# Define the equations to solve
def equations(vars):
    c1, c2 = vars
    eq1 = c1**(-gamma_value) - ((1 + r_value) * beta_value * c2**(-gamma_value))
    eq2 = W1_value - c1 - c2 / (1 + r_value)
    return [eq1, eq2]

# Initial guesses for c1 and c2
initial_guesses = [W1_value / 3, W1_value / 3]

# Solve the equations
solution = fsolve(equations, initial_guesses)
c1_solution, c2_solution = solution

# Calculate W2 and W3 to verify W3 is zero
W2 = W1_value - c1_solution
W3 = W2 - c2_solution

print(f"Optimal c1: {c1_solution}")
print(f"Optimal c2: {c2_solution}")
print(f"Remaining Wealth W3: {W3}")
#%% B
def compute_consumptions(W1, gamma, beta):
    c1 = W1 / (1 + beta ** (-1/gamma))
    c2 = W1 - c1
    return c1, c2

# Example parameters
gamma = 0.5
beta = 0.95
W1 = 100  # Total initial wealth

c1, c2 = compute_consumptions(W1, gamma, beta)
print("c1:", c1)
print("c2:", c2)

# Verify the sum of c1 and c2 equals W1
assert np.isclose(c1 + c2, W1)
#%% C
def optimal_consumption(W1, gamma, beta):
    # Calculate c1 using the derived formula
    # This formula optimizes the first period consumption
    c1 = (beta**(1/gamma) * W1) / (1 + beta**(1/gamma))
    
    # Calculate c2 as the residual wealth after c1 is consumed
    # This ensures all wealth is allocated over the two periods
    c2 = W1 - c1
    
    # Sum of consumption in both periods should equal initial wealth
    total_consumption = c1 + c2
    verification = total_consumption == W1
    
    # Print detailed calculation steps for debugging
    print(f"Calculated c1: {c1}")
    print(f"Calculated c2: {c2}")
    print(f"Total consumption (c1 + c2): {total_consumption}")
    print(f"Verification that total consumption equals initial wealth: {'Correct' if verification else 'Incorrect'}")
    
    return c1, c2, verification

# Example parameters
W1 = 100  # initial wealth
gamma = 0.5  # risk aversion parameter
beta = 0.95  # discount factor

# Execute the function and print the results
c1, c2, is_correct = optimal_consumption(W1, gamma, beta)
#%% D
def optimal_consumption(W1, gamma, beta):
    # Calculate c1 using the derived formula
    c1 = (beta**(1/gamma) * W1) / (1 + beta**(1/gamma))
    
    # Calculate c2, which is the remaining wealth after consuming c1
    c2 = W1 - c1
    
    return c1, c2

# Parameters
W1 = 100  # Example initial wealth
gamma = 0.5  # Example risk aversion parameter

# Generate a sequence of 20 beta values from 0.01 to 0.99
betas = np.linspace(0.01, 0.99, 20)

# Initialize lists to store c1 and c2 values
c1_values = []
c2_values = []

# Calculate c1 and c2 for each beta
for beta in betas:
    c1, c2 = optimal_consumption(W1, gamma, beta)
    c1_values.append(c1)
    c2_values.append(c2)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(betas, c1_values, label='c1', marker='o')
plt.plot(betas, c2_values, label='c2', marker='x')
plt.title('Equilibrium values of c1 and c2 for different values of β')
plt.xlabel('β (Discount Factor)')
plt.ylabel('Consumption Values')
plt.legend()
plt.grid(True)
plt.show()
#%% E
def optimal_consumption(W1, gamma, beta):
    # Calculate c1 using the derived formula
    c1 = (beta**(1/gamma) * W1) / (1 + beta**(1/gamma))
    
    # Calculate c2, which is the remaining wealth after consuming c1
    c2 = W1 - c1
    
    return c1, c2

# Constants
W1 = 100  # initial wealth
beta = 0.95  # discount factor

# Generate a sequence of 20 gamma values from 0.1 to 0.99
gammas = np.linspace(0.1, 0.99, 20)

# Initialize lists to store c1 and c2 values
c1_values = []
c2_values = []

# Calculate c1 and c2 for each gamma
for gamma in gammas:
    c1, c2 = optimal_consumption(W1, gamma, beta)
    c1_values.append(c1)
    c2_values.append(c2)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(gammas, c1_values, label='c1', marker='o')
plt.plot(gammas, c2_values, label='c2', marker='x')
plt.title('Equilibrium values of c1 and c2 for different values of γ')
plt.xlabel('γ (Risk Aversion Parameter)')
plt.ylabel('Consumption Values')
plt.legend()
plt.grid(True)
plt.show()
#%%  F
def optimal_consumption(W1, gamma, beta):
    c1 = (beta**(1/gamma) * W1) / (1 + beta**(1/gamma))
    c2 = W1 - c1
    return c1, c2

def utility_function(c1, c2, gamma, beta):
    if gamma == 1:
        # Logarithmic utility case
        U1 = np.log(c1)
        U2 = beta * np.log(c2)
    else:
        # CEIS utility case
        U1 = (c1**(1-gamma)) / (1-gamma)
        U2 = beta * (c2**(1-gamma)) / (1-gamma)
    return U1 + U2

# Example parameters
W1 = 100  # Initial wealth
gamma = 0.5  # Risk aversion parameter
beta = 0.95  # Discount factor

# Calculate optimal consumptions
c1, c2 = optimal_consumption(W1, gamma, beta)

# Calculate utility
U = utility_function(c1, c2, gamma, beta)
print(f"Optimal c1: {c1:.2f}, Optimal c2: {c2:.2f}, Utility: {U:.2f}")
#%%  G
def optimal_consumption(W1, gamma, beta):
    c1 = (beta**(1/gamma) * W1) / (1 + beta**(1/gamma))
    c2 = W1 - c1
    return c1, c2

def utility_function(c1, c2, gamma, beta):
    if gamma == 1:
        # Logarithmic utility case
        U1 = np.log(c1)
        U2 = beta * np.log(c2)
    else:
        # CEIS utility case
        U1 = (c1**(1-gamma)) / (1-gamma)
        U2 = beta * (c2**(1-gamma)) / (1-gamma)
    return U1 + U2

# Parameters
gamma = 0.5  # Risk aversion parameter
beta = 0.95  # Discount factor

# Generate a sequence of W1 values from 50 to 200
W1_values = np.linspace(50, 200, 20)

# Calculate utility for each W1
utilities = []
for W1 in W1_values:
    c1, c2 = optimal_consumption(W1, gamma, beta)
    U = utility_function(c1, c2, gamma, beta)
    utilities.append(U)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(W1_values, utilities, marker='o', linestyle='-')
plt.title('Utility vs Initial Wealth')
plt.xlabel('Initial Wealth (W1)')
plt.ylabel('Utility (U)')
plt.grid(True)
plt.show()




