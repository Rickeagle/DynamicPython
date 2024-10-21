# -*- coding: utf-8 -*-
"""
@author: Riccardo Negrisoli
   # Matricola : 1104591
"""

print ('Problem set 2')
print ('  Non linear Equations and Function Approximation')
### Ex 1
print ('Exercize 1')
print ('Two period Fisher Model')
#%% Point C
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# parameters
α=.3  #cobb douglas
g=.03 #productivity growth
A=.5
ρ=.05
θ=.5
k1=0.77
l1=0.99
l2=l1
def c2(k2):
    return k2+A*(1+g)*(k2**α)*(l2**(1-α))

def c1(k2):
    return 1+A*(k1**α)*(l1**(1-α))-k2

def r2(k2):
    return α*A*(1+g)*(k2**(α-1))*(l2**(1-α))

def mrs(k2):
    return ((c1(k2)/c2(k2))**(θ))/(1+ρ)

def mrt(k2):
    return 1/(1+r2(k2))

def ge(k2):
    return mrs(k2)-mrt(k2)


def eqr2(k2):
    return α *A*(1+g)*k2**(α-1)*(l2**(1-α))

def eqw2(k2):
    return (1-α)*A*(1+g)*(k2**(α))*(l2**(-α))

def bigU(k2):
    return (c1(k2)**(1-θ))/(1-θ)+ ((c2(k2)**(1-θ))/(1-θ))/(1+ρ)

def indc(cons1,U):
    return (U*(1-θ)*(1+ρ)-(1+ρ)*cons1**(1-θ))**(1/(1-θ))

def trc(cons1):
    return 1+A-cons1+A*(1+g)*(1+A-cons1)**(α)
def eqr2(k2):
    return α*A(1+g)*k2*(α-1)

def eqw2(k2):
    return (1-α)*A*(1+g)*k2*(α)

def bigU(k2):
    return (c1(k2)*(1-θ))/(1-θ)+ ((c2(k2)*(1-θ))/(1-θ))/(1+ρ)

def indc(cons1,U):    
    return (U*(1-θ)(1+ρ)-(1+ρ)*cons1(1-θ))*(1/(1-θ))      

def trc(cons1):
    return 1+A-cons1+A*(1+g)(1+A-cons1)*(α)
grid_max=0.99 
grid_size=100 
grid_min=0.01
grid= np.linspace(grid_min, grid_max, grid_size)
zer=np.zeros(len(grid))
gridinv= np.linspace(grid_max, grid_min, grid_size)

geq4=fsolve(ge,.2) # Find General Equilibrium with Fsolve

print("General Equilibrium with Fsolve", geq4)


geq5=fsolve(ge,grid_max-.2) # Find General Equilibrium with Fsolve

print("General Equilibrium with Fsolve", geq5)

#%% Point D
import os
os.getcwd()
os.getcwd()
##
# Function definition  #### Necessary to have 2 inputs variables
def c1(k2, l0):
    return 1 + A * (k1 ** α) * (l1 ** (1 - α)) - k2

def c2(k2, l0):
    return k2 + A * (1 + g) * (k2 ** α) * (l0 ** (1 - α))
def out1(k1,l0):
    return A*(k1**α) * (l0**(1-α))
def out2(k2,l0):
    return A * (1 + g) * k2 ** (α) * (l0 ** (1 - α))

def eqr2(k2, l0):
    return α * A * (1 + g) * k2 ** (α-1) * (l2 ** (1 - α))

def eqw2(k2, l0):
    return (1 - α) * A * (1 + g) * (k2 ** α) * (l2 ** (-α))

def ge(k2, l0):
    return c2(k2, l0) - 1 - A * (k1 ** α) * (l1 ** (1 - α))



# Grid parameters
grid_size = 10
l0_values = np.linspace(0.01, 0.99, grid_size)

# Lists to store equilibrium values
k2_values = []
y1_values = []
y2_values = []
w1_values = []
w2_values = []

# Iterating over different values of l0
for l0 in l0_values:
    # Find equilibrium k2 using fsolve
    geq = fsolve(ge, 0.2, args=(l0,))  # Initial guess for fsolve and passing l0 as argument
    k2_values.append(geq[0])  # Store equilibrium k2
    
    # Compute other variables based on equilibrium k2 and l0
    y1_values.append(out1(geq, l0))
    y2_values.append(out2(geq, l0))
    w1_values.append(eqr2(geq, l0))
    w2_values.append(eqw2(geq, l0))

# Plotting
plt.plot(l0_values, k2_values, label='Equilibrium k2',color='green')
plt.plot(l0_values, y1_values, label='Output y1',color='blue')
plt.plot(l0_values, y2_values, label='Output y2',color='yellow')
plt.plot(l0_values, w1_values, label='Wage w1',color='purple')
plt.plot(l0_values, w2_values, label='Wage w2',color='red')
plt.xlabel('Initial labor input l\u2080')
plt.ylabel('Values')
plt.title('Comparative Statics: Effects of changes in l\u2080 on k2, y1, y2, w1, w2')
plt.legend(bbox_to_anchor=(0.05, 0.55), loc='center left')
plt.grid(True)
plt.show()
#%% POINT E
# Iterate changing K0 this time
# Grid parameters
grid_size = 10
k0_values = np.linspace(0.01, 0.99, grid_size)

# Lists to store equilibrium values
k2_values = []
y1_values = []
y2_values = []
w1_values = []
w2_values = []

# Iterating over different values of K0
for k0 in k0_values:
    # Find equilibrium k2 using fsolve
    geq = fsolve(ge, 0.2, args=(k0,))  # Initial guess for fsolve and passing l0 as argument
    k2_values.append(geq[0])  # Store equilibrium k2
    
    # Compute other variables based on equilibrium k2 and l0
    y1_values.append(out1(geq, l0))
    y2_values.append(out2(geq, l0))
    w1_values.append(eqr2(geq, l0))
    w2_values.append(eqw2(geq, l0))

# Plotting
plt.plot(l0_values, k2_values, label='Equilibrium k2',color='green')
plt.plot(l0_values, y1_values, label='Output y1',color='blue')
plt.plot(l0_values, y2_values, label='Output y2',color='yellow')
plt.plot(l0_values, w1_values, label='Wage w1',color='purple')
plt.plot(l0_values, w2_values, label='Wage w2',color='red')
plt.xlabel('Initial capital input K\u2080')
plt.ylabel('Values')
plt.title('Comparative Statics: Effects of changes in k0 on k2, y1, y2, w1, w2')
plt.legend(bbox_to_anchor=(0.05, 0.58), loc='center left')
plt.grid(True)
plt.show()


###################################
#######################


################
#%% 
print ('Exercize 2')
print ('Ammortization of loan as a difference equation')

## Point B
import numpy as np
import matplotlib.pyplot as plt

# Function to compute debt for T periods
def compute_debt(T, r, Z, D0):
    debt = np.zeros(T+1)
    debt[0] = D0
    for t in range(T):
        debt[t+1] = debt[t] + r * debt[t] - Z
    return debt

# Assigning values
r = 0.1  # Interest rate
Z = 5    # Repayment per period
D0 = 100 # Initial debt
T = 20   # Number of periods

# Compute debt for T periods
debt = compute_debt(T, r, Z, D0)

# Plotting
time_periods = np.arange(T+1)
plt.plot(time_periods, debt, marker='o', linestyle='-')
plt.xlabel('Time')
plt.ylabel('Debt')
plt.title('Debt Dynamics Over Time')
plt.grid(True)
plt.show()
#%%  C

# Function to compute debt for T periods
def compute_debt(T, r, Z, D0):
    debt = np.zeros(T+1)
    debt[0] = D0
    for t in range(T):
        debt[t+1] = debt[t] + r * debt[t] - Z
    return debt

# Assigning values
Z_values = [2, 4, 6]  # Different repayment per period
r_values = [0.05, 0.1, 0.15]  # Different interest rates
D0 = 100  # Initial debt
T = 20    # Number of periods

# Plotting for different combinations of r and Z
plt.figure(figsize=(10, 6))
for r, Z in zip(r_values, Z_values):
    debt = compute_debt(T, r, Z, D0)
    plt.plot(np.arange(T+1), debt, marker='o', linestyle='--', label=f'r={r}, Z={Z}')

plt.xlabel('Time')
plt.ylabel('Debt')
plt.title('Debt Dynamics Over Time for Different r and Z')
plt.legend()
plt.grid(True)
plt.show()
"""
We conducted simulations to analyze the evolution of debt under various 
combinations of interest rates (denoted as "r") and repayment amounts per 
period (denoted as "Z").
As the interest rate increases, the debt tends to accumulate at a faster pace 
over time. This phenomenon occurs because higher interest rates result in larger
 interest payments accruing on the outstanding debt. Consequently, 
 the debt grows more rapidly, amplifying its burden on the debtor.

Increasing the repayment amount per period facilitates a more rapid reduction 
in the outstanding debt over time. A higher repayment means that a greater 
portion of the debtor's resources is allocated towards debt repayment.
 Consequently, the debt balance decreases more swiftly, alleviating the
 financial strain associated with debt.
 
The interplay between interest rates and repayment amounts shapes the overall
 trajectory of debt dynamics. For instance, a scenario characterized by a high
 interest rate coupled with a low repayment may lead to a scenario where debt 
 accumulates steadily. However, a high repayment can offset the adverse impact
 of a high interest rate by accelerating debt reduction.
Ensuring the stability and sustainability of debt dynamics is crucial. 
Persistent growth in debt or its approach towards unsustainable levels may 
precipitate financial instability or insolvency. It's imperative to monitor debt
dynamics closely to preemptively address any emerging risks to financial stability.
These simulations offer valuable insights for policymakers regarding the
 ramifications of varying interest rates and repayment policies on debt 
 sustainability. Policy adjustments such as modifications to interest rates or 
 repayment schedules may be warranted to achieve desired outcomes in debt
 management and mitigate associated risks.
"""
#%% E
# Function to compute Z* given r, T, and D0
def compute_repayment(r, T, D0):
    numerator = (1 + r)**T * D0
    denominator = ((1 + r)**T - 1) / r
    return numerator / denominator

# Assigning values
r_values = np.linspace(0.01, 0.2, 50)  # Interest rate range
D0_values = np.linspace(50, 200, 50)    # Initial debt range
T_values = np.arange(5, 30, 5)          # Number of periods range

# Plotting
plt.figure(figsize=(12, 8))
for T in T_values:
    Z_star_values = [compute_repayment(r, T, D0) for r in r_values]
    plt.plot(r_values, Z_star_values, label=f'T={T}')

plt.xlabel('Interest Rate (r)')
plt.ylabel('Repayment (Z*)')
plt.title('Repayment vs. Interest Rate for Different Values of T')
plt.legend(title='Number of Periods (T)')
plt.grid(True)
plt.show()
"""
we plotted the value of the repayment Z* as a function of different values of 
the interest rate (r), initial debt (D), and number of periods (T). We can then
 provide the following results

Effect of Interest Rate (r): As the interest rate increases, the required 
repayment Z* generally increases as well. This is because higher interest 
rates lead to larger interest payments, requiring larger repayments to offset 
the growth of the debt.

Impact of Initial Debt (D) The initial debt also influences the required 
repayment Z*. Higher initial debts generally require larger repayments to reach
 zero debt by the end of the specified period.

Dependency on Number of Periods (T): The number of periods (T) plays a crucial 
role in determining the required repayment Z*. Longer periods typically require 
larger repayments, as the debt has more time to accumulate interest.

Interaction between Parameters: The relationship between Z, r, D, and T is 
complex and interdependent. Changes in one parameter can affect the required 
repayment, and these effects may be amplified or mitigated by changes in other 
parameters.

Policy Implications: Understanding how changes in interest rates, initial debt 
levels, and repayment periods affect the required repayment can inform 
policymakers about the trade-offs involved in debt management policies.
 Adjustments to interest rates or repayment schedules may be necessary to ensure
 debt sustainability and financial stability. Overall, the plot provides 
 valuable insights into the dynamics of debt repayment and how it is influenced
 by various economic factors.
 
 """
########################


# Function to compute debt for T periods
def compute_debt(T, r, Z, D0):
    debt = np.zeros(T+1)
    debt[0] = D0
    for t in range(T):
        debt[t+1] = debt[t] + r * debt[t] - Z
    return debt

# Function to find the repayment required for debt to reach 0
def find_repayment_for_zero_debt(D0, r, T):
    numerator = (1 + r)**T * D0
    denominator = ((1 + r)**T - 1) / r
    return numerator / denominator

# Assigning values
D0 = 150    # Initial debt
r = 0.05    # Interest rate
T = 20      # Number of periods

# Find the repayment required for debt to reach 0
Z_star = find_repayment_for_zero_debt(D0, r, T)

# Compute debt for T periods with this Z
debt = compute_debt(T, r, Z_star, D0)

# Plotting
time_periods = np.arange(T+1)
plt.plot(time_periods, debt, marker='o', linestyle='--', label='Debt',color='k')
plt.axhline(y=Z_star, color='g', linestyle='-', label='Repayment Z')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.title('Debt Dynamics Over Time with DT=0')
plt.legend()
plt.grid(True)
plt.plot(20, 0, marker='o', markersize=4, color='red')
plt.show()

"""
The graph depicts how with specific initial parameters we reach the point 
where initial debt is annihilated. This is an example of sustainable path
"""

###########################


###################################
#%% Exercize 3
print("_____________")
print("_____________")
print ('Exercize 3')
print('Keynes’ Multiplier as a Difference Equation')
print("_____________")
print("_____________")
 ##### Point B and C

import numpy as np
import matplotlib.pyplot as plt

def compute_income(b, i, g, T):
    time_periods = np.arange(T+1)
    return i * (1 - b**time_periods) / (1 - b) + g * (1 - b**time_periods) / (1 - b)

i = 100
g = 50
T = 20
b_values = [0.5, 0.7, 0.9]

plt.figure(figsize=(10, 6))
for b in b_values:
    income = compute_income(b, i, g, T)
    plt.plot(np.arange(T+1), income, marker='o', linestyle='-', label=f'b={b}')

plt.xlabel('Time (t)')
plt.ylabel('Income (y)')
plt.title('Dynamic Path of Income for Different Values of b')
plt.legend(title='b')
plt.grid(True)
plt.show()
#%% Point D
# Assigning values
i = 100  # Investment
T = 20   # Number of periods
b = 0.8  # Consumption parameter
g_values = [30, 50, 70]  # Different values of g

# Plotting
plt.figure(figsize=(10, 6))
for g in g_values:
    income = compute_income(b, i, g, T)
    plt.plot(np.arange(T+1), income, marker='o', linestyle='-', label=f'g={g}')

plt.xlabel('Time (t)')
plt.ylabel('Income (y)')
plt.title('Dynamic Path of Income for Different Values of g')
plt.legend(title='g')
plt.grid(True)
plt.show()
#%% Point E
# Function to compute the multiplier f(b,T--> ∞)
def multiplier(b, i, g, T):
    return (i + g) / ( (1 - b))

# Assigning values
i = 100  # Investment
g = 50   # Government spending
T_values = np.arange(1, 101)  # Number of periods range
b = 0.8  # Consumption parameter

# Compute multiplier for each T
multipliers = [multiplier(b, i, g, T) for T in T_values]

# Plotting
plt.plot(T_values, multipliers)
plt.xlabel('Number of Periods (T)')
plt.ylabel('Income')
plt.title('Income as T Approaches Infinity')
plt.grid(True)
plt.show()
"""
The multiplier represents the ratio by which an initial change in autonomous 
expenditure (such as investment or government spending) leads to a larger change
 in overall income or output in the economy. 

Here,i represents investment and g represents government spending. The expression
 {i + g}/{1 - b} denotes the equilibrium level of income y when consumption 
 depends on the income of the previous period Yt-1 with a marginal propensity to
 consume b. Interpreting this expression in terms of the multiplier, we can see 
 that an increase in investment i or government spending g leads to a multiplied 
 increase in equilibrium income. The size of this multiplier is determined by 
 the marginal propensity to consume b.
For this example, if b = 0.8, then the multiplier would be 1/{1 - 0.8} = 5 
 This means that for every euro increase in investment or government spending, 
 overall income would increase by five euros. 
Thus, the expression encapsulates the Keynesian idea that changes in autonomous 
expenditure can have amplified effects on overall income and output in the 
economy due to induced changes in consumption, leading to a multiplier effect.

In our example as T goes to infinite and the steady steate is reached
 we have an income constant at 750 €
"""
#%% Point F
# Initial values
C = 100   # Consumption
I = 50    # Investment
G = 50    # Government spending
y0 = C + I + G  # Initial equilibrium output

# Increase in government spending
G_new = 70  # New government spending
y1 = C + I + G_new  # New equilibrium output

# Plotting
plt.figure(figsize=(8, 6))
comment = 'The 45-degree line has a slope of 1,meaning that for every increase \nin aggregate output (y),\nthere\'s an equal increase in aggregate expenditure (AE)'
plt.annotate(comment, xy=(40, 151), fontsize=10, color='red')
# Aggregate expenditure function (AE)
y_values = [y0, y1]
plt.plot(y_values, y_values, color='black', linestyle='--', label='45-degree line')

# Initial AE curve
plt.plot([0, y0], [C + I, y0], color='blue', label='Initial AE curve')

# Increased government spending AE curve
plt.plot([0, y1], [C + I, y1], color='green', label='Increased G AE curve')


# Equilibrium points
plt.scatter([y0, y1], [y0, y1], color='red')
plt.xlabel('Aggregate Output (y)')
plt.ylabel('Aggregate Expenditure (AE)')
plt.title('Keynesian Cross Diagram')
plt.grid(True)
plt.legend()
plt.show()
#####################


############################################
#%% Exercize 4
print("_____________")
print("_____________")
print ('Exercize 4')
print('Non linear difference equation and Fixed point iteration')
print("_____________")
print("_____________")
##### Point B
import numpy as np
import matplotlib.pyplot as plt

def generate_sequence(alpha, beta, z, x0, T):
    sequence = [x0]
    for t in range(T):
        xt = z * (sequence[-1] ** alpha) + beta * sequence[-1]
        sequence.append(xt)
    return sequence

# Parameters
alpha = 0.5
beta = 0.3
z = 2
x0 = 1
T = 50  # Number of time steps

# Generate sequence
xt_sequence = generate_sequence(alpha, beta, z, x0, T)

# Find fixed point (point of convergence)
fixed_point = xt_sequence[-1]  # Last element of the sequence

# Plotting
plt.plot(range(T+1), xt_sequence, marker='o', linestyle='-', label='Sequence')
plt.axhline(y=fixed_point, color='r', linestyle='-', label='Fixed Point')
plt.text(57, fixed_point, f'{fixed_point:.2f}', fontsize=10, ha='right')
plt.xlabel('Time (t)')
plt.ylabel('Sequence value (xt)')
plt.title('Sequence with Fixed Point, given X0<X*')
plt.legend()
plt.grid(True)
plt.show()

"""
The provided code utilizes fixed point iteration to find the convergence point 
of a nonlinear difference equation. Through successive iterations, the sequence
 approaches the fixed point, representing the long-term behavior of the system.
 This method offers a straightforward approach for numerical approximation, 
 although its convergence depends on the characteristics of the equation.
"""
#%% Point C
x0 = 15
# Generate sequence
xt_sequence = generate_sequence(alpha, beta, z, x0, T)

# Find fixed point (point of convergence)
fixed_point = xt_sequence[-1]  # Last element of the sequence

# Plotting
plt.plot(range(T+1), xt_sequence, marker='o', linestyle='-', label='Sequence')
plt.axhline(y=fixed_point, color='r', linestyle='-', label='Fixed Point')
plt.text(57, fixed_point, f'{fixed_point:.2f}', fontsize=10, ha='right')
plt.xlabel('Time (t)')
plt.ylabel('Sequence value (xt)')
plt.title('Sequence with Fixed Point, given X0>X*')
plt.legend()
plt.grid(True)
plt.show()
"""
By observing points both above and below the fixed point, we can gain insight
 into the behavior of the sequence in relation to its convergence.
"""
#%% Point D
# Parameters
alphas = [0.2, 0.5, 0.8]  # Different values of alpha
betas = [0.3]             # Fixed value of beta
z_values = [2]            # Fixed value of z
x0 = 1                    # Initial condition
T = 100                   # Number of time steps

# Subplot for sequences with α changing
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('Sequences with α changing')
for alpha in alphas:
    for beta in betas:
        for z in z_values:
            xt_sequence = generate_sequence(alpha, beta, z, x0, T)
            plt.plot(range(T+1), xt_sequence, label=f'Alpha={alpha}, Beta={beta}, z={z}')
plt.xlabel('Time (t)')
plt.ylabel('Sequence value (xt)')
plt.legend()
plt.grid(True)

# Parameters for sequences with β changing
alphas = [0.5]  # Fixed value of alpha
betas = [0.2, 0.5, 0.8]  # Different values of beta
z_values = [2]  # Fixed value of z

plt.subplot(1, 3, 2)
plt.title('Sequences with β changing')
for alpha in alphas:
    for beta in betas:
        for z in z_values:
            xt_sequence = generate_sequence(alpha, beta, z, x0, T)
            plt.plot(range(T+1), xt_sequence, label=f'Alpha={alpha}, Beta={beta}, z={z}')
plt.xlabel('Time (t)')
plt.ylabel('Sequence value (xt)')
plt.legend()
plt.grid(True)

# Parameters for sequences with z changing
alphas = [0.5]  # Fixed value of alpha
betas = [0.3]   # Fixed value of β
z_values = [1, 2, 3]  # Different values of z

plt.subplot(1, 3, 3)
plt.title('Sequences with z changing')
for alpha in alphas:
    for beta in betas:
        for z in z_values:
            xt_sequence = generate_sequence(alpha, beta, z, x0, T)
            plt.plot(range(T+1), xt_sequence, label=f'Alpha={alpha}, Beta={beta}, z={z}')
plt.xlabel('Time (t)')
plt.ylabel('Sequence value (xt)')
plt.legend(bbox_to_anchor=(0.3, 0.7), loc='best')
plt.grid(True)

plt.tight_layout()
plt.show()

"""
 With this code we showed how the fixed point changes and how quickly the 
 sequence converges changing every time different parameters.
"""
 




