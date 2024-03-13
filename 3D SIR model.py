# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set plot font size and figure size
plt.rcParams.update({'font.size': 15})
plt.rcParams["figure.figsize"] = [10, 8]

# Total population size
N = 7694005

# Initial conditions
S0 = 7694000 # Initial number of susceptible individuals
I0 = 5 # Initial number of infected individuals
R0 = 0 # Initial number of recovered individuals
initial_vals = [S0, I0, R0]

# Model Parameters
infection_rate = 0.213
recovery_rate = 1 / 14

# Function for simulation
def SIR_Euler(b, k, initial_conds):
    t0 = 0 # Initial time
    t_end = 346 # Endpoint time (approximately 1 year)
    h = 1 # Step size 1 day
    steps = int((t_end - t0) / h + 1) # Calculate the number of steps
    
    # Arrays to store variables values over time
    t = np.linspace(t0, t_end, steps)
    S = np.zeros(steps)
    I = np.zeros(steps)
    R = np.zeros(steps)
    
    # Initial conditions:
    S[0] = initial_conds[0]
    I[0] = initial_conds[1]
    R[0] = initial_conds[2]
    
    # Euler's method for solving the SIR model
    for n in range(steps - 1):
        S[n+1] = S[n] + h * (-b * S[n] * I[n] / N) # Using the Euler's method formula to find successive Susceptible population value
        I[n+1] = I[n] + h * (b * S[n] * I[n] / N - k * I[n])
        R[n+1] = R[n] + h * (k * I[n])

    return t, S, I, R # Return the arrays with t, S, I, R values 

# Run simulation
t, S, I, R = SIR_Euler(b=infection_rate, k=recovery_rate, initial_conds=initial_vals)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(S, I, R, c=I, cmap='cool', marker='o') # Scatter plot with colors based on I(t) values
ax.set_xlabel('Susceptible, millions', labelpad=15)
ax.set_ylabel('Infected, millions', labelpad=15)
ax.set_zlabel('Recovered, millions', labelpad=15)

# Create a color bar
cbar = plt.colorbar(sc, pad=0.1)
cbar.set_label('Infected population', labelpad = 10)

ax.set_title('SIR Simulation for b = 0.213, k = 0.071', y=1.05)
plt.show()