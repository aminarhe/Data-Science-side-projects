 # Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Set plot font size and figure size
plt.rcParams.update({'font.size': 14})
plt.rcParams["figure.figsize"] = [8,5]

# Total population size
N = 7694000

# Initial conditions: 
S0 = 7694000 - 5 # Initial number of susceptible individuals
I0 = 5 # Initial number of infected individuals
R0 = 0 # Initial number of recovered individuals
initial_vals = [S0,I0, R0]

# Model parameters:
infection_rate = 0.213
recovery_rate = 1 / 14  # Recovery rate, assuming 14-day recovery period


def SIR_Euler(b,k,initial_conds):
    t0 = 0 # Initial time
    t_end = 346 # Endpoint time (approximately 1 year)
    h = 1 # Step size 1 day
    steps = int((t_end - t0)/h + 1) # Calculate the number of steps

    # Arrays to store variables values over time
    t = np.linspace(t0, t_end, steps) # Storing t values
    S = np.zeros(steps) # Storing Susceptible population values
    I = np.zeros(steps) # Storing Infected population values
    R = np.zeros(steps) # Storing Recovered population values

    # Initial conditions:
    S[0] = initial_conds[0] 
    I[0] = initial_conds[1]
    R[0] = initial_conds[2]
    
    # Euler's method for solving the SIR model 
    for n in range(steps-1): # At each step, compute the subsequent values of S, I, and R using the functions' values and their derivatives at the previous step
        S[n+1] = S[n] + h * (-b * S[n] * I[n] / N) # Using the Euler's method formula to find subsequent Susceptible population value
        I[n+1] = I[n] + h * (b * S[n] * I[n] / N - k * I[n])
        R[n+1] = R[n] + h * (k * I[n])

    # Plot the results
    plt.plot(t, S, linewidth=2, label='S(t), Susceptible')
    plt.plot(t, I, linewidth=2, label='I(t), Infected')
    plt.plot(t, R, linewidth=2, label='R(t), Recovered')  # Plot the recovered population
    plt.ylim(0, 8000000)
    plt.xlabel('t, days')
    plt.ylabel('Population, millions')
    plt.legend(loc='best')
    plt.title(f'COVID-19 in Washington, SIR simulation for b = {infection_rate}, k = {round(recovery_rate,3)}')
    plt.show()

# Call the function to run the simulation
SIR_Euler(b=infection_rate, k=recovery_rate, initial_conds=initial_vals)
