# %% Imports
import os

script_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(script_path))  # setting working directory for modules import

# My modules
from functions import *

os.chdir(os.path.dirname(os.path.dirname(script_path)))

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Standard packages
import numpy as np
import pandas as pd


#%%
data_dir = os.path.join(os.getcwd(), "data")
os.makedirs(data_dir, exist_ok=True)
patients_dir = os.path.join(os.getcwd(), "patients")
os.makedirs(patients_dir, exist_ok=True)
os.makedirs(os.path.join(patients_dir, "Group Statistics"), exist_ok=True)

files_list = os.listdir(data_dir)

#%%


#%%

def check_RR(rr):
    minimum_rr = min(rr)
    maximum_rr = max(rr)
    all_categories = [category for category in range(minimum_rr, maximum_rr+1, 8)]
    print(minimum_rr, maximum_rr)
    print(all_categories)
    print(len(set(rr)), len(all_categories))
    print(all_categories)
    print(sorted(set(rr)))
    
    if len(set(rr)) == len(all_categories):
        print("gra")
    
    else:
        print("nie gra")
    

for file in files_list:
    file_name, file_extension = os.path.splitext(file)
    df = pd.read_csv(os.path.join(data_dir, file_name+file_extension), sep='\t', names=['RR Interval', 'Index'])
    RR = df['RR Interval']
    check_RR(RR)


#%%

numerical_list = [648, 656, 664, 672, 680, 688, 600, 608, 616, 632, 640]

k=1
bins = [i for i in range(min(numerical_list), max(numerical_list)+(8*k+1), k*8)]
numerical_series = pd.Series(numerical_list)
categorical_series = pd.cut(numerical_series, bins=bins, labels=False, right=False)
categorical_list = categorical_series.tolist()

print("Original Numerical List:", numerical_list)
print("Converted Categorical List:", categorical_list)

#%%

def convert_to_categorical(rr_intervals, k):
    bins = [i for i in range(min(rr_intervals), max(rr_intervals)+(8*k+1), k*8)]
    numerical_series = pd.Series(rr_intervals)
    categorical_series = pd.cut(numerical_series, bins=bins, labels=False, right=False)
    categorical_list = categorical_series.tolist()
    return categorical_list


    
#%%
conv = convert_to_categorical(RR, 10)

# Tworzenie macierzy przejścia
macierz_przejscia = np.zeros((max(conv)+1, max(conv)+1), dtype=int)

# Przejście przez ciąg i zliczanie przejść


for i in range(len(conv)-1):
    aktualny_stan = conv[i]
    nastepny_stan = conv[i + 1]
    macierz_przejscia[aktualny_stan, nastepny_stan] += 1

print("Macierz Przejścia:")
print(macierz_przejscia)

macierz_przejscia_normalized = macierz_przejscia / macierz_przejscia.sum(axis=1, keepdims=True)

print(np.round(macierz_przejscia_normalized, 2))

#%%
# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(macierz_przejscia_normalized)

print("Eigenvalues:")
print(np.round(eigenvalues, 2))

print("\nEigenvectors:")
print(np.round(eigenvectors, 2))

#%%
import matplotlib.pyplot as plt
import numpy as np

def monte_carlo_simulation(transition_matrix, num_steps):
    num_states = len(transition_matrix)

    # Initialize system at a random state
    current_state = np.random.choice(np.arange(num_states))

    # Store the states during the simulation
    states_history = [current_state]

    # Monte Carlo simulation
    for step in range(num_steps):
        # Choose a new state based on the transition probabilities
        new_state = np.random.choice(np.arange(num_states), p=transition_matrix[current_state])

        # Update the current state
        current_state = new_state

        # Store the state in history
        states_history.append(current_state)

    return states_history

# Given transition matrix
P = np.array([[0.84, 0.15, 0.  , 0.01, 0.  , 0.  , 0.  , 0.  , 0.  ],
              [0.02, 0.79, 0.14, 0.02, 0.01, 0.01, 0.  , 0.  , 0.  ],
              [0.  , 0.13, 0.62, 0.16, 0.02, 0.05, 0.03, 0.  , 0.  ],
              [0.  , 0.01, 0.13, 0.69, 0.16, 0.01, 0.01, 0.  , 0.  ],
              [0.  , 0.  , 0.  , 0.06, 0.79, 0.14, 0.  , 0.  , 0.  ],
              [0.  , 0.  , 0.  , 0.  , 0.07, 0.87, 0.06, 0.  , 0.  ],
              [0.  , 0.  , 0.  , 0.  , 0.  , 0.37, 0.61, 0.02, 0.  ],
              [0.  , 0.  , 0.  , 0.  , 0.  , 0.03, 0.45, 0.52, 0.  ],
              [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.  ]])

# Number of Monte Carlo steps
num_steps = 100000

# Perform the Monte Carlo simulation
states_history = monte_carlo_simulation(macierz_przejscia_normalized, num_steps)

# Plot the states over time
plt.plot(states_history)
plt.title("Monte Carlo Simulation - State Transitions")
plt.xlabel("Time Steps")
plt.ylabel("State")
plt.show()





