# %% Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from functools import reduce

script_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(script_path))  # setting working directory for modules import

# My modules
from functions import (create_transition_matrix, classify_states, find_stationary_distribution_eigen,
                       find_stationary_distribution_limit, find_stationary_distribution_monte_carlo,
                       check_reversibility, save_matrix, write_file, write_reversibility_file, 
                       chunk_list, rr_to_events, create_transition_matrix_from_events,
                       find_stationary_distribution_eigen_other, create_graph,)

os.chdir(os.path.dirname(os.path.dirname(script_path)))

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# %% Tworzenie katalogów
data_dir = os.path.join(os.getcwd(), "data")
os.makedirs(data_dir, exist_ok=True)

patients_dir = os.path.join(os.getcwd(), "patients")
os.makedirs(patients_dir, exist_ok=True)

files_list = os.listdir(data_dir)

for file in files_list:
    file_name, file_extension = os.path.splitext(file)
    patient_path = os.path.join(patients_dir, file_name)
    os.makedirs(patient_path, exist_ok=True)

    for subsection in ["A", "B", "C", "D"]:
        os.makedirs(f'{patient_path}/{subsection}', exist_ok=True)

        for subcatalog in ["transition_matrices", "states_classification_and_stationary_distibutions", "graphs"]:
            os.makedirs(f'{patient_path}/{subsection}/{subcatalog}', exist_ok=True)

# %% A

print("A:")

for file in files_list:
    file_name, file_extension = os.path.splitext(file)
    df = pd.read_csv(os.path.join(data_dir, file_name + file_extension), sep='\t', names=['RR Interval', 'Index'])
    RR = df['RR Interval']

    for k in range(1, 11):
        # Tworzymy i zapisujemy macierz
        transition_matrix, states = create_transition_matrix(RR, k)
        save_matrix(transition_matrix, states, patients_dir, file_name, "A", f"transition_matrix_k{k}.txt")

        # Klasyfikacja stanów
        state_classifications = classify_states(transition_matrix)
        path = f'{patients_dir}/{file_name}/A/states_classification_and_stationary_distibutions/states_classification.txt'
        write_file(state_classifications, states, k, path)

        # Znajdujemy wektor stacjonarny za pomocą wartości własnych
        stationary_distribution_eigen = np.round(find_stationary_distribution_eigen(transition_matrix), 3)
        path = f'{patients_dir}/{file_name}/A/states_classification_and_stationary_distibutions/stationary_distribution_eig.txt'
        write_file(stationary_distribution_eigen, states, k, path)

        # Znajdujemy wektor stacjonarny za badając granica P^n n->inf
        stationary_distribution_limit = np.round(find_stationary_distribution_limit(transition_matrix), 3)
        path = f'{patients_dir}/{file_name}/A/states_classification_and_stationary_distibutions/stationary_distribution_pn.txt'
        write_file(stationary_distribution_limit, states, k, path)

        # Symulacja Monte Carlo
        stationary_distribution_monte_carlo = np.round(find_stationary_distribution_monte_carlo(transition_matrix), 3)
        path = f'{patients_dir}/{file_name}/A/states_classification_and_stationary_distibutions/stationary_distribution_monte_carlo.txt'
        write_file(stationary_distribution_monte_carlo, states, k, path)

        # Sprawdzamy odwracalność dynamiki
        is_reversible = check_reversibility(transition_matrix, stationary_distribution_eigen)
        path = f'{patients_dir}/{file_name}/A/states_classification_and_stationary_distibutions/chain_reversibility.txt'
        write_reversibility_file(is_reversible, k, path)

        # Tworzymy graf
        title = f'Graf Przejścia - {file_name} - k={k}'
        path = f'{patients_dir}/{file_name}/A/graphs/graph_k{k}'
        create_graph(transition_matrix, states, title, path)
    
    print(f"\t{file_name} -> ukończono")

print("\n-------------------------------------")


# %% B

print("B:")

lags = [1, 2, 3, 10]

for lag in lags:
    
    print(f"\n\tdelta = {lag}:")

    for file in files_list:
        file_name, file_extension = os.path.splitext(file)
        df = pd.read_csv(os.path.join(data_dir, file_name + file_extension), sep='\t', names=['RR Interval', 'Index'])
        RR = df['RR Interval']
        diffed_series = RR.diff(lag).dropna().astype(int)

        for k in range(1, 11):
            # Tworzymy i zapisujemy macierz
            transition_matrix, states = create_transition_matrix(diffed_series, k)
            save_matrix(transition_matrix, states, patients_dir, file_name, "B",
                        f"transition_matrix_k{k}_delta{lag}.txt")

            # Klasyfikacja stanów
            state_classifications = classify_states(transition_matrix)
            path = f'{patients_dir}/{file_name}/B/states_classification_and_stationary_distibutions/states_classification_delta{lag}.txt'
            write_file(state_classifications, states, k, path)

            # Znajdujemy wektor stacjonarny za pomocą wartości własnych
            stationary_distribution_eigen = np.round(find_stationary_distribution_eigen(transition_matrix), 3)
            path = f'{patients_dir}/{file_name}/B/states_classification_and_stationary_distibutions/stationary_distribution_eig_delta{lag}.txt'
            write_file(stationary_distribution_eigen, states, k, path)

            # Znajdujemy wektor stacjonarny za badając granica P^n n->inf
            stationary_distribution_limit = np.round(find_stationary_distribution_limit(transition_matrix), 3)
            path = f'{patients_dir}/{file_name}/B/states_classification_and_stationary_distibutions/stationary_distribution_pn_delta{lag}.txt'
            write_file(stationary_distribution_limit, states, k, path)

            # Symulacja Monte Carlo
            stationary_distribution_monte_carlo = np.round(find_stationary_distribution_monte_carlo(transition_matrix), 3)
            path = f'{patients_dir}/{file_name}/B/states_classification_and_stationary_distibutions/stationary_distribution_monte_carlo_delta{lag}.txt'
            write_file(stationary_distribution_monte_carlo, states, k, path)

            # Sprawdzamy odwracalność dynamiki
            is_reversible = check_reversibility(transition_matrix, stationary_distribution_eigen)
            path = f'{patients_dir}/{file_name}/B/states_classification_and_stationary_distibutions/chain_reversibility_delta{lag}.txt'
            write_reversibility_file(is_reversible, k, path)

            # Tworzymy graf
            title = f'Graf Przejścia - {file_name} - k={k} - delta={lag}'
            path = f'{patients_dir}/{file_name}/B/graphs/graph_k{k}_delta{lag}'
            create_graph(transition_matrix, states, title, path)
        
        print(f"\t\t{file_name} -> ukończono")

print("\n-------------------------------------")


# %% C

print("C:")

window_size = [30, 60, 120, 200]

for size in window_size:

    print(f"\n\twindows_size = {size}:")    

    for file in files_list:
        file_name, file_extension = os.path.splitext(file)
        df = pd.read_csv(os.path.join(data_dir, file_name + file_extension), sep='\t', names=['RR Interval', 'Index'])
        RR = df['RR Interval']
        splited_rr = list(chunk_list(RR, size))

        medians_from_windows = np.median(splited_rr, axis=1)
        stds_from_windows = np.std(splited_rr, axis=1)

        lowest_median_window = splited_rr[np.argmin(medians_from_windows)]
        highest_median_window = splited_rr[np.argmax(medians_from_windows)]
        lowest_std_window = splited_rr[np.argmin(stds_from_windows)]
        highest_std_window = splited_rr[np.argmax(stds_from_windows)]

        windows = [lowest_median_window, highest_median_window, lowest_std_window, highest_std_window]
        windows_names = ["lowest_median", "highest_median", "lowest_std", "highest_std"]

        windows_matrices = [create_transition_matrix(window, 1)[0] for window in windows]
        matrix_states = [create_transition_matrix(window, 1)[1] for window in windows]

        for transition_matrix, states, description in zip(windows_matrices, matrix_states, windows_names):
            # Zapisujemy macierz
            save_matrix(transition_matrix, states, patients_dir, file_name, "C",
                        f"{description}_transition_matrix_window{size}.txt")

            # Klasyfikacja stanów
            state_classifications = classify_states(transition_matrix)
            path = f'{patients_dir}/{file_name}/C/states_classification_and_stationary_distibutions/states_classification_window{size}.txt'
            write_file(state_classifications, states, description, path, "window")

            # Znajdujemy wektor stacjonarny za pomocą wartości własnych
            stationary_distribution_eigen = np.round(find_stationary_distribution_eigen_other(transition_matrix), 3)
            path = f'{patients_dir}/{file_name}/C/states_classification_and_stationary_distibutions/stationary_distribution_eig_window{size}.txt'
            write_file(stationary_distribution_eigen, states, description, path, "window")

            # Znajdujemy wektor stacjonarny za badając granica P^n n->inf
            stationary_distribution_limit = np.round(find_stationary_distribution_limit(transition_matrix), 3)
            path = f'{patients_dir}/{file_name}/C/states_classification_and_stationary_distibutions/stationary_distribution_pn_window{size}.txt'
            write_file(stationary_distribution_limit, states, description, path, "window")

            # Symulacja Monte Carlo
            stationary_distribution_monte_carlo = np.round(find_stationary_distribution_monte_carlo(transition_matrix), 3)
            path = f'{patients_dir}/{file_name}/C/states_classification_and_stationary_distibutions/stationary_distribution_monte_carlo_window{size}.txt'
            write_file(stationary_distribution_monte_carlo, states, description, path, "window")

            # Sprawdzamy odwracalność dynamiki
            is_reversible = check_reversibility(transition_matrix, stationary_distribution_eigen)
            path = f'{patients_dir}/{file_name}/C/states_classification_and_stationary_distibutions/chain_reversibility_window{size}.txt'
            write_reversibility_file(is_reversible, description, path, "window")

            # Tworzymy graf
            title = f'Graf Przejścia - {file_name} - window={description} - window size={size}'
            path = f'{patients_dir}/{file_name}/C/graphs/{description}_window{size}'
            create_graph(transition_matrix, states, title, path)
        
        print(f"\t\t{file_name} -> ukończono")

print("\n-------------------------------------")

# %%D

print("D:")

lags = [1, 2, 3, 10]
k = 1
single_events = ["A", "D", "a", "d", "z"]
double_events = ["".join(pair) for pair in product(single_events, repeat=2)]
tripple_events = ["".join(pair) for pair in product(single_events, repeat=3)]

for lag in lags:
    
    print(f"\n\tdelta = {lag}:")
    
    for file in files_list:
        file_name, file_extension = os.path.splitext(file)
        df = pd.read_csv(os.path.join(data_dir, file_name + file_extension), sep='\t', names=['RR Interval', 'Index'])
        RR = df['RR Interval']
        diffed_series = RR.diff(lag).dropna().astype(int)

        series_symbolization = rr_to_events(RR, lag)
        pairs_in_series_symbolization = [reduce(lambda x, y: x + y, tpl) for tpl in list(chunk_list(series_symbolization, 2))]
        tripples_in_series_symbolization = [reduce(lambda x, y: x + y, tpl) for tpl in list(chunk_list(series_symbolization, 3))]

        # POJEDYNCZE WYDARZENIA
        # Tworzymy i zapisujemy macierz
        transition_matrix, states = create_transition_matrix_from_events(series_symbolization, single_events)
        save_matrix(transition_matrix, states, patients_dir, file_name, "D", f"transition_matrix_1elem_delta{lag}.txt")

        # Klasyfikacja stanów
        state_classifications = classify_states(transition_matrix)
        path = f'{patients_dir}/{file_name}/D/states_classification_and_stationary_distibutions/states_classification_1elem.txt'
        write_file(state_classifications, states, lag, path, "delta")

        # Znajdujemy wektor stacjonarny za pomocą wartości własnych
        stationary_distribution_eigen = np.round(find_stationary_distribution_eigen(transition_matrix), 3)
        path = f'{patients_dir}/{file_name}/D/states_classification_and_stationary_distibutions/stationary_distribution_eig_1elem.txt'
        write_file(stationary_distribution_eigen, states, lag, path, "delta")

        # Znajdujemy wektor stacjonarny za badając granica P^n n->inf
        stationary_distribution_limit = np.round(find_stationary_distribution_limit(transition_matrix), 3)
        path = f'{patients_dir}/{file_name}/D/states_classification_and_stationary_distibutions/stationary_distribution_pn_1elem.txt'
        write_file(stationary_distribution_limit, states, lag, path, "delta")

        # Symulacja Monte Carlo
        stationary_distribution_monte_carlo = np.round(find_stationary_distribution_monte_carlo(transition_matrix), 3)
        path = f'{patients_dir}/{file_name}/D/states_classification_and_stationary_distibutions/stationary_distribution_monte_carlo_1elem.txt'
        write_file(stationary_distribution_monte_carlo, states, lag, path, "delta")

        # Sprawdzamy odwracalność dynamiki
        is_reversible = check_reversibility(transition_matrix, stationary_distribution_eigen)
        path = f'{patients_dir}/{file_name}/D/states_classification_and_stationary_distibutions/chain_reversibility_1elem.txt'
        write_reversibility_file(is_reversible, lag, path, "delta")

        # Tworzymy graf
        title = f'Graf Przejścia - {file_name} - 1 elem - delta={lag}'
        path = f'{patients_dir}/{file_name}/D/graphs/graph_1elem_delta{lag}'
        create_graph(transition_matrix, states, title, path)

        # PODWÓJNE ZLEPKI
        # Tworzymy i zapisujemy macierz
        transition_matrix, states = create_transition_matrix_from_events(pairs_in_series_symbolization, double_events)
        save_matrix(transition_matrix, states, patients_dir, file_name, "D", f"transition_matrix_2elem_delta{lag}.txt")

        # Klasyfikacja stanów
        state_classifications = classify_states(transition_matrix)
        path = f'{patients_dir}/{file_name}/D/states_classification_and_stationary_distibutions/states_classification_2elem.txt'
        write_file(state_classifications, states, lag, path, "delta")

        # Znajdujemy wektor stacjonarny za pomocą wartości własnych
        stationary_distribution_eigen = np.round(find_stationary_distribution_eigen(transition_matrix), 3)
        path = f'{patients_dir}/{file_name}/D/states_classification_and_stationary_distibutions/stationary_distribution_eig_2elem.txt'
        write_file(stationary_distribution_eigen, states, lag, path, "delta")

        # Znajdujemy wektor stacjonarny za badając granica P^n n->inf
        stationary_distribution_limit = np.round(find_stationary_distribution_limit(transition_matrix), 3)
        path = f'{patients_dir}/{file_name}/D/states_classification_and_stationary_distibutions/stationary_distribution_pn_2elem.txt'
        write_file(stationary_distribution_limit, states, lag, path, "delta")

        # Symulacja Monte Carlo
        stationary_distribution_monte_carlo = np.round(find_stationary_distribution_monte_carlo(transition_matrix), 3)
        path = f'{patients_dir}/{file_name}/D/states_classification_and_stationary_distibutions/stationary_distribution_monte_carlo_2elem.txt'
        write_file(stationary_distribution_monte_carlo, states, lag, path, "delta")

        # Sprawdzamy odwracalność dynamiki
        is_reversible = check_reversibility(transition_matrix, stationary_distribution_eigen)
        path = f'{patients_dir}/{file_name}/D/states_classification_and_stationary_distibutions/chain_reversibility_2elem.txt'
        write_reversibility_file(is_reversible, lag, path, "delta")

        # Tworzymy graf
        title = f'Graf Przejścia - {file_name} - 2 elem - delta={lag}'
        path = f'{patients_dir}/{file_name}/D/graphs/graph_2elem_delta{lag}'
        create_graph(transition_matrix, states, title, path)

        # POTRÓJNE ZLEPKI
        # Tworzymy i zapisujemy macierz
        transition_matrix, states = create_transition_matrix_from_events(tripples_in_series_symbolization, tripple_events)
        save_matrix(transition_matrix, states, patients_dir, file_name, "D", f"transition_matrix_3elem_delta{lag}.txt")

        # Klasyfikacja stanów
        state_classifications = classify_states(transition_matrix)
        path = f'{patients_dir}/{file_name}/D/states_classification_and_stationary_distibutions/states_classification_3elem.txt'
        write_file(state_classifications, states, lag, path, "delta")

        # Znajdujemy wektor stacjonarny za pomocą wartości własnych
        stationary_distribution_eigen = np.round(find_stationary_distribution_eigen(transition_matrix), 3)
        path = f'{patients_dir}/{file_name}/D/states_classification_and_stationary_distibutions/stationary_distribution_eig_3elem.txt'
        write_file(stationary_distribution_eigen, states, lag, path, "delta")

        # Znajdujemy wektor stacjonarny za badając granica P^n n->inf
        stationary_distribution_limit = np.round(find_stationary_distribution_limit(transition_matrix), 3)
        path = f'{patients_dir}/{file_name}/D/states_classification_and_stationary_distibutions/stationary_distribution_pn_3elem.txt'
        write_file(stationary_distribution_limit, states, lag, path, "delta")

        # Symulacja Monte Carlo
        stationary_distribution_monte_carlo = np.round(find_stationary_distribution_monte_carlo(transition_matrix), 3)
        path = f'{patients_dir}/{file_name}/D/states_classification_and_stationary_distibutions/stationary_distribution_monte_carlo_3elem.txt'
        write_file(stationary_distribution_monte_carlo, states, lag, path, "delta")

        # Sprawdzamy odwracalność dynamiki
        is_reversible = check_reversibility(transition_matrix, stationary_distribution_eigen)
        path = f'{patients_dir}/{file_name}/D/states_classification_and_stationary_distibutions/chain_reversibility_3elem.txt'
        write_reversibility_file(is_reversible, lag, path, "delta")

        # Tworzymy graf
        title = f'Graf Przejścia - {file_name} - 3 elem - delta={lag}'
        path = f'{patients_dir}/{file_name}/D/graphs/graph_3elem_delta{lag}'
        create_graph(transition_matrix, states, title, path)
        
        print(f"\t\t{file_name} -> ukończono")

print("\n-------------------------------------")

print("! UKOŃCZONO !")


