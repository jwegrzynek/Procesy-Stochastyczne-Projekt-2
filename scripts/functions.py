import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import random
from itertools import islice


def remove_zero_rows_and_columns(matrix):

    """
    Function that removes from matrix row made of zeros and corresponding column
    
    Parameters:
        - matrix - list of lists
    Returns:
        - reduced_matrix - matrix with removed rows and columns made of zeros
        - non_zero_rows_and_columns - list of binary values. True corresponds 
                                      to an index where the row and column was 
                                      non-empty, False corresponds to a row and 
                                      column consisting of zeros
    """

    non_zero_rows = np.any(matrix != 0, axis=1)
    non_zero_columns = np.any(matrix != 0, axis=0)
    non_zero_rows_and_columns = np.logical_or(non_zero_rows, non_zero_columns)
    reduced_matrix = matrix[non_zero_rows_and_columns][:, non_zero_rows_and_columns]
    
    return reduced_matrix, non_zero_rows_and_columns


def return_states_names(bins, k):
    
    """
    Function that returns a dictionary with descriptions of states that looks 
    like this {state number: description, ...}
    e.g. {0: "608-616",
          1: "624-632"}
    
    Parameters:
        - bins - a list with interval boundaries that are used to divide into classes
        - k
    """
    
    if k == 1:
        states_dict = {i: f'{num}' for i, num in enumerate(bins[:-1])}
    else:
        states_dict = {i: f'{num}-{bins[i+1]-8}' for i, num in enumerate(bins[:-1])}
    return states_dict


def return_bins(rr_intervals, k):
    
    """
    Function that returns a list with interval boundaries that are used to divide into classes
    
    Parameters:
        - rr_intervals - list of RR Intervals
        - k - (int) - k*8ms
    """
    
    bins = [i for i in range(min(rr_intervals), max(rr_intervals)+(8*k+1), k*8)]
    return bins


def convert_to_categorical(rr_intervals, k):
    """
    Function that converts RR Intervals to categorical variables depending on bins.
    For example if k=1 [600, 608, 616] -> [0, 1, 2]
                if k=2 [600, 608, 616] -> [0, 0, 1]
                if k=3 [600, 608, 616] -> [0, 0, 0]
                
    Parameters:
        - rr_intervals - list of RR Intervals
        - k - (int) - k*8ms
    Returns:
        - list of converted values
        - dictionary with descriptions of states
    """
    
    bins = [i for i in range(min(rr_intervals), max(rr_intervals)+(8*k+1), k*8)]
    numerical_series = pd.Series(rr_intervals)
    categorical_series = pd.cut(numerical_series, bins=bins, labels=False, right=False)
    categorical_list = categorical_series.tolist()
    states = return_states_names(bins, k)
    
    return categorical_list, states


def create_transition_matrix(rr_intervals, k):
    
    """
    Function that converts RR Intervals into transition matrix
    
    Parameters:
        - rr_intervals - list of RR Intervals
        - k - (int) - k*8ms
    Returns:
        - transition matrix
        - dictionary with descriptions of states
    """
    
    list_of_states, states = convert_to_categorical(rr_intervals, k)

    n = max(list_of_states) + 1
    count_matrix = np.zeros((n, n), dtype=int)
    for i in range(len(list_of_states)-1):
        current_state = list_of_states[i]
        next_state = list_of_states[i + 1]
        count_matrix[current_state, next_state] += 1

    transition_matrix = count_matrix / count_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.nan_to_num(transition_matrix, nan=0)
    
    transition_matrix, removed_indexes = remove_zero_rows_and_columns(transition_matrix)
    
    if sum(removed_indexes) != len(removed_indexes):
        states = {key: value for key, value in states.items() if removed_indexes[key]}
        states = {i: f'{item[1]}' for i, item in enumerate(states.items())}

    return transition_matrix, states


def create_transition_matrix_from_events(list_of_states, events):
    
    """
    Function that converts list of events that occur in RR Intervals e.g. into transition matrix
    
    Parameters:
        - list_of_states - list that takes list of strings
    Returns:
        - transition matrix
    """

    n = len(events)
    count_matrix = np.zeros((n, n), dtype=int)
    for i in range(len(list_of_states) - 1):
        current_state_index = events.index(list_of_states[i])
        next_state_index = events.index(list_of_states[i + 1])
        count_matrix[current_state_index, next_state_index] += 1

    transition_matrix = count_matrix / count_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.nan_to_num(transition_matrix, nan=0)
    
    transition_matrix, removed_indexes = remove_zero_rows_and_columns(transition_matrix)
    
    states = {i: f'{item}' for i, item in enumerate(events)}
    
    if sum(removed_indexes) != len(removed_indexes):
        states = {key: value for key, value in states.items() if removed_indexes[key]}
        states = {i: f'{item[1]}' for i, item in enumerate(states.items())}
    
    return transition_matrix, states


###############################################################################

def rr_to_events(rr_intervals, lag):
    
    """
    Function that converts RR Intervals into series symbolization
    
    Parameters:
        - rr_intervals - list of RR Intervals
    Returns:
        - list that represents series symbolization e.g. ['a', 'd', 'z', 'a']
    """
    
    delta_rr = rr_intervals.diff(lag).dropna()
    series_symbolization = list(np.zeros(len(delta_rr), dtype=object))

    for i, diff in enumerate(delta_rr):
        if 0 < diff < 40:
            series_symbolization[i] = "d"
        elif -40 < diff < 0:
            series_symbolization[i] = "a"
        elif 40 <= diff:
            series_symbolization[i] = "D"
        elif diff <= -40:
            series_symbolization[i] = "A"
        else:
            series_symbolization[i] = "z"
            
    return series_symbolization


def chunk_list(flattened_list, chunk_size):
    iterator = iter(flattened_list)
    chunked_list = iter(lambda: tuple(islice(iterator, chunk_size)), ())
    chunked_list = filter(lambda x: len(x) == chunk_size, chunked_list)
    return chunked_list


def classify_states(transition_matrix):
    num_states = len(transition_matrix)
    communication_classes = []

    visited = np.zeros(num_states, dtype=bool)

    def dfs(current_state, communication_class):
        visited[current_state] = True
        communication_class.add(current_state)

        for next_state_prob in transition_matrix[current_state]:
            next_state = np.argmax(next_state_prob)
            if not visited[next_state]:
                dfs(next_state, communication_class)

    for state in range(num_states):
        if not visited[state]:
            communication_class = set()
            dfs(state, communication_class)
            communication_classes.append(communication_class)

    state_classifications = []

    for communication_class in communication_classes:
        if any(transition_matrix[state][state] > 0 for state in communication_class):
            state_classifications.append("Recurrent")
        else:
            state_classifications.append("Transient")

    return state_classifications


def find_stationary_distribution_eigen(P):
    # Obliczamy wartości własne i wektory własne
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    
    # Znajdujemy indeksy wartości własnych bliskich 1
    stationary_indices = np.where(np.isclose(eigenvalues, 1))[0]
    
    # Wybieramy odpowiadające wektory własne
    stationary_vectors = eigenvectors[:, stationary_indices]
    
    # Normalizujemy, aby suma elementów była równa 1
    stationary_distribution = stationary_vectors[:, 0] / np.sum(stationary_vectors[:, 0])
    stationary_distribution = np.real_if_close(stationary_distribution)
    
    return stationary_distribution


def find_stationary_distribution_eigen_other(P):
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    stationary_indices = eigenvalues.argmax()
    stationary_distribution = (eigenvectors[:,stationary_indices]/eigenvectors.sum())
    stationary_distribution = np.real_if_close(stationary_distribution)
    return stationary_distribution


def find_stationary_distribution_monte_carlo(P, num_steps=10000):
    num_states = P.shape[0]
    current_state = np.random.choice(num_states)  # Inicjalizacja losowym stanem

    state_counts = np.zeros(num_states)

    for _ in range(num_steps):
        state_counts[current_state] += 1
        try:
            next_state = np.random.choice(num_states, p=P[current_state])
        except ValueError:
            random_index = random.randint(0, len(P[current_state]) - 1)
            P[current_state][random_index] = 1
            next_state = np.random.choice(num_states, p=P[current_state])
        
        current_state = next_state

    stationary_distribution = state_counts / num_steps
    return stationary_distribution


def find_stationary_distribution_limit(P):
    
    p_n = np.round(np.linalg.matrix_power(P, 100), 3)
    stationary_distribution = p_n[0]
    
    return stationary_distribution


def check_reversibility(P, stationary_distribution):
    num_states = P.shape[0]

    for i in range(num_states):
        for j in range(num_states):
            left_side = stationary_distribution[i] * P[i, j]
            right_side = stationary_distribution[j] * P[j, i]

            if not np.isclose(left_side, right_side):
                return "Chain is NOT reversible"

    return "Chain is reversible"


def create_graph(transition_matrix, labels, title, path):
    options = {'node_color': 'skyblue',
               'node_size': 5000,
               'width': 1,
               'arrowstyle': '->',
               'arrowsize': 20
               }
    plt.rcParams["figure.figsize"] = (20, 15)
    plt.figure()
    graph = nx.DiGraph(transition_matrix)
    nx.draw_networkx(graph, arrows=True, labels=labels, **options)
    plt.title(title, fontsize=35)
    plt.savefig(path)
    

def save_matrix(transition_matrix, states, patients_dir, patient_name, subsection, file_name):
    matrix = np.round(transition_matrix, 2)
    row_column_names = states
    index = [row_column_names[i] for i in range(matrix.shape[0])]
    columns = [row_column_names[i] for i in range(matrix.shape[1])]
    df_matrix = pd.DataFrame(matrix, index=index, columns=columns)
    path = f'{patients_dir}/{patient_name}/{subsection}/transition_matrices/{file_name}'
    df_matrix.to_csv(path, sep='\t')


def write_file(iterable, states, k, path, description="k"):
    with open(path, 'a') as file:
        file.write(f'---------{description} = {k}---------\n')
        for i, item in enumerate(iterable):
            file.write(f'{i}\tState {states[i]} -> {item}\n')
        file.write('\n')
        file.close()
        
def write_reversibility_file(is_reversible, k, path, description="k"):
    with open(path, 'a') as file:
        file.write(f'---------{description} = {k}---------\n')
        file.write(is_reversible+'\n')
        file.write('\n')
        file.close()