import numpy as np

'''Discrete-Time Markov Chain Model'''
# Uses transition probabilities between discrete states to predict future states.

transition_matrix = np.array([
    [0.5, 0.4, 0.1, 0.0, 0.0], 
    [0.3, 0.4, 0.2, 0.1, 0.0],
    [0.1, 0.3, 0.4, 0.2, 0.0],  
    [0.0, 0.2, 0.3, 0.4, 0.1],
    [0.0, 0.1, 0.2, 0.4, 0.3] 
])

category_to_index = {
    "Good": 0,
    "Moderate": 1,
    "Poor": 2,
    "Very Poor": 3,
    "Hazardous": 4
}

index_to_category = {v: k for k, v in category_to_index.items()}  

def predict_next_state(current_category):
    """Predicts future air quality state based on the current category."""
    if current_category not in category_to_index:
        raise ValueError(f"Invalid Air Quality Category: {current_category}")

    current_index = category_to_index[current_category]
    next_state_index = np.random.choice([0, 1, 2, 3, 4], p=transition_matrix[current_index])
    return index_to_category[next_state_index]

print(" DTMC Model Fixed & Ready!")
