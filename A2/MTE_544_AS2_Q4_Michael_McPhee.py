"""
Student Name & Last Name: Michael McPhee
Origianl Author : Pi Thanacha Choopojcharoen
"""

import numpy as np

def decode_measurement(measurements):
    readings = []
    for measurement in measurements:
        # The one-hot measurement identifies F, R, or P by its nonzero entry.
        readings.append(int(np.argmax(measurement) + 1))  # +1 for 1-based indexing
    return readings

def moodeng_behavior_update(state, A):
    # Given the current state, and the transition matrix A, 
    # randomly return the next state of Moo-Deng based on A
    ##### ADD your code here : #####
    # Columns of A are the current state; rows are the possible next states.
    transition_probabilities = A[:, state - 1]
    next_state = int(np.random.choice(np.arange(1, A.shape[0] + 1), p=transition_probabilities))
    ##### END #####
    return next_state


def sensor_measurement(state, C):
    # Given a state, and the matrix C, 
    # randomly return the encoded measurement based on C
    # Note that : 
    # F -> np.array([1,0,0])
    # R -> np.array([0,1,0])
    # P -> np.array([0,0,1])
    ##### ADD your code here : #####
    # Sample a sensor outcome from the column for the current hidden state.
    measurement_index = np.random.choice(C.shape[0], p=C[:, state - 1])
    measurement = np.eye(C.shape[0], dtype=int)[measurement_index]
    ##### END #####
    return measurement

def sim_moodeng(initial_state=1,iteration = 20):
    # Given an initial state of Moo-Deng's whereabout and number of iteration,
    # simulate Moo-Deng's behavior and the designed state estimator
    
    ##### ADD your code here : #####
    # Start with complete certainty about the supplied initial state.
    A = np.array([
        [0.6, 0.2, 0.2],
        [0.4, 0.4, 0.1],
        [0.0, 0.4, 0.7],
    ])
    C = np.array([
        [0.8, 0.2, 0.05],
        [0.1, 0.7, 0.1],
        [0.1, 0.1, 0.85],
    ])
    belief = np.zeros(A.shape[0])
    belief[initial_state - 1] = 1.0
    ##### END #####

    states = []
    measurements = []
    estimated_states = []
    beliefs = []
    state = initial_state
    for i in range(iteration):
        ##### ADD your code here : #####
        next_state = moodeng_behavior_update(state, A)
        measurement = sensor_measurement(next_state, C)

        # Predict the next state, then update it using the new measurement.
        predicted_belief = A @ belief
        likelihood = C.T @ measurement
        unnormalized_belief = likelihood * predicted_belief
        # Normalize so the belief values form a probability distribution.
        updated_belief = unnormalized_belief / np.sum(unnormalized_belief)
        estimated_state = int(np.argmax(updated_belief) + 1)

        state = next_state
        belief = updated_belief
        ##### END #####
        states.append(next_state)
        measurements.append(measurement)
        beliefs.append(updated_belief)
        estimated_states.append(estimated_state)
        
    return states, measurements, estimated_states, beliefs

    
# Run the simulation

states, measurements, estimated, beliefs = sim_moodeng(initial_state=1, iteration=20)

readings = decode_measurement(measurements)
# Print results
print("True states:         ", states)
print("Sensor measurements: ", readings)
print("Estimated states:    ", estimated)
print("Belief sequence:      ", np.array(beliefs))
