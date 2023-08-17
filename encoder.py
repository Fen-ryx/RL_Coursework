import numpy as np
import argparse

def readStates(path):
    file = open(path, 'r')
    states = file.read()
    file.close()

    num_balls, num_runs = int(states[:2]), int(states[2:4])
    states_list, state_to_index_map, index_to_state_map, i = [], {}, {}, 0
    states = states.split('\n')
    
    for elem in states:
        if elem != '':
            stateA = (int(elem[:2]), int(elem[2:]), 'A')
            stateB = (int(elem[:2]), int(elem[2:]), 'B')
            states_list.append(stateA)
            states_list.append(stateB)
            state_to_index_map[stateA] = i
            index_to_state_map[i] = stateA
            i += 1
            state_to_index_map[stateB] = i
            index_to_state_map[i] = stateB
            i += 1
    
    end_states = [(-1, -1, None), (0, 1, None), (1, 0, None)]

    for state in end_states:
        state_to_index_map[state] = i
        index_to_state_map[i] = state
        i += 1
    
    states_list += end_states
    return num_balls, num_runs, states_list, state_to_index_map, index_to_state_map, end_states

def readParams(path):
    file = open(path, 'r')
    parameters = file.read()
    file.close()

    parameters = parameters.split('\n')
    parameters = parameters[1:]

    choice_to_index_map = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4}
    index_to_choice_map = {0: 0, 1: 1, 2: 2, 3: 4, 4: 6}
    outcome_to_index_map = {-1: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 6: 6}
    index_to_outcome_map = {0: -1, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 6}

    choice_outcome_prob_matrix = np.zeros((5, 7))
    for i, row in enumerate(parameters):
        elements = row.split(' ')
        elements = np.array(elements[1:], dtype=float)
        choice_outcome_prob_matrix[i, :] = elements
    
    return choice_outcome_prob_matrix, choice_to_index_map, index_to_choice_map, outcome_to_index_map, index_to_outcome_map

def encode(state_path, params_path, q):
    num_balls, num_runs, states_list, state_to_index_map, index_to_state_map, end_states = readStates(state_path)
    choice_outcome_prob_matrix, choice_to_index_map, index_to_choice_map, outcome_to_index_map, index_to_outcome_map = readParams(params_path)

    num_states = len(states_list)
    num_actions = len(choice_outcome_prob_matrix)
    num_outcomes = num_actions + 2
    
    transition_matrix , reward_matrix = loadTransitionsAndRewards(num_balls, num_states, num_actions, end_states, state_to_index_map, index_to_state_map, choice_outcome_prob_matrix, index_to_outcome_map, q)

    mdp_type = 'episodic'
    gamma = 1.0

    def write():
        print('numStates ' + str(num_states))
        print('numActions ' + str(num_actions))
        print('end ',end='')
        for i, state in enumerate(end_states):
            state_index = state_to_index_map[state]
            if i == len(end_states) - 1:
                print(str(state_index))
            else:
                print(str(state_index) + ' ', end='')

        for state in range(num_states):
            bb, rr, player = index_to_state_map[state]

            for action in range(num_actions):
                for next_state in range(num_states):
                    if transition_matrix[action][state][next_state] != 0:
                        prob, reward = transition_matrix[action][state][next_state], reward_matrix[action][state][next_state]
                        print('transition ' + str(state) + ' ' + str(action) + ' ' + str(next_state) + ' ' + str(reward) + ' ' + str(prob))
        
        print('mdptype episodic')
        print('discount ' + str(gamma))
    
    write()

def loadTransitionsAndRewards(num_balls, num_states, num_actions, end_states, state_to_index_map, index_to_state_map, choice_outcome_prob_matrix, index_to_outcome_map, q):
    transition_matrix = np.zeros((num_actions, num_states, num_states))
    reward_matrix = np.zeros((num_actions, num_states, num_states))

    for action in range(num_actions):
        action_outcomes = choice_outcome_prob_matrix[action, :]
        nonzero_outcomes = np.nonzero(action_outcomes)
        nonzero_outcomes = nonzero_outcomes[0]

        for state in range(num_states):
            current_balls, current_runs, player = index_to_state_map[state]
            if player == 'B':
                outcome = -1
                next_state = (-1, -1, None)
                transition_matrix[action][state][state_to_index_map[next_state]] = q
                reward_matrix[action][state][state_to_index_map[next_state]] = 0

                outcome = 0
                new_balls, new_runs = current_balls - 1, current_runs
                if new_balls == 0 and new_runs > 0:
                    next_state = (0, 1, None)
                    transition_matrix[action][state][state_to_index_map[next_state]] = (1 - q) / 2
                    reward_matrix[action][state][state_to_index_map[next_state]] = 0
                
                elif new_balls != 6 and new_balls != 12:
                    next_state = (new_balls, new_runs, 'B')
                    transition_matrix[action][state][state_to_index_map[next_state]] = (1 - q) / 2
                    reward_matrix[action][state][state_to_index_map[next_state]] = 0
                
                elif new_balls == 6 or new_balls == 12:
                    next_state = (new_balls, new_runs, 'A')
                    transition_matrix[action][state][state_to_index_map[next_state]] = (1 - q) / 2
                    reward_matrix[action][state][state_to_index_map[next_state]] = 0
                
                outcome = 1
                new_balls, new_runs = current_balls - 1, current_runs - 1
                if new_balls == 0 and new_runs > 0:
                    next_state = (0, 1, None)
                    transition_matrix[action][state][state_to_index_map[next_state]] = (1 - q) / 2
                    reward_matrix[action][state][state_to_index_map[next_state]] = 0
                
                elif new_balls >= 0 and new_runs <= 0:
                    next_state = (1, 0, None)
                    transition_matrix[action][state][state_to_index_map[next_state]] = (1 - q) / 2
                    reward_matrix[action][state][state_to_index_map[next_state]] = 1
                
                elif new_balls == 6 or new_balls == 12:
                    next_state = (new_balls, new_runs, 'B')
                    transition_matrix[action][state][state_to_index_map[next_state]] = (1 - q) / 2
                    reward_matrix[action][state][state_to_index_map[next_state]] = 0
                
                elif new_balls !=6 and new_balls != 12:
                    next_state = (new_balls, new_runs, 'A')
                    transition_matrix[action][state][state_to_index_map[next_state]] = (1 - q) / 2
                    reward_matrix[action][state][state_to_index_map[next_state]] = 0
            
            elif player == 'A':
                for outcome_index in nonzero_outcomes:
                    runs_scored = index_to_outcome_map[outcome_index]
                    new_balls, new_runs = current_balls - 1, current_runs - runs_scored

                    if runs_scored == -1:
                        next_state = (-1, -1, None)
                        transition_matrix[action][state][state_to_index_map[next_state]] = action_outcomes[outcome_index]
                        reward_matrix[action][state][state_to_index_map[next_state]] = 0
                    
                    elif new_balls == 0 and new_runs > 0:
                        next_state = (0, 1, None)
                        transition_matrix[action][state][state_to_index_map[next_state]] = action_outcomes[outcome_index]
                        reward_matrix[action][state][state_to_index_map[next_state]] = 0
                    
                    elif new_balls >= 0 and new_runs <= 0:
                        next_state = (1, 0, None)
                        transition_matrix[action][state][state_to_index_map[next_state]] += action_outcomes[outcome_index]
                        reward_matrix[action][state][state_to_index_map[next_state]] = 1
                    
                    elif runs_scored % 2 == 0 and (new_balls != 6 and new_balls != 12):
                        next_state = (new_balls, new_runs, 'A')
                        transition_matrix[action][state][state_to_index_map[next_state]] = action_outcomes[outcome_index]
                        reward_matrix[action][state][state_to_index_map[next_state]] = 0
                    
                    elif runs_scored % 2 != 0 and (new_balls == 6 or new_balls == 12):
                        next_state = (new_balls, new_runs, 'A')
                        transition_matrix[action][state][state_to_index_map[next_state]] = action_outcomes[outcome_index]
                        reward_matrix[action][state][state_to_index_map[next_state]] = 0
                    
                    elif runs_scored % 2 == 0 and (new_balls == 6 or new_balls == 12):
                        next_state = (new_balls, new_runs, 'B')
                        transition_matrix[action][state][state_to_index_map[next_state]] = action_outcomes[outcome_index]
                        reward_matrix[action][state][state_to_index_map[next_state]] = 0
                    
                    elif runs_scored % 2 != 0 and (new_balls != 6 and new_balls != 12):
                        next_state = (new_balls, new_runs, 'B')
                        transition_matrix[action][state][state_to_index_map[next_state]] = action_outcomes[outcome_index]
                        reward_matrix[action][state][state_to_index_map[next_state]] = 0
    
    return transition_matrix, reward_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--states', type=str)
    parser.add_argument('--parameters', type=str)
    parser.add_argument('--q', type=str)

    args = parser.parse_args()
    statesPath, paramsPath, q = args.states, args.parameters, float(args.q)

    encode(statesPath, paramsPath, q)