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

def readValuePolicy(value_policy_file):
    file = open(value_policy_file, 'r')
    value_policy = file.read()
    file.close()

    values, policies = [], []
    value_policy = value_policy.split('\n')

    for element in value_policy:
        if element != '':
            value_policy_elem = element.split()
            value, policy = float(value_policy_elem[0]), int(value_policy_elem[1])
            values.append(value)
            policies.append(policy)
    
    return values, policies

def decode(statesfile, value_policy_file):
    values, policies = readValuePolicy(value_policy_file)
    num_balls, num_runs, states_list, state_to_index_map, index_to_state_map, end_states = readStates(statesfile)
    num_states = len(states_list)
    index_to_choice_map = {0: 0, 1: 1, 2: 2, 3: 4, 4: 6}

    for state in range(num_states - 3):
        bb, rr, player = index_to_state_map[state]
        
        if player == 'A':
            if bb < 10:
                bb = '0' + str(bb)
            else:
                bb = str(bb)

            if rr < 10:
                rr = '0' + str(rr)
            else:
                rr = str(rr)
            
            print(bb + rr + ' ' + str(index_to_choice_map[policies[state]]) + ' ' + str.format('{0:.6f}', values[state]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--value-policy', type=str)
    parser.add_argument('--states', type=str)

    args = parser.parse_args()
    value_policy_file, statesfile = args.value_policy, args.states

    decode(statesfile, value_policy_file)