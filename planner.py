import numpy as np
import pulp as pulp
import math
import sys

np.random.seed(0)

def display(V, policy):
    for v, p in zip(V, policy):
        print(str.format('{0:.6f}', v), ' ', p)

def readPolicy(path):
    file = open(path, 'r')
    policy_string = file.read()
    file.close()

    policy = []
    sentences = policy_string.split('\n')
    
    for sentence in sentences:
        if sentence == '':
            break

        words = sentence.split(' ')
        policy.append(int(words[-1]))
    
    return np.array(policy)

def readMDP(path):

    file = open(path, 'r')
    mdp_string = file.read()
    file.close()

    def readStates():
        nonlocal mdp_string
        space = mdp_string.index(' ')
        newline = mdp_string.index('\n')

        num_states = int(mdp_string[space+1:newline])
        mdp_string = mdp_string[newline+1:]

        return num_states
    
    def readActions():
        nonlocal mdp_string
        space = mdp_string.index(' ')
        newline = mdp_string.index('\n')

        num_actions = int(mdp_string[space+1:newline])
        mdp_string = mdp_string[newline+1:]

        return num_actions
    
    def readEnd():
        nonlocal mdp_string
        space = mdp_string.index(' ')
        newline = mdp_string.index('\n')

        end = mdp_string[space+1:newline].split()
        end = [int(end[i]) for i in range(len(end))]
        mdp_string = mdp_string[newline+1:]

        return end
    
    def readTransitions_Type_Discount(num_states, num_actions):
        nonlocal mdp_string
        sentences = mdp_string.split('\n')
        length = len(sentences)
        
        transition_matrix = np.zeros((num_actions, num_states, num_states))
        reward_matrix = np.zeros((num_actions, num_states, num_states))

        for i, sentence in enumerate(sentences):
            words = sentence.split()

            if i == length - 3:
                mdp_type = words[-1]
                continue
            elif i == length - 2:
                gamma = float(words[-1])
                break
            
            state1, action, state2, reward, prob = int(words[1]), int(words[2]), int(words[3]), float(words[4]), float(words[5])
            transition_matrix[action][state1][state2] = prob
            reward_matrix[action][state1][state2] = reward
        
        return transition_matrix, reward_matrix, mdp_type, gamma

    num_states = readStates()
    num_actions = readActions()
    end_states = np.array(readEnd())
    transition_matrix, reward_matrix, mdp_type, gamma = readTransitions_Type_Discount(num_states, num_actions)

    return num_states, num_actions, end_states, transition_matrix, reward_matrix, mdp_type, gamma

def ValueIteration(path):
    num_states, num_actions, end_states, transition_matrix, reward_matrix, mdp_type, gamma = readMDP(path)

    V_current = np.reshape(np.random.rand(num_states), (-1, 1))
    policy = np.zeros(num_states, dtype=int)

    while True:
        intermediate_matrix_1 = np.sum(transition_matrix * reward_matrix, axis=2)
        intermediate_matrix_2 = gamma * np.matmul(transition_matrix, V_current)
        mat_shape = np.shape(intermediate_matrix_2)
        intermediate_matrix_2 = np.reshape(intermediate_matrix_2, (mat_shape[0], mat_shape[1]))

        final_matrix = intermediate_matrix_1 + intermediate_matrix_2
        V_next = np.max(final_matrix, axis=0)
        
        if np.max(np.abs(V_next - V_current)) <= 0.00000000001:
            policy = np.argmax(final_matrix, axis=0)
            return V_next, policy
        else:
            V_current = np.array(V_next)

def createNewPolicy(policy):
    choice_to_index_map = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4}

    new_policy = []
    for opt_action in policy:
        new_policy.append(choice_to_index_map[opt_action])
        new_policy.append(0)
    
    for i in range(3):
        new_policy.append(0)
    
    return new_policy

def EvaluatePolicy(mdp_path, policy_path, policy=None, num_states=None, num_actions=None, end_states=None, transition_matrix=None, reward_matrix=None, gamma=None):
    mode = 'eval_policy' if policy_path is not None else 'eval_policy_hpi'

    if mode == 'eval_policy':
        num_states, num_actions, end_states, transition_matrix, reward_matrix, mdp_type, gamma = readMDP(mdp_path)
        policy = readPolicy(policy_path)

        if num_states != len(policy):
            policy = createNewPolicy(policy)
    
    V_current, V_next = np.reshape(np.zeros(num_states), (-1, 1)), np.zeros(num_states)

    while True:
        intermediate_matrix_1 = np.sum(transition_matrix * reward_matrix, axis=2)
        intermediate_matrix_2 = gamma * np.matmul(transition_matrix, V_current)
        mat_shape = np.shape(intermediate_matrix_2)
        intermediate_matrix_2 = np.reshape(intermediate_matrix_2, (mat_shape[0], mat_shape[1]))

        final_matrix = intermediate_matrix_1 + intermediate_matrix_2
        
        for state in range(num_states):
            V_next[state] = final_matrix[policy[state]][state]
        
        if np.max(np.abs(V_next - V_current)) <= 0.0000000000000001:
            if mode == 'eval_policy':
                return V_next, policy
            else:
                return V_next
        else:
            V_current = np.array(V_next)

def HowardsPolicyIteration(path):
    num_states, num_actions, end_states, transition_matrix, reward_matrix, mdp_type, gamma = readMDP(path)
    policy = np.random.randint(0, num_actions, num_states)
    valuePolicy = EvaluatePolicy(None, None, policy, num_states=num_states, num_actions=num_actions, end_states=end_states, transition_matrix=transition_matrix, reward_matrix=reward_matrix, gamma=gamma)

    def Iterate():
        policyInternal = np.array(policy)
        valuePolicyInternal = np.array(valuePolicy)
        policy_next = np.array(policyInternal)

        while True:
            num_improving_actions = 0

            for state in range(num_states):
                improving_actions = []

                for action in range(num_actions):
                    if action == policyInternal[state]:
                        continue

                    valueFunction = 0

                    for next_state in range(num_states):
                        valueFunction += transition_matrix[action][state][next_state] * (reward_matrix[action][state][next_state] + gamma * valuePolicyInternal[next_state])
                    
                    if valueFunction > valuePolicyInternal[state]:
                        improving_actions.append(action)
                        num_improving_actions += 1
                    
                
                if len(improving_actions) != 0:
                    policy_next[state] = np.random.choice(improving_actions)
            
            if num_improving_actions == 0:
                return policy_next
            else:
                policyInternal = np.array(policy_next)
                valuePolicyInternal = EvaluatePolicy(None, None, policyInternal, num_states=num_states, num_actions=num_actions, end_states=end_states, transition_matrix=transition_matrix, reward_matrix=reward_matrix, gamma=gamma)
    
    finalPolicy = Iterate()
    
    finalValue = EvaluatePolicy(None, None, finalPolicy, num_states=num_states, num_actions=num_actions, end_states=end_states, transition_matrix=transition_matrix, reward_matrix=reward_matrix, gamma=gamma)
    return finalValue, finalPolicy

def LinearProgramming(path):
    num_states, num_actions, end_states, transition_matrix, reward_matrix, mdp_type, gamma = readMDP(path)
    
    def defineAndSolveLPP():
        puzzle = pulp.LpProblem('MDP_Solution', pulp.LpMinimize)
        
        lp_variables = []
        for state in range(num_states):
            variable = str('V' + str(state))
            lp_variable = pulp.LpVariable(variable)
            lp_variables.append(lp_variable)
        
        puzzle += pulp.lpSum(lp_variables)

        variable = 'constant'
        lp_variable = pulp.LpVariable(variable, 1.0, 1.0)
        lp_variables.append(lp_variable)

        for state in range(num_states):
            for action in range(num_actions):
                expression_list, constant = [], 0
                
                for next_state in range(num_states):
                    constant += transition_matrix[action][state][next_state] * reward_matrix[action][state][next_state]
                    if next_state == state:
                        expression_list.append((lp_variables[next_state], gamma * transition_matrix[action][state][next_state] - 1))
                    else:
                        expression_list.append((lp_variables[next_state], gamma * transition_matrix[action][state][next_state]))
                
                expression_list.append((lp_variables[-1], constant))
                
                e = pulp.LpAffineExpression(expression_list)
                c = pulp.LpConstraint(e, -1, rhs=0)
                puzzle += c
        
        output = puzzle.solve(pulp.PULP_CBC_CMD(msg = 0))
        
        valueFunction = np.zeros(len(puzzle.variables()) - 1)

        for v in puzzle.variables():
            if v.name == 'constant':
                continue

            index = int(v.name[1:])
            valueFunction[index] = v.varValue

        optimalPolicy = np.zeros(num_states, dtype=int)

        for state in range(num_states):
            for action in range(num_actions):
                value = 0

                for next_state in range(num_states):
                    value += transition_matrix[action][state][next_state] * (reward_matrix[action][state][next_state] + gamma * valueFunction[next_state])
                
                if abs(value - valueFunction[state]) <= 10**-5:
                    optimalPolicy[state] = action
                    break
        
        return valueFunction, optimalPolicy
    
    return defineAndSolveLPP()

if __name__ == '__main__':
    path = sys.argv[2]

    try:
        mode = sys.argv[3]
    except:
        mode = '--algorithm'
    
    if mode == '--algorithm':
        try:
            mode_type = sys.argv[4]
        except:
            mode_type = 'vi'
        
        if mode_type == 'vi':
            V_star, optimal_policy = ValueIteration(path)
            display(V_star, optimal_policy)
        elif mode_type == 'hpi':
            V_star, optimal_policy = HowardsPolicyIteration(path)
            display(V_star, optimal_policy)
        elif mode_type == 'lp':
            V_star, optimal_policy = LinearProgramming(path)
            display(V_star, optimal_policy)
    
    elif mode == '--policy':
        policy_path = sys.argv[4]
        V, policy = EvaluatePolicy(path, policy_path=policy_path)
        display(V, policy)