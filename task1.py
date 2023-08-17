"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need
def klCalc(p_hat, q, constant):
    if p_hat == 1:
        kl = math.log(1 / q)
    elif p_hat == 0:
        kl = math.log(1 / (1 - q))
    else:
        kl = p_hat * math.log(p_hat / q) + (1 - p_hat) * math.log((1 - p_hat) / (1 - q))
    return kl - constant

def binSearch(p_hat, constant, tolerance):
    lb, ub = p_hat, 1
    mid = (lb + ub) / 2.0

    while abs(ub - lb) > tolerance:
        mid = (lb + ub) / 2.0

        kl_lb = klCalc(p_hat, lb, constant)
        kl_mid = klCalc(p_hat, mid, constant)

        if kl_lb * kl_mid < 0:
            ub = mid
        elif kl_lb * kl_mid > 0:
            lb = mid
        else:
            if kl_lb == 0:
                return lb
            else:
                return mid
    
    return mid
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.timestep = 0
        self.num_arms = num_arms
        self.empirical_means = np.zeros(self.num_arms)
        self.num_pulls_each_arm = np.ones(self.num_arms)
        self.exploration_bonus = np.reshape(np.full((self.num_arms, 1), math.sqrt(2 * math.log(self.num_arms) / 1.0)), -1)
        self.ucb = np.zeros(self.num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if self.timestep < self.num_arms:
            return self.timestep
        else:
            return np.argmax(self.ucb)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if self.timestep < self.num_arms:
            self.timestep += 1
            self.empirical_means[arm_index] += reward
            self.ucb[arm_index] += self.empirical_means[arm_index] + self.exploration_bonus[arm_index]
        else:
            self.timestep += 1
            n = self.num_pulls_each_arm[arm_index]
            self.num_pulls_each_arm[arm_index] += 1
            mean = self.empirical_means[arm_index]
            self.empirical_means[arm_index] = (n * mean + 1 * reward) / (n + 1)
            self.exploration_bonus = math.sqrt(2 * math.log(self.timestep)) / np.sqrt(self.num_pulls_each_arm)
            self.ucb = self.empirical_means + self.exploration_bonus
        # END EDITING HERE

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.const = 3
        self.timestep = 0
        self.tolerance = 0.005
        self.num_arms = num_arms
        self.num_pulls_each_arm = np.ones(self.num_arms)
        self.empirical_means = np.zeros(self.num_arms)
        self.ucb_kl = np.zeros(self.num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if self.timestep < self.num_arms:
            return self.timestep
        else:
            return np.argmax(self.ucb_kl)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if self.timestep < self.num_arms - 1:
            self.timestep += 1
            self.empirical_means[arm_index] += reward
        elif self.timestep == self.num_arms - 1:
            self.timestep += 1
            constant = math.log(self.timestep) + self.const * math.log(math.log(self.timestep))
            self.empirical_means[arm_index] += reward
            self.ucb_kl = np.array([binSearch(self.empirical_means[i], constant / self.num_pulls_each_arm[i], self.tolerance) for i in range(self.num_arms)])
        else:
            constant = math.log(self.timestep) + self.const * math.log(math.log(self.timestep))
            self.timestep += 1
            n = self.num_pulls_each_arm[arm_index]
            self.num_pulls_each_arm[arm_index] += 1
            mean = self.empirical_means[arm_index]
            self.empirical_means[arm_index] = (n * mean + 1 * reward) / (n + 1)
            self.ucb_kl = np.array([binSearch(self.empirical_means[i], constant / self.num_pulls_each_arm[i], self.tolerance) for i in range(self.num_arms)])
        # END EDITING HERE


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.num_arms = num_arms
        self.num_heads = np.ones(num_arms)
        self.num_tails = np.ones(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        sample_values = np.array([np.random.beta(self.num_heads[i], self.num_tails[i]) for i in range(self.num_arms)])
        return np.argmax(sample_values)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward == 1:
            self.num_heads[arm_index] += 1
        else:
            self.num_tails[arm_index] += 1
        # END EDITING HERE
