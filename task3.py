"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the AlgorithmManyArms class. Here are the method details:
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
"""

from xml.dom import InvalidAccessErr
import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class AlgorithmManyArms:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        # Horizon is same as number of arms
        # START EDITING HERE
        # You can add any other variables you need here
        self.recent = False
        self.recent_two = False
        self.sampled_arm = -1
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        # print(self.num_arms)
        if self.recent or self.recent_two:
            return self.sampled_arm
        else:
            return np.random.randint(0, self.num_arms)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        # print(reward)
        if reward == 1:
            self.recent = True
            self.recent_two = True
            self.sampled_arm = arm_index
        else:
            self.recent_two = self.recent
            self.recent = False
        # END EDITING HERE
