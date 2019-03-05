import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class td_agent:
    def __init__(self, values, p_random_move):
        self.vals = values
        self.p_random_move = p_random_move
                
    def get_move_update_values(self, move_inds, current_state_ind, alpha):
        if np.random.binomial(1, self.p_random_move) == 1:
            move_ind = np.random.randint(0, len(move_inds))
            move = move_inds[move_ind]
        else:
            greedy_move = np.argmax(self.vals[move_inds])
            move = move_inds[greedy_move]
            self.vals[current_state_ind] += alpha * (self.vals[move] - self.vals[current_state_ind])
        
        return move
    
    def exploit_learned_probs(self, move_inds, current_state_ind, alpha, learn=False):
        move = move_inds[np.argmax(self.vals[move_inds])]
        if learn == True:
            self.vals[current_state_ind] += alpha * (self.vals[move] - self.vals[current_state_ind])
        return move
    
    