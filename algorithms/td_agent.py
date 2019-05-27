import numpy as np

class tdagent:
    def __init__(self, values, p_random_move, alpha_start):
        self.vals = values
        self.p_random_move = p_random_move
        self.alpha = alpha_start
    
    def alpha_anneal(self, alpha_start, alpha_end, num_iters):
        self.alpha -= (alpha_start - alpha_end)/num_iters 
    
    def get_move_update_values(self, move_inds, current_state_ind):
        if np.random.binomial(1, self.p_random_move) == 1:
            move_ind = np.random.randint(0, len(move_inds))
            move = move_inds[move_ind]
        else:
            greedy_move = np.argmax(self.vals[move_inds])
            move = move_inds[greedy_move]
            self.vals[current_state_ind] += self.alpha * (self.vals[move] - self.vals[current_state_ind])
        
        return move
    
    def exploit_learned_probs(self, move_inds, current_state_ind, learn=False):
        move = move_inds[np.argmax(self.vals[move_inds])]
        if learn == True:
            self.vals[current_state_ind] += self.alpha * (self.vals[move] - self.vals[current_state_ind])
        return move
    
    