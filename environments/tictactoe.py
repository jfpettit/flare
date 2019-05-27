import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class tictactoe:
    def __init__(self):
        self.x_win = False
        self.o_win = False
        self.game_over = False
        self.v = dict()
        self.chars = [0, 1, 2]
        
    def gen_state_arr(self):
        arr = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for t in range(3):
                        for u in range(3):
                            for s in range(3):
                                for q in range(3):
                                    for y in range(3):
                                        for z in range(3):
                                            tmp = np.array([self.chars[i], self.chars[j], self.chars[k], self.chars[t], self.chars[u], self.chars[s],
                                                       self.chars[q], self.chars[y], self.chars[z]]).reshape(3, 3)
                                            if self.check_legal(tmp) == True:
                                                arr.append(tmp)
                                                
        return np.asarray(arr)
        
    def gen_x_values(self, state_arr):
        vals = []
        for i in range(len(state_arr)):
            value = self.check_over(state_arr[i])
            vals.append(value)
        vals = np.asarray(vals)
        vals = vals[vals < 3]
        return vals
    
    def gen_o_values(self, x_values):
        return 1 - x_values
    
    def check_legal(self, state):
        x_count, o_count, empty_count = 0, 0, 0
        state_cpy = np.copy(state).ravel()
        for i in range(len(state_cpy)):
            if state_cpy[i] == 1: x_count += 1
            elif state_cpy[i] == 2: o_count += 1
            elif state_cpy[i] == 0: empty_count += 1
        
        if x_count == o_count + 1 or x_count == o_count:
            # check for X and O win
            if self.check_over(state) == 3:
                return False
            # check for O win and num Xs = num Os
            if x_count != o_count and self.check_over(state) == 0:
                return False
            # check for X win and num Xs = num Os + 1
            if self.check_over(state) == 1 and x_count != o_count + 1:
                return False
            return True
        return False

        
    def check_over(self, state):
        x_vec = np.ones(3, dtype=int)
        o_vec = np.full(3, 2, dtype=int)
        x_win, o_win = 0, 0
        
        state_diag = np.diag(state).astype(int)
        state_rev_diag = np.fliplr(state).diagonal().astype(int)
        
        if np.array_equal(state_diag, x_vec) or np.array_equal(state_rev_diag, x_vec):
            x_win += 1
        
        if np.array_equal(state_diag, o_vec) or np.array_equal(state_rev_diag, o_vec):
            o_win += 1
        
        cols = [state[:, 0].astype(int), state[:, 1].astype(int), state[:, 2].astype(int)]
        rows = [state[0, :].astype(int), state[1, :].astype(int), state[2, :].astype(int)]
        
        for i in cols:
            if np.array_equal(i, x_vec):
                x_win += 1
            elif np.array_equal(i, o_vec):
                o_win += 1
        
        for i in rows:
            if np.array_equal(i, x_vec):
                x_win += 1
            elif np.array_equal(i, o_vec):
                o_win += 1
        
        if x_win >= 1 and o_win == 1:
            return 3
        elif x_win >= 1:
            return 1
        elif o_win == 1:
            return 0
        else:
            return 0.5
    
    def human_viz(self, state):
        tmp_board = np.empty(state.shape, dtype=str)
        for i in range(len(state)):
            for j in range(len(state[i])):
                if state[i][j] == 1:
                    tmp_board[i][j] = 'X'
                elif state[i][j] == 2:
                    tmp_board[i][j] = 'O'
                elif state[i][j] == 0:
                    tmp_board[i][j] = '-'
        print(tmp_board)
        
    def find_next_move_slow(self, state_arr, state, player):
        tmp_states = []
        inds = []
        if player is 1:
            for i in range(len(state)):
                for j in range(len(state[i])):
                    tmp = np.copy(state)
                    if tmp[i][j] == 0:
                        tmp[i][j] = 1
                        tmp_states.append(tmp)
            
            for i in range(len(tmp_states)):
                for j in range(len(state_arr)):
                    if np.array_equal(tmp_states[i], state_arr[j]):
                        inds.append(j)
            return inds
        
        if player is 2:
            for i in range(len(state)):
                for j in range(len(state[i])):
                    tmp = np.copy(state)
                    if tmp[i][j] == 0:
                        tmp[i][j] = 2
                        tmp_states.append(tmp)
                        
            for i in range(len(tmp_states)):
                for j in range(len(state_arr)):
                    if np.array_equal(tmp_states[i], state_arr[j]):
                        inds.append(j)
            return inds
    
    def find_next_move(self, state_arr, state, player):
        tmp_states = []
        inds = []
        sort_inds = np.asarray(np.where(state == 0))
        if player is 1:
            for i in range(sort_inds.shape[1]):
                tmp = np.copy(state)
                tmp_ind = sort_inds[:, i]
                tmp[tmp_ind[0], tmp_ind[1]] = 1
                inds.append(np.argmax((tmp == state_arr).sum(axis=1).sum(axis=1)))
            return inds
        
        if player is 2:
            for i in range(sort_inds.shape[1]):
                tmp = np.copy(state)
                tmp_ind = sort_inds[:, i]
                tmp[tmp_ind[0], tmp_ind[1]] = 2
                inds.append(np.argmax((tmp == state_arr).sum(axis=1).sum(axis=1)))
            return inds
        
    def get_current_state_index(self, state, state_arr):
        for i in range(len(state_arr)):
            if np.array_equal(state_arr[i], state):
                return state, i
        return 'Current state invalid.'
    
    def human_play(self, state_arr, agent, agent_char):
        state = state_arr[0]
        iter_count = 0
        if agent_char is 1:
            while self.check_over(state) == 0.5 and iter_count <= 8:
                cur_state_ind = self.get_current_state_index(state, state_arr)
                if iter_count % 2 == 0:
                    move_inds = self.find_next_move(state_arr, state, 1)
                    new_move_ind = agent.exploit_learned_probs(move_inds, cur_state_ind[1], .1)
                    state = state_arr[new_move_ind]
                else:
                    self.human_viz(state)
                    move_valid = False
                    while move_valid == False:
                        human_move = input('Input your move coordinates, separated by a comma: ')
                        coord_1, coord_2 = int(human_move[0]), int(human_move[-1])
                        if state[coord_1, coord_2] != 0:
                            print('Invalid move, try again.')
                            continue
                        tmp_state = np.copy(state)
                        tmp_state[coord_1, coord_2] = 2
                        if self.check_legal(tmp_state) == True:
                            state = tmp_state
                            move_valid = True
                iter_count += 1

        if agent_char is 2:
            while self.check_over(state) == 0.5 and iter_count <= 8:
                cur_state_ind = self.get_current_state_index(state, state_arr)
                if iter_count % 2 != 0:
                    move_inds = self.find_next_move(state_arr, state, 2)
                    new_move_ind = agent.exploit_learned_probs(move_inds, cur_state_ind[1], .1)
                    state = state_arr[new_move_ind]
                else:
                    self.human_viz(state)
                    move_valid = False
                    while move_valid == False:
                        human_move = input('Input your move coordinates, separated by a comma: ')
                        coord_1, coord_2 = int(human_move[0]), int(human_move[-1])
                        if state[coord_1, coord_2] != 0:
                            print('Invalid move, try again.')
                            continue
                        tmp_state = np.copy(state)
                        tmp_state[coord_1, coord_2] = 1
                        if self.check_legal(tmp_state) == True:
                            state = tmp_state
                            move_valid = True
                iter_count += 1

        self.human_viz(state)
