import numpy as np

class dynamic_programming_policy_iteration:
    def __init__(self, environment, discount, threshold, maxiter):
        self.environment = environment
        self.discount = discount
        self.num_states = environment.nS
        self.num_actions = environment.nA
        self.threshold = threshold
        self.maxiter = maxiter
        
    def evaluation(self, policy):
        v_0 = np.zeros(self.num_states)

        for i in range(self.maxiter):
            DELTA = 0
            for i in range(self.num_states):
                v_ = 0
                for act, actprob in enumerate(policy[i]):
                    for prob, state, r, over in self.environment.P[i][act]:
                        v_ += actprob * prob * (r + self.discount * v_0[state])

                DELTA = max(DELTA, np.abs(v_ - v_0[i]))
                v_0[i] = v_

            if DELTA < self.threshold: 
                break

        return np.array(v_0)
        
    def lookahead(self, current_state, V):
        A = np.zeros(self.num_actions)

        for acts in range(self.num_actions):
            for prob, next_state, r, over in self.environment.P[current_state][acts]:
                A[acts] += prob * (r + self.discount * V[next_state])
        return A
    
    def improvement(self, evaluation_function):
        policy = np.ones([self.num_states, self.num_actions]) / self.num_actions
        
        for i in range(self.maxiter):
            V_func = evaluation_function(policy)
            stable = True
            
            for state in range(self.num_states):
                picked_action = np.argmax(policy[state])
                
                act_vals = self.lookahead(state, V_func)
                optimal_action = np.argmax(act_vals)
                
                if picked_action is not optimal_action:
                    stable = False
                    
                policy[state] = np.eye(self.num_actions)[optimal_action]
            
            if stable:
                return policy, V_func
            
        return policy, V_func
        