import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class blackjack_game(gym.Env):
    def __init__(self):
        self.cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
        self.action_space = spaces.Discrete(2)
        self.obs_space = spaces.Tuple((
                        spaces.Discrete(32),
                        spaces.Discrete(11),
                        spaces.Discrete(2)))
        self._seed()
        self._reset()
        self.nA = 2
    
    def ace_is_usable(self, cards):
        if 1 in cards:
            if sum(cards) + 10 <= 21:
                return 1
        else:
            return 0
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _step(self, action):
        if action:
            self.player_cards.append(self.draw_card())
            if self.went_bust(self.player_cards):
                episode_over = True
                reward = -1
            else:
                episode_over = False
                reward = 0
        else:
            episode_over = True
            while self.player_total(self.dealer_cards) < 17:
                self.dealer_cards.append(self.draw_card())
            reward = int(self.hand_reward(self.player_cards) > self.hand_reward(self.dealer_cards))-int(self.hand_reward(self.player_cards)<self.hand_reward(self.dealer_cards))
        return self.observations(), reward, episode_over, {}
    
    def observations(self):
        return(self.player_total(self.player_cards), self.dealer_cards[0], self.ace_is_usable(self.player_cards))
    
    def _reset(self):
        self.player_cards = self.get_hand()
        self.dealer_cards = self.get_hand()
        
        while self.player_total(self.player_cards) < 12:
            self.player_cards.append(self.draw_card())
        
        return self.observations()
        
    def get_hand(self):
        return [self.draw_card(), self.draw_card()]
      
    def hand_reward(self, player_cards):
        if self.went_bust(player_cards) is 1:
            return 0
        else:
            return self.player_total(player_cards)
    
    def went_bust(self, player_cards):
        return self.player_total(player_cards) > 21
    
    def player_total(self, player_cards):
        if self.ace_is_usable(player_cards):
            return np.sum(player_cards) + 10
        else:
            return np.sum(player_cards)
    
    def draw_card(self):
        return self.np_random.choice(self.cards)
    
    def render(self, player_action):
        arr = np.array(['Player points: ', self.player_total(self.player_cards), 'Player action: ',
                        player_action, 'Dealer points: ', self.player_total(self.dealer_cards), 'Dealer action:',
                        int(self.player_total(self.dealer_cards) < 17)])
        arr = arr.reshape((2,4))
        return arr