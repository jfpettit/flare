import numpy as np
import gym
from gym.spaces import Discrete, Box
from queue import deque
from scipy.spatial.distance import euclidean

class Arena:
	def __init__(self, arena_size=(15, 15), headcolor=250, bodycolor=150, foodcolor=200, num_food=1):
		self.arena_size = arena_size
		self.arena = np.zeros(tuple(self.arena_size))

		self.headcolor = headcolor
		self.bodycolor = bodycolor
		self.foodcolor = foodcolor

		self.open_area = self.arena_size[0]*self.arena_size[1]
		self.num_food = num_food
		self.food_locs = []

	def check_off_grid(self, loc):
		return loc[0] < 0 or loc[0] >= self.arena_size[0] or loc[1] < 0 or loc[1] >= self.arena_size[1]	

	def check_snake_space(self, loc):
		val = self.arena[loc[0], loc[1]]
		return val == self.bodycolor 

	def check_snake_dead(self, loc):
		return self.check_off_grid(loc) or self.check_snake_space(loc)

	def erase_point(self, loc):
		if self.check_off_grid(loc):
			return False
		self.arena[loc[0], loc[1]] = 0
		self.open_area = self.open_area + 1

	def check_food_space(self, loc):
		return self.arena[loc[0], loc[1]] == self.foodcolor

	def connect_points(self, loc1, loc2):
		try:
			assert (np.abs(loc1[0]-loc2[0]) == 1 and np.abs(loc1[1]-loc2[1]) == 0)
		except:
			assert (np.abs(loc1[0]-loc2[0]) == 0 and np.abs(loc1[1]-loc2[1]) == 1)

		self.arena[loc2[0], loc2[1]] = self.arena[loc1[0], loc1[1]]

	def draw_point(self, loc, color):
		self.arena[loc[0], loc[1]] = color
		self.open_area -= 1

	def draw_snake(self, snake_head, snake_body):
		self.arena[snake_head[0], snake_head[1]] = self.headcolor
		for i in range(1, len(snake_body)):
			bdy = snake_body[i]
			self.arena[bdy[0], bdy[1]] = self.bodycolor

	def erase_snake(self, snake_head, snake_body):
		self.arena[snake_head[0], snake_head[1]] = 0
		for i in range(1, len(snake_body)):
			self.arena[snake_body[i][0], snake_body[i][1]] = 0

	def gen_food(self):
		if self.open_area < 1:
			return False
		food = np.random.randint(0, self.arena_size[0], 2)
		while self.arena[food[0], food[1]] != 0:
			food = np.random.randint(0, self.arena_size[0], 2)
		self.food_locs.append(food)
		self.arena[food[0], food[1]] = self.foodcolor
		self.open_area -= 1
		return True



class Snake:
	def __init__(self, snake_head, snake_size=2):
		'''
		Arena is M*N array
		snake head will be represented with value 250, body with value 150
		background will be 0
		food will be 200

		snake_size represents total length of the snake
		snake_head is location of the snake head

		headcolor and bodycolor are for rendering, what color to draw those things as

		arena is the grid for the game

		snake_body is deque object which will contain coordinates of the snakes body
		'''
		self.snake_size = snake_size
		self.snake_head = snake_head
		self.snake_body = deque()
		for i in range(snake_size-1, 0, -1):
			self.snake_body.append(np.asarray([self.snake_head[0]-i, i]).astype(np.int))
		
		self.move = np.random.randint(0, 4)

		self.UP = 0
		self.RIGHT = 1
		self.DOWN = 2
		self.LEFT = 3


	def grow_snake(self, loc1, loc2):
		self.snake_size += 1
		self.snake_body.append(loc2)

	def step_snake(self, loc, move):
		assert move in list(range(4))
		if move == self.UP:
			return np.array([loc[0]+1, loc[1]], dtype=np.int)
		elif move == self.RIGHT:
			return np.array([loc[0], loc[1]+1], dtype=np.int)
		elif move == self.DOWN:
			return np.array([loc[0]-1, loc[1]], dtype=np.int)
		elif move == self.LEFT:
			return np.array([loc[0], loc[1]-1], dtype=np.int)

	def snake_action(self, move):
		if np.abs(self.move - move) != 2:
			self.move = move

		self.snake_body.append(self.snake_head)
		self.snake_head = self.step_snake(self.snake_head, self.move)
		return self.snake_head

class GameControl:
	def __init__(self, arena_size=(15, 15), snake_size=2, num_snakes=1, snake_heads=None, num_food=1, headcolor=250, bodycolor=150, 
			foodcolor=200):
		self.arena_size = arena_size
		self.headcolor = headcolor
		self.bodycolor = bodycolor
		self.foodcolor = foodcolor
		self.num_food = num_food

		self.arena = Arena(arena_size=arena_size, headcolor=headcolor, bodycolor=bodycolor, foodcolor=foodcolor, num_food=num_food)

		self.snake_size = snake_size
		self.num_snakes = num_snakes
		self.dead_snakes = []
		assert type(snake_heads) is list or type(snake_heads) is tuple or type(snake_heads) is np.ndarray
		if snake_heads is not None:
			self.snake_heads = snake_heads
		else:
			self.snake_heads = []
			for i in range(num_snakes):
				arr = np.random.randint(self.arena_size[0]-1, self.arena_size[1]-1, 2)
				while arr in self.snake_heads:
					arr = np.random.randint(self.arena_size[0]-1, self.arena_size[1]-1, 2)
				self.snake_heads.append(arr)
		
		self.snakes = []
		for i in range(num_snakes):
			self.snakes.append(Snake(self.snake_heads[i], self.snake_size))
			self.arena.draw_snake(self.snakes[-1].snake_head, self.snakes[-1].snake_body)
			self.dead_snakes.append(None)

		for i in range(self.num_food):
			self.arena.gen_food()

		self.DIRS = [np.array([-1,0]), np.array([0,1]), np.array([1,0]), np.array([0,-1])]

	def move_snake(self, direction, snake_idx):
		snake = self.snakes[snake_idx]
		if type(snake) == type(None):
			return

		self.arena.arena[snake.snake_head[0], snake.snake_head[1]] = self.bodycolor
		self.arena.erase_point(snake.snake_body[0])
		snake.snake_action(direction)

	def move_result(self, direction, snake_idx=0):
		snake = self.snakes[snake_idx]

		if type(snake) == type(None):
			return

		if self.arena.check_snake_dead(snake.snake_head):
			self.dead_snakes[snake_idx] = snake
			self.snakes[snake_idx] = None
			self.arena.arena[snake.snake_head[0], snake.snake_head[1]] = self.headcolor
			self.arena.connect_points(snake.snake_body.popleft(), snake.snake_body[0])
			reward = -1
		elif self.arena.check_food_space(snake.snake_head):
			self.arena.arena[snake.snake_body[0], snake.snake_body[1]] = self.bodycolor
			self.arena.connect_points(snake.snake_body[0], snake.snake_body[1])
			self.arena.arena[snake.snake_head] = self.headcolor
			reward = 1
			self.arena.gen_food()
			self.snake.grow_snake(tuple(np.array(snake.snake_body[0])) + self.DIRS[direction])
		else:
			# reward being average distance from all foods may not produce desirable behavior, need to experimentally validate that
			# doing this yields a good signal
			reward = np.asarray([euclidean(snake.snake_head, self.arena.food_locs[i]) for i in range(len(self.arena.food_locs))]).mean()/self.arena_size[0]
			print(snake.snake_body)
			empt = snake.snake_body.popleft()
			self.arena.connect_points(empt, snake.snake_body[0])
			self.arena.arena[snake.snake_head[0], snake.snake_head[1]] = self.headcolor

		self.arena.connect_points(snake.snake_body[-1], snake.snake_head)

		return reward

	def kill_snake(self, snake_idx):
		assert self.dead_snakes[snake_idx] is not None
		self.arena.erase_point(self.dead_snakes[snake_idx].snake_head)
		self.arena.erase_snake(self.dead_snakes[snake_idx].snake_head, self.dead_snakes[snake_idx].snake_body)
		self.dead_snakes[snake_idx] = None
		self.num_snakes -= 1

	def get_dist_from_food(self, snake_loc, food_locs):
		dists = []
		for i in range(len(food_locs)):
		    dists.append(euclidean(np.asarray(snake_loc), np.asarray(i)))
		return dists

	def get_obs(self, snake):
		head = snake.snake_head
		avg_food_dist = np.asarray(self.get_dist_from_food(head, self.arena.food_locs)).mean()
		return (head, len(snake.snake_body), avg_food_dist)

	def step(self, directions):
		if self.num_snakes < 1 or self.arena.open_area < 1:
			if type(directions) == type(int()) or len(directions) == 1:
				return self.get_obs(self.snakes[0]), 0, True, {"snakes_remaining" : self.num_snakes}
			else:
				obs = [self.get_obs(i) for i in self.snakes]
				return obs, [0]*len(directions), True, {"snakes_remaining", self.num_snakes}

		rewards = []
		if type(directions) == type(int()):
			directions = [directions]
		
		for i, direction in enumerate(directions):
			if self.snakes[i] is None and self.dead_snakes[i] is not None:
				self.kill_snake(i)
			self.move_snake(direction, i)
			rewards.append(self.move_result(direction, i))

		done = self.num_snakes < 1 or self.arena.open_area < 1
		

		if len(rewards) is 1:
			return self.get_obs(self.snakes[0]), rewards[0], done, {"snakes_remaining", self.num_snakes}
		else:
			obs = [self.get_obs(i) for i in self.snakes]
			return obs, rewards, done, {"snakes_remaining", self.num_snakes}
















		