import numpy as np
import random
from scipy.spatial.distance import euclidean

class Snake:
	DIRS = [np.array([-1,0]), np.array([0,1]), np.array([1,0]), np.array([0,-1])]
	def __init__(self, identification, start_position, start_position_idx, start_len):
		self.current_position_idx = start_position_idx
		self.snake_id = identification
		self.is_alive = True
		self.body = [start_position]
		current_loc = np.array(start_position)
		for i in range(1, start_len):
			current_loc = current_loc - self.DIRS[start_position_idx]
			self.body.append(tuple(current_loc))

	def grow(self, loc):
		self.body.append(tuple(loc))

	def check_action_ok(self, action):
		return (action != self.current_position_idx) and (action != (self.current_position_idx+2)%len(self.DIRS))

	def step(self, action):
		if self.check_action_ok(action):
			self.current_direction_idx = action

		tail = self.body[-1]
		self.body = self.body[:-1]

		updated_head = np.array(self.body[0] + self.DIRS[self.current_position_idx])

		self.body = [tuple(updated_head)] + self.body

		return tuple(updated_head), tail

class Arena:
	def __init__(self, size=(15, 15), num_snakes=1, num_foods=1, num_obstacles=None, walls=False, vector=True, rew_func=None):
		self.die_reward = -1.
		self.move_reward = 0.
		self.eat_reward = 1.
		if rew_func is not None:
			self.die_reward = rew_func[0]
			self.move_reward = rew_func[1]
			self.eat_reward = rew_func[2]

		self.food = 64
		self.wall = 255
		self.vector = vector

		self.size = size
		self.arena = np.zeros(self.size)

		if walls:
			self.arena[0, :] = self.wall
			self.arena[:, 0] = self.wall
			self.arena[-1, :] = self.wall
			self.arena[:, -1] = self.wall

		self.available_posns = set(zip(*np.where(self.arena == 0)))

		self.snakes = []
		self.food_locs = []
		for _ in range(num_snakes):
			snake = self.register_snake()

		self.put_food(num_foods = num_foods)

	def register_snake(self):
		snakesize = 3
		position = (random.randint(snakesize, self.size[0] - snakesize), random.randint(snakesize, self.size[1] - snakesize))
		while position in self.snakes:
			position = (random.randint(snakesize, self.size[0] - snakesize), random.randint(snakesize, self.size[1] - snakesize))

		start_direction_idx = random.randrange((len(Snake.DIRS)))

		new_snake = Snake(100 + 2 * len(self.snakes), position, start_direction_idx, snakesize)
		self.snakes.append(new_snake)
		return new_snake

	def get_alive_snakes(self):
		return [snake for snake in self.snakes if snake.is_alive]

	def put_food(self, num_foods):
		available_positions = self.available_posns

		for snake in self.get_alive_snakes():
			available_positions -= set(snake.body)

		for _ in range(num_foods):
			pos_choice = random.choice(list(available_positions))
			self.arena[pos_choice[0], pos_choice[1]] = self.food
			self.food_locs.append(pos_choice)
			available_positions.remove(pos_choice)

	def render_obs(self):
		obs = self.arena.copy()

		for snake in self.get_alive_snakes():
			for part in snake.body:
				obs[part[0], part[1]] = snake.snake_id
			obs[snake.body[0][0], snake.body[0][1]] = snake.snake_id + 1
		return obs

	def get_obs(self):
		obs = []
		if self.vector:
			for snake in self.get_alive_snakes():
				head_loc = snake.body[0]
				length = len(snake.body)
				dist_from_food = np.array([euclidean(head_loc, i) for i in self.food_locs]).mean()
				obs.append((head_loc, length, dist_from_food))
			return obs
		else:
			return self.render_obs()

	def check_in_bounds(self, loc):
		return (0 <= loc[0] < self.size[0]) or not(0 <= loc[1] < self.size[1]) or self.arena[loc[0], loc[1]] == self.wall

	def move_snake(self, actions):
		rewards = [0] * len(self.snakes)
		dones = []
		new_food_needed = 0
		for i, (snake, action) in enumerate(zip(self.snakes, actions)):
			if not snake.is_alive:
				continue
			new_snake_head, old_snake_tail = snake.step(action)
			if not self.check_in_bounds(new_snake_head):
				snake.body = snake.body[1:]
				snake.is_alive = False
			elif new_snake_head in snake.body[1:]:
				snake.is_alive = False
			for j, other_snake in enumerate(self.snakes):
				print('new', new_snake_head, 'other', other_snake.body[0])
				if i != j and other_snake.is_alive:
					if new_snake_head == other_snake.body[0]:
						snake.is_alive = False
						other_snake.is_alive = False
					elif new_snake_head in other_snake.body[1:]:
						snake.is_alive = False
			if snake.is_alive and self.arena[new_snake_head[0], new_snake_head[1]] == self.food:
				self.arena[new_snake_head[0], new_snake_head[1]] = 0

				snake.body.append(old_snake_tail)

				new_food_needed += 1
				rewards[i] = self.eat_reward

			elif snake.is_alive:
				rewards[i] = self.move_reward

		dones = [not snake.is_alive for snake in self.snakes]
		rewards = [r if snake.is_alive else self.die_reward for r, snake in zip(rewards, self.snakes)]

		if new_food_needed > 0:
			self.put_food(num_foods=new_food_needed)
		return rewards, dones






