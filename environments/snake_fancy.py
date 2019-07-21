import numpy as np
import gym
from gym.spaces import Discrete, Box
from queue import deque


class GameArena:
	SNAKEBODYCOLOR = np.array([0, 255, 0], dtype=np.uint8)
	SNAKEHEADCOLOR = np.array([0, 0, 255], dtype=np.uint8)
	BACKGROUNDCOLOR = np.array([1, 0, 0], dtype=np.uint8)
	FOODCOLOR = np.array([255, 0, 0], dtype=np.uint8)

	def __init__(self, arenasize=[15, 15], unitsize=10, unitgap=1):
		self.arena_size = np.asarray(arenasize, dtype=np.int)
		self.unit_size = int(unitsize)
		self.unit_gap = int(unitgap)
		pixel_height = self.arena_size[0] * self.unit_size
		pixel_width = self.arena_size[1] * self.unit_size
		color_channels = 3
		self.arena = np.zeros((pixel_height, pixel_width, color_channels), dtype=np.uint8)
		self.arena[:, :, :] = self.BACKGROUNDCOLOR
		self.arena_area = self.arena_size[0] * self.arena_size[1]

	def check_off_grid(self, snake_coordinates)
		return snake_coordinates[0] < 0 or snake_coordinates[0] >= self.arena_size[0] or snake_coordinates[1] < 0 or snake_coordinates[1] >= self.arena_size[1]

	def coordinate_color(self, snake_coordinates):
		return self.arena[int(snake_coordinates[0]*self.unit_size), int(snake_coordinates[1]*self.unit_size), :]

	def check_if_snake(self, snake_coordinates):
		color = self.coordinate_color(snake_coordinates)
		return np.array_equal(color, self.SNAKEBODYCOLOR) or color[0] == self.SNAKEHEADCOLOR[2]

	def check_snake_dead(self, snake_head_coordinates):
		return self.check_off_grid(snake_head_coordinates) or self.check_if_snake(snake_head_coordinates)

	def connect_coordinates(self, coordinate_1, coordinate_2, color=self.SNAKEBODYCOLOR):
		try:
			assert (np.abs(coordinate_1[0] - coordinate_2[0]) == 1) and (np.abs(coordinate_1[1] - coordinate_2[1]) == 0)
			case1 = True
		except:
			assert (np.abs(coordinate_1[0] - coordinate_2[1]) == 0) and (np.abs(coordinate_1[1] - coordinate_2[1]) == 1)
			case1 = False

		if case1:
			minx, maxx = sorted(coordinate_1[0], coordinate_2[0])
			minx = minx * self.unit_size+self.unit_size - self.unit_gap
			maxx = maxx * self.unit_size
			self.arena[coordinate_1[1]*self.unit_size, minx:maxx, :] = color
			self.arena[coordinate_1[1]*self.unit_size+self.unit_size - self.unit_gap-1, minx:maxx, :] = color

		else:
			miny, maxy = sorted(coordinate_1[1], coordinate_2[1])
			miny = miny * self.unit_size + self.unit_size - self.unit_gap
			maxy = maxy * self.unit_size
			self.arena[miny:maxy, coordinate_1[0] * self.unit_size + self.unit_size - self.unit_gap - 1, :] = color
			self.arena[miny:maxy, coordinate_1[0] * self.unit_size, :] = color

	def cover_coordinate(self, coordinate, color):
		if self.check_off_grid(coordinate):
			return False
		x = int(coordinate[0] * self.unit_size)
		y = int(coordinate[1] * self.unit_size)
		lastx = x + self.unit_size - self.unit_gap
		lasty = y + self.unit_size - self.unit_gap
		self.arena[y:lasty, x:lastx, :] = np.asarray(color, dtype=np.uint8)
		return True

	def draw_coordinate(self, coordinate, color):
		if self.cover_coordinate(coordinate, color):
			self.arena_area -= 1
			return True
		else:
			return False

	def draw_snake(self, snake, head_color=self.SNAKEHEADCOLOR):
		self.draw_coordinate(snake.head, head_color)
		previous_coordinate = None
		for i in range(len(snake.snake_body)):
			body_coordinate = snake.snake_body.popleft()
			self.draw_coordinate(body_coordinate, self.SNAKEBODYCOLOR)
			if previous_coordinate is not None:
				self.connect_coordinates(previous_coordinate, body_coordinate, color=self.SNAKEBODYCOLOR)
			snake.snake_body.append(body_coordinate)
			previous_coordinate = body_coordinate
		self.connect_coordinates(previous_coordinate, snake.head, self.SNAKEBODYCOLOR)

	def erase_coordinate(self, coordinate):
		if self.check_off_grid(coordinate):
			return False
		self.arena_area += 1
		x = int(coordinate[0] * self.unit_size)
		lastx = x + self.unit_size
		y = int(coordinate[1] * self.unit_size)
		lasty = y + self.unit_size
		self.arena[y:lasty, x:lastx, :] = self.BACKGROUNDCOLOR
		return True

	def erase_connections(self, coordinate):
		if self.check_off_grid(coordinate):
			return False
		x = int(coordinate[0]*self.unit_size)
		lastx = x + self.unit_size
		y = int(coordinate[1]*self.unit_size)
		lasty = y + self.unit_size
		self.arena[y:lasty, x:lastx, :] = self.BACKGROUNDCOLOR

		x = int(coordinate[0] * self.unit_size)+self.unit_size-self.unit_gap
		lastx = x + self.unit_gap
		y = int(coordinate_1 * self.unit_size)
		lasty = y+self.unit_size
		self.arena[y:lasty, x:lastx, :] = self.BACKGROUNDCOLOR

		return True

	def erase_snake_body(self, snake):
		for i in range(len(snake.snake_body)):
			self.erase_coordinate(snake.snake_body.popleft())

	def check_if_food(self, coordinate):
		return np.array_equal(self.coordinate_color(coordinate), self.FOODCOLOR)

	def put_food(self, coordinate):
		if self.arena_area < 1 or not np.array_equal(self.coordinate_color(coordinate), self.BACKGROUNDCOLOR):
			return False
		self.draw_coordinate(coordinate, self.FOODCOLOR)
		return True

	def random_put_food(self):
		if self.arena_area < 1:
			return False

		need_space = True
		while need_space:
			space = (np.random.randint(0, self.arena_size[0]), np.random.randint(0, self.arena_size[1]))
			if np.array_equal(self.coordinate_color(space), self.BACKGROUNDCOLOR):
				need_space = False
		self.draw_coordinate(space, self.FOODCOLOR)
		return True

class Snake:
	UP = 0
	RIGHT = 1
	DOWN = 2
	LEFT = 3

	def __init__(self, snake_coordinate_start, length=0):
		self.direction = np.random.randint(0, 4)
		self.snake_head = np.asarray(snake_coordinate_start).astype(np.int)
		self.snake_head_color = np.array([0, 0, 255], np.uint8)
		self.snake_body = deque()
		for i in range(length-1, 0, -1):
			self.snake_body.append(self.snake_head - np.array([0, i]).astype(np.int))

	def step(self, coordinate, direction):
		assert direction < 4 and direction >= 0

		if direction == self.UP:
			return np.asarray([coordinate[0], coordinate[1]-1]).astype(np.int)
		elif direction == self.RIGHT:
			return np.asarray([coordinate[0]+1, coordinate[1]]).astype(np.int)
		elif direction == self.DOWN:
			return np.asarray([coordinate[0], coordinate[1]+1]).astype(np.int)
		else:
			return np.array([coordinate[0]-1, coordinate[1]]).astype(np.int)

	def action(self, direction):
		direction = int(direction) % 4

		if np.abs(self.direction - direction) != 2:
			self.direction = direction

		self.snake_body.append(self.snake_head)
		self.snake_head = self.step(self.snake_head, self.direction)
		return self.snake_head

	class MultiDiscrete:
		def __init__(self, num_actions):
			self.dtype = np.int32
			self.num = num_actions
			self.actions = np.arange(self.num, dtype=self.dtype)
			self.shape = self.actions.shape

		def contains(self, argument):
			for action in self.actions:
				if action == argument:
					return True
			return False

		def sample(self):
			return np.random.choice(self.num)

			


class CoreEnv(gym.Env):
	def __init__(self, numsnakes=1, arenasize=[15, 15], numfoods=1, unitsize=10, unitgap=1):
		super(CoreEnv, self).__init__()
		self.snakes_left = numsnakes
		self.snake_size = 0
		self.num_foods = numfoods
		self.unit_size = unitsize
		self.unit_gap = unitgap



		''' 
		Observation set:
			location of the snakes head
			length of the snake
			distance from food
			if other snakes (?): 
				number of other snakes
				distance from nearest other snake
				length of largest snake

		There are arbitrarily many permutations of the observation space. Changes in observation space should most likely be accompanied by
		a change in reward function. The agent should not be given unnecessary extra information, it is best to figure out what the minimum 
		observation info required to solve the game is, and then use only that.
		'''

		self.observation_space = Box(-np.inf, np.inf, (3,))

	def check_dead(self, loc):
