import unittest as unit
import numpy as np 
from snake import Arena, Snake, GameControl

class TestArena(unit.TestCase):
	def test_check_off_grid(self):
		ARENA = Arena(arena_size=(3,3))
		on_grid = (1, 1)
		off_grid = (5, 5)

		on = ARENA.check_off_grid(on_grid)
		off = ARENA.check_off_grid(off_grid)
		self.assertEqual(on, False)
		self.assertEqual(off, True)

	def test_check_snake_space(self):
		ARENA = Arena(arena_size=(3,3))
		ARENA.arena[1, 1] = ARENA.bodycolor
		ARENA.arena[2, 2] = ARENA.headcolor
		spot1 = [0, 0]
		spot2 = [1, 1]
		spot3 = [2, 2]

		one = ARENA.check_snake_space(spot1)
		two = ARENA.check_snake_space(spot2)
		three = ARENA.check_snake_space(spot3)

		self.assertEqual(one, False)
		self.assertEqual(two, True)
		self.assertEqual(three, False)

	def test_check_snake_dead(self):
		ARENA = Arena(arena_size=(3,3))
		ARENA.arena[1, 1] = ARENA.bodycolor
		dead_off = [4, 4]
		dead_touch = [1, 1]
		alive = [0, 2]

		dead_off = ARENA.check_snake_dead(dead_off)
		dead_touch = ARENA.check_snake_dead(dead_touch)
		alive = ARENA.check_snake_dead(alive)

		self.assertEqual(dead_off, True)
		self.assertEqual(dead_touch, True)
		self.assertEqual(alive, False)

	def test_erase_point(self):
		ARENA = Arena(arena_size=(3,3))
		point = [1, 1]
		off = [7, 7]
		ARENA.arena[point[0], point[1]] = ARENA.headcolor
		ARENA.open_area -=1

		off_r = ARENA.erase_point(off)
		ARENA.erase_point(point)
		self.assertEqual(off_r, False)
		self.assertEqual(ARENA.arena[1, 1], 0)
		self.assertEqual(ARENA.open_area, 3*3)

	def test_draw_point(self):
		ARENA = Arena(arena_size=(3,3))
		point = [1, 1]

		ARENA.draw_point(point, ARENA.headcolor)

		self.assertEqual(ARENA.arena[point[0], point[1]], ARENA.headcolor)
		self.assertEqual(ARENA.open_area, 8)

	def test_check_food_space(self):
		ARENA = Arena(arena_size=(3,3))
		ARENA.arena[1, 1]= ARENA.foodcolor

		true = ARENA.check_food_space([1, 1])
		false = ARENA.check_food_space([0, 0])
		self.assertEqual(True, true)
		self.assertEqual(False, false)

	def test_connect_points(self):
		ARENA = Arena(arena_size=(3,3))
		point = [0, 0]
		adj = [0, 1]
		anotheradj = [1, 0]
		notadj = [1, 1]
		ARENA.arena[point[0], point[1]] = 5

		ARENA.connect_points(point, adj)
		ARENA.connect_points(point, anotheradj)
		#self.assertEqual(AssertionError, ARENA.connect_points(point, notadj))
		self.assertEqual(ARENA.arena[point[0], point[1]], ARENA.arena[adj[0], adj[1]])
		self.assertEqual(ARENA.arena[point[0], point[1]], ARENA.arena[anotheradj[0], anotheradj[1]])

	def test_draw_snake_erase_snake(self):
		ARENA = Arena(arena_size=(3,3))

		snake_head = [0, 0]
		snake_body = ([0, 1], [0, 2])
		snake = (snake_head, snake_body[0], snake_body[1])

		ARENA.draw_snake(snake_head, snake)
		head = ARENA.arena[snake_head[0], snake_head[1]]
		bdy = snake_body[0]
		body1 = ARENA.arena[bdy[0], bdy[1]]

		self.assertEqual(head, ARENA.headcolor)
		self.assertEqual(body1, ARENA.bodycolor)

		ARENA.erase_snake(snake_head, snake)
		self.assertEqual(ARENA.arena[snake_head[0], snake_head[1]], 0)
		self.assertEqual(ARENA.arena[bdy[0], bdy[1]], 0)

	def test_gen_food(self):
		ARENA = Arena(arena_size=(3,3))
		ARENA.open_area = 0
		self.assertEqual(ARENA.gen_food(), False)

		ARENA.open_area = 9
		self.assertEqual(ARENA.gen_food(), True)


class TestSnake(unit.TestCase):
	def test_grow_snake(self):
		SNAKE = Snake([0, 1])
		curr_head = SNAKE.snake_head
		self.assertEqual(curr_head, [0, 0])

		add = [0, 2]









if __name__ == '__main__':
	unit.main()