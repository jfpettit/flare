#from rlpack.algorithms.policy_gradients import simple_policy_net, VanillaPolicyGradient
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import sys
import numpy as np
import gym

def mlp(outsize):
    model = keras.Sequential()
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(4, activation='softmax'))
    return model

def valnet():
    model = keras.Sequential()
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(1, activation=None))
    return model

def advantage(obs_R, value):
    return obs_R - value

if __name__ == '__main__':
	#env = gym.make('CartPole-v0')
	#pol = simple_policy_net(4, 2)
	#val = simple_policy_net(4, 1)

	#vpg = VanillaPolicyGradient(env, pol, val)

	#vpg.train(True, 100)
    env = gym.make('gym_snakerl:BasicSnake-vector-16-v0')

    x = np.random.rand(4,1)
    print('X:', x)

    perceptron = mlp(4)
    tensor = perceptron(x)
    sess = tf.compat.v1.Session()
    with sess.as_default():
        tensor = perceptron(x)
        print_op = tf.print("tensors:", tensor, {2: tensor * 2},
                            output_stream=sys.stdout)
        with tf.control_dependencies([print_op]):
          tripled_tensor = tensor * 1
        sess.run(tripled_tensor)





