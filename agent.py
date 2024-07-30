import json
import os
import random
import time
from collections import deque
import numpy as np
import torch

from game import Game
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 32
LR = 0.0001
BLOCK_WIDTH = 40


class Agent:
    def __init__(self, ROWS, COLS):
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(12, 512, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.data_file = 'data.json'
        self.record = 0
        self.avg_last_100_episodes = 0
        self.moves_in_games = deque(maxlen=100)
        self.model_folder_path = './model'
        self.t_step = 0
        self.episodes = 0

        self.ROWS = ROWS
        self.COLS = COLS

    def get_state(self, game):

        x = game.robot_pos[0]
        y = game.robot_pos[1]

        maze = game.maze

        fire_pits = game.fire_pits

        point_left = [(x - 1), y]
        point_right = [x + 1, y]
        point_up = [x, y - 1]
        point_down = [x, y + 1]

        # [wall up, wall right, wall down, wall left,
        #  fire up, fire right, fire down, fire left,
        #  target up, target right, target down, target left]

        state = [
            (point_up[1] < 0 or maze[point_up[0]][point_up[1]] == '1'),
            (point_right[0] >= self.COLS or maze[point_right[0]][point_right[1]] == '1'),
            (point_down[1] >= self.ROWS or maze[point_down[0]][point_down[1]] == '1'),
            (point_left[0] < 0 or maze[point_left[0]][point_left[1]] == '1'),

            (tuple(point_up) in fire_pits),
            (tuple(point_right) in fire_pits),
            (tuple(point_down) in fire_pits),
            (tuple(point_left) in fire_pits),

            (point_up == game.end_pos),
            (point_right == game.end_pos),
            (point_down == game.end_pos),
            (point_left == game.end_pos),

        ]

        return np.array(state, dtype=int)

    def load(self, file_name='model.pth'):

        file_path = os.path.join(self.model_folder_path, file_name)
        if os.path.exists(file_path):
            self.model.load_state_dict(torch.load(file_path))
            print("Model loaded.")

            self.retrieve_data()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_memory(self):
        if len(self.memory) < BATCH_SIZE:
            return
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        for state, action, reward, next_state, done in mini_sample:
            self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: trade off exploration / exploitation
        final_move = [0, 0, 0, 0]  # [up, right, down, left]
        if np.random.rand() <= self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def save_data(self, n_games, record, epsilon, file_name='data.json'):

        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)
        complete_path = os.path.join(self.model_folder_path, file_name)

        data = {'episodes': n_games, 'record': record, 'epsilon': epsilon}
        with open(complete_path, 'w') as file:
            json.dump(data, file, indent=4)

    def retrieve_data(self):
        model_data_path = os.path.join(self.model_folder_path, self.data_file)
        if os.path.exists(model_data_path):
            with open(model_data_path, 'r') as file:
                data = json.load(file)

            if data is not None:
                self.episodes = data['episodes']
                self.record = data['record']
                self.epsilon = data['epsilon']


def train():
    game = Game()
    agent = Agent(game.ROWS, game.COLS)
    agent.load()

    while True:
        time.sleep(game.wait)
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get the new state
        reward, done, move = game.run_step(final_move)
        state_new = agent.get_state(game)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        # train after every 4 steps
        agent.t_step = (agent.t_step + 1) % 4
        if agent.t_step == 0:
            agent.train_memory()

        if done:
            agent.episodes += 1
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            if move > agent.record:
                agent.record = move

            agent.model.save()

            game.message = "Episodes: " + str(agent.episodes)

            game.record = agent.record
            agent.save_data(agent.episodes, agent.record, agent.epsilon)
            game.reset()

if __name__ == '__main__':
    train()
