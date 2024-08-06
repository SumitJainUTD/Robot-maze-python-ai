import json
import os
import random
import time
from collections import deque
import numpy as np
import torch

from game import Game
from model import Linear_QNet, QTrainer

MAX_MEMORY = 150_000
BATCH_SIZE = 64
LR = 0.001
BLOCK_WIDTH = 40
TARGET_UPDATE_INTERVAL = 4000


class Agent:
    def __init__(self, ROWS, COLS):
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.01
        self.gamma = 0.99  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(12, 256, 4)
        self.target_model = Linear_QNet(12, 256, 4)
        self.trainer = QTrainer(self.model, self.target_model, LR, self.gamma)
        self.data_file = 'data.json'
        self.record = 0
        self.avg_last_100_episodes = 0
        self.moves_in_games = deque(maxlen=100)
        self.model_folder_path = './model'
        self.t_step = 0
        self.episodes = 0
        self.LEARNING_STARTS_IN = 40000
        self.ROWS = ROWS
        self.COLS = COLS

    def get_state(self, game):

        robot_x = game.robot_pos[0]
        robot_y = game.robot_pos[1]

        maze = game.maze

        fire_pits = game.fire_pits

        point_left = [(robot_x - 1), robot_y]
        point_right = [robot_x + 1, robot_y]
        point_up = [robot_x, robot_y - 1]
        point_down = [robot_x, robot_y + 1]

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

            # (point_up == game.end_pos),
            # (point_right == game.end_pos),
            # (point_down == game.end_pos),
            # (point_left == game.end_pos),

            robot_x < game.end_pos[0],  # target right
            robot_x > game.end_pos[0],  # target left
            robot_y < game.end_pos[1],  # target down
            robot_y > game.end_pos[1]  # target up

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
        if self.LEARNING_STARTS_IN >= 0:
            self.LEARNING_STARTS_IN -= 1

    def train_memory(self):
        if len(self.memory) > BATCH_SIZE and self.LEARNING_STARTS_IN <= 0:
            mini_sample = random.sample(self.memory, BATCH_SIZE)

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

    def save_data(self, n_games, epsilon, file_name='data.json'):

        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)
        complete_path = os.path.join(self.model_folder_path, file_name)

        data = {'episodes': n_games, 'epsilon': epsilon, 'learning_starts_in': self.LEARNING_STARTS_IN}
        with open(complete_path, 'w') as file:
            json.dump(data, file, indent=4)

    def retrieve_data(self):
        model_data_path = os.path.join(self.model_folder_path, self.data_file)
        if os.path.exists(model_data_path):
            with open(model_data_path, 'r') as file:
                data = json.load(file)

            if data is not None:
                self.episodes = data['episodes']
                self.epsilon = data['epsilon']
                self.LEARNING_STARTS_IN = data['learning_starts_in']


def train():
    game = Game()
    agent = Agent(game.ROWS, game.COLS)
    agent.load()
    status = deque(maxlen=100)

    while True:
        time.sleep(0.0008)
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get the new state
        reward, done = game.run_step(final_move)
        state_new = agent.get_state(game)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        # train after every 4 steps
        if agent.t_step % 4 == 0:
            agent.train_memory()

        # update target model
        agent.t_step = (agent.t_step + 1) % 5000
        if agent.t_step == 0:
            print("updating target model")
            print(agent.epsilon)
            agent.target_model.load_state_dict(agent.model.state_dict())

        if done:
            agent.episodes += 1
            if len(agent.memory) > agent.LEARNING_STARTS_IN and agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            if agent.LEARNING_STARTS_IN == 0:
                print("learning started, Episodes will reset")

            agent.model.save()

            status.append(game.outcome)

            wins = sum(status)
            loses = len(status) - wins

            game.message = "Episodes: " + str(agent.episodes) + \
                            "       Wins/Loses:  " + str(wins) \
                            + "/" + str(loses) + "" \
                                                 " (Last 100)"
            print(game.message + " epsilon: " + str(agent.epsilon))
            agent.save_data(agent.episodes, agent.epsilon)
            game.reset()


if __name__ == '__main__':
    train()
