from collections import deque

import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 40

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
ORANGE = (255, 165, 0)
BLUE = (0, 0, 255)
LIGHT_GREY = (211, 211, 211)

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Robot Maze Game - AI - Deep Q Learning')


class Game:

    def __init__(self):

        self.ROWS, self.COLS = HEIGHT // GRID_SIZE, WIDTH // GRID_SIZE

        self.SCREEN_UPDATE = pygame.USEREVENT
        pygame.time.set_timer(self.SCREEN_UPDATE, 300)

        # Define the maze as a list of strings with correct dimensions
        self.maze = [
            "111111111111111",
            "100000000000001",
            "101101111011101",
            "100000001000001",
            "101111000011101",
            "101000001000001",
            "101011110011101",
            "100010000000101",
            "111010111001101",
            "100000100000001",
            "101111111011011",
            "101000001000001",
            "101011101011101",
            "100000101000001",
            "100000000000000"
        ]

        # Define fire pits as a list of tuples (row, col)
        self.fire_pits = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 12)]

        self.flags = [(2, 9), (8, 5), (15, 1), (19, 7), (13, 10)]

        # Robot starting position
        self.robot_pos = [1, 1]
        # End position
        self.end_pos = [14, 14]

        robot = pygame.image.load("resources/robot.jpg")
        self.robot_img = pygame.transform.scale(robot, (GRID_SIZE, GRID_SIZE))

        fire = pygame.image.load("resources/fire.jpg")
        self.fire_img = pygame.transform.scale(fire, (GRID_SIZE, GRID_SIZE))

        flag = pygame.image.load("resources/flag.jpg")
        self.flag_img = pygame.transform.scale(flag, (GRID_SIZE, GRID_SIZE))

        destination = pygame.image.load("resources/destination.jpg")
        self.destination = pygame.transform.scale(destination, (GRID_SIZE, GRID_SIZE))

        self.moves = 0
        self.game_over = False
        self.reward = 0
        self.max_moves = 500
        self.position_history = deque(maxlen=2)
        self.wait = 0.008
        self.message = ''
        self.outcome = 0  # 1 for win and 0 for lose

    def reset(self):
        self.moves = 0
        self.robot_pos = [1, 1]
        self.game_over = False
        self.reward = 0
        self.outcome = 0  # 1 for win and 0 for lose

    def display_moves(self):
        pass
        # font = pygame.font.SysFont('arial', 20)
        # move_msg = "Moves: " + str(self.moves)
        # score = font.render(f"{move_msg}", True, (200, 200, 200))
        # screen.blit(score, (100, 10))

    def display_episodes(self):
        font = pygame.font.SysFont('arial', 20)
        score = font.render(f"{self.message}", True, (200, 200, 200))
        screen.blit(score, (100, 10))

    def display_objects(self):
        # Draw the self.maze
        screen.fill(WHITE)
        for row in range(self.ROWS):
            for col in range(self.COLS):
                if self.maze[row][col] == '0':
                    color = LIGHT_GREY
                elif self.maze[row][col] == '1':
                    color = BLACK
                pygame.draw.rect(screen, color, pygame.Rect(row * GRID_SIZE, col * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # Draw the fire pits
        for pit in self.fire_pits:
            screen.blit(self.fire_img, (pit[0] * GRID_SIZE, pit[1] * GRID_SIZE))
            # pygame.draw.rect(screen, RED, pygame.Rect(pit[1] * GRID_SIZE, pit[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # Draw the end position
        screen.blit(self.destination, (self.end_pos[0] * GRID_SIZE, self.end_pos[1] * GRID_SIZE))
        # pygame.draw.rect(screen, GREEN, pygame.Rect(self.end_pos[1] * GRID_SIZE, self.end_pos[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # Draw Flags
        # for flag in self.flags:
        #     screen.blit(self.flag_img, (flag[0] * GRID_SIZE, flag[1] * GRID_SIZE))

        # Draw the robot
        screen.blit(self.robot_img, (self.robot_pos[0] * GRID_SIZE, self.robot_pos[1] * GRID_SIZE))

        self.display_moves()
        self.display_episodes()

    def move(self):
        self.reward = -0.5

        # Check for win condition
        if self.robot_pos == self.end_pos:
            # print("You win!")
            self.reward = 50
            self.game_over = True
            self.outcome = 1  # win

        # Check if robot stepped on a fire pit
        if tuple(self.robot_pos) in self.fire_pits:
            # print("You stepped on a fire pit! Game over!")
            self.reward = -10
            self.game_over = True
            self.outcome = 0  # lose

        # # Award for getting to reach flags
        # if tuple(self.robot_pos) in self.flags:
        #     self.reward = 2

        # if self.moves > self.max_moves:
        #     self.reward = -1
        #     self.game_over = True
        #     print("Just Roaming around, Game over")

        # Check for repetition by tracking positions
        # if (self.robot_pos[0], self.robot_pos[1]) in self.position_history:
        #
        #     self.reward = -0.5
        # self.position_history.append((self.robot_pos[0], self.robot_pos[1]))

    def run_step(self, action):

        # action = [up, right, down, left]

        new_pos = self.robot_pos.copy()

        direction = 'up'

        if action[0] == 1:
            direction = 'up'
            new_pos[1] -= 1
        elif action[1] == 1:
            direction = 'right'
            new_pos[0] += 1
        elif action[2] == 1:
            direction = 'down'
            new_pos[1] += 1
        elif action[3] == 1:
            direction = 'left'
            new_pos[0] -= 1

        # Check if the new position is within the maze bounds and not a wall
        if 0 <= new_pos[0] < self.ROWS and 0 <= new_pos[1] < self.COLS and self.maze[new_pos[0]][new_pos[1]] == '0':
            self.robot_pos = new_pos
            self.moves += 1
        else:
            # negative reward for hitting the wall
            self.reward = -1
            return self.reward, self.game_over
            # print("WALL")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    if self.wait > 0.001:
                        self.wait -= 0.001
                        print("speed increased: " + str(self.wait))
                elif event.key == pygame.K_DOWN:
                    self.wait += 0.001
                    print("speed DECREASE: " + str(self.wait))
        self.move()
        self.display_objects()
        pygame.display.flip()

        return self.reward, self.game_over
