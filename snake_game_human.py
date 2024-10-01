import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.SysFont("arial", 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
DARK_GREEN = (0, 100, 0)

# Game parameters
BLOCK_SIZE = 20


# Gradient function for smooth colors
def gradient_color(color1, color2, blend_factor):
    return (
        int(color1[0] * (1 - blend_factor) + color2[0] * blend_factor),
        int(color1[1] * (1 - blend_factor) + color2[1] * blend_factor),
        int(color1[2] * (1 - blend_factor) + color2[2] * blend_factor),
    )


class SnakeGame:

    def __init__(
        self,
        w=640,
        h=480,
        speed=15,
        parallel_food=1,
        wall_collision=True,
        is_auto=False,
    ):
        self.w = w
        self.h = h
        self.speed = speed
        self.aux = None
        self.is_auto = is_auto
        self.wall_collision = wall_collision
        self.parallel_food = parallel_food
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Enhanced Snake Game")
        self.clock = pygame.time.Clock()

        # Initialize game state
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]

        self.score = 0
        self.foods = []
        self._place_food()

    def _place_food(self):
        self.foods = self.foods or []  # Clear any existing food
        parallel_food = self.parallel_food
        for _ in range(parallel_food - len(self.foods)):  # Place 3 food items
            while True:
                x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                new_food = Point(x, y)
                if new_food not in self.snake and new_food not in self.foods:
                    self.foods.append(new_food)
                    break

    def get_random_direction(self, current_direction):
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

        # Prevent the snake from moving in the opposite direction directly
        if current_direction == Direction.UP:
            directions.remove(Direction.DOWN)
        elif current_direction == Direction.DOWN:
            directions.remove(Direction.UP)
        elif current_direction == Direction.LEFT:
            directions.remove(Direction.RIGHT)
        elif current_direction == Direction.RIGHT:
            directions.remove(Direction.LEFT)

        return random.choice(directions)

    def get_safe_directions(self, snake, head, direction, w, h):

        # Get the current snake body and head positions
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        safe_directions = []

        for dir in directions:
            # Predict the next head position based on the current direction
            next_x, next_y = head.x, head.y
            if dir == Direction.RIGHT:
                next_x += BLOCK_SIZE
            elif dir == Direction.LEFT:
                next_x -= BLOCK_SIZE
            elif dir == Direction.DOWN:
                next_y += BLOCK_SIZE
            elif dir == Direction.UP:
                next_y -= BLOCK_SIZE

            next_head = Point(next_x, next_y)

            # Check if the next move will result in a collision with the wall (without wrap-around)
            if next_x >= w or next_x < 0 or next_y >= h or next_y < 0:
                continue  # Avoid moves that will cause a wall collision

            # Check if the next move will result in a collision with the snake's body
            if next_head in snake:
                continue  # Avoid moves that will cause self-collision

            # If no collision, add the direction to safe_directions
            safe_directions.append(dir)

        # If no safe directions found, keep the current direction
        if not safe_directions:
            return direction
        return random.choice(safe_directions)

    def play_step(self):

        if self.is_auto:
            # Automatic mode: randomly change the snake's direction
            # self.direction = self.get_random_direction(self.direction)
            # self.automatic_move()
            self.direction = self.get_safe_directions(
                self.snake, self.head, self.direction, self.w, self.h
            )
        else:
            # Handle user input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                        self.direction = Direction.LEFT
                    elif (
                        event.key == pygame.K_RIGHT and self.direction != Direction.LEFT
                    ):
                        self.direction = Direction.RIGHT
                    elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                        self.direction = Direction.UP
                    elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                        self.direction = Direction.DOWN

        # Move snake
        self._move(self.direction)
        self.snake.insert(0, self.head)

        # Check for collisions
        if self._is_collision():
            return True, self.score

        # Check if the snake has eaten any food
        if self.head in self.foods:
            self.score += 1
            self.foods.remove(self.head)  # Remove eaten food
            self._place_food()  # Add a new food to keep 3 on the board
        else:
            self.snake.pop()

        # Update game visuals and clock
        self._update_ui()
        self.clock.tick(self.speed)

        return False, self.score

    def _is_collision(self):
        if self.wall_collision:
            # Check if snake hits boundary
            if (
                self.head.x >= self.w
                or self.head.x < 0
                or self.head.y >= self.h
                or self.head.y < 0
            ):
                return True
        # Check if snake hits itself
        if self.head in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)  # Enhanced background color

        # Draw the snake with a gradient effect
        for idx, pt in enumerate(self.snake):
            blend_factor = idx / len(self.snake)
            snake_color = gradient_color(GREEN, YELLOW, blend_factor)
            pygame.draw.rect(
                self.display,
                snake_color,
                pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE),
            )

        # Draw all foods
        for food in self.foods:
            pygame.draw.circle(
                self.display,
                RED,
                (food.x + BLOCK_SIZE // 2, food.y + BLOCK_SIZE // 2),
                BLOCK_SIZE // 2,
            )

        # Display score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        if not self.wall_collision:
            # Wrap around if the snake crosses the boundary
            if x >= self.w:
                x = 0  # Appears on the left edge
            elif x < 0:
                x = self.w - BLOCK_SIZE  # Appears on the right edge

            if y >= self.h:
                y = 0  # Appears at the top edge
            elif y < 0:
                y = self.h - BLOCK_SIZE  # Appears at the bottom edge

        self.head = Point(x, y)


if __name__ == "__main__":
    max_tries = 5  # Define the number of tries
    total = 0

    while total < max_tries:
        game = SnakeGame(
            speed=20, parallel_food=300, wall_collision=True, is_auto=True
        )  # Reset the game for each new try

        # Main game loop
        while True:
            game_over, score = game.play_step()

            if game_over:
                total += 1
                print(f"Try {total}: Final Score", score)
                break  # Exit current game loop when game is over

    pygame.quit()
    print(f"You played {max_tries} tries. Game over.")
