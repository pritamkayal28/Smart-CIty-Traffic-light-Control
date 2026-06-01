import random
import pygame
import numpy as np
from src.config import (
    WIDTH, HEIGHT, CENTER, STOP, ROAD_WIDTH, CAR_SIZE, GAP,
    EMERGENCY_COLOR, BLUE, CAR_SPAWN_PROB
)

class Car:
    """
    Represents a vehicle in the traffic simulation.
    """
    def __init__(self, direction):
        self.dir = direction  # 'N', 'S', 'E', or 'W'
        self.speed = 2.5
        self.wait = 0
        self.committed = False
        self.emergency = random.random() < 0.05  # 5% probability of being an emergency vehicle
        self.passed = False

        # Set initial positions based on starting direction
        if direction == 'N':
            self.x, self.y = CENTER - 40, -30
        elif direction == 'S':
            self.x, self.y = CENTER + 30, HEIGHT + 30
        elif direction == 'E':
            self.x, self.y = WIDTH + 30, CENTER - 40
        elif direction == 'W':
            self.x, self.y = -30, CENTER + 30

    def in_intersection(self):
        """
        Checks if the car is currently inside the central intersection box.
        """
        return (CENTER - STOP < self.x < CENTER + STOP and 
                CENTER - STOP < self.y < CENTER + STOP)

    def is_off_screen(self):
        """
        Checks if the car has fully driven off the screen.
        """
        if self.dir == 'N' and self.y > HEIGHT + 50:
            return True
        if self.dir == 'S' and self.y < -50:
            return True
        if self.dir == 'E' and self.x < -50:
            return True
        if self.dir == 'W' and self.x > WIDTH + 50:
            return True
        return False

    def forward(self):
        """
        Advances the car's position in its current direction.
        """
        if self.dir == 'N':
            self.y += self.speed
        elif self.dir == 'S':
            self.y -= self.speed
        elif self.dir == 'E':
            self.x -= self.speed
        elif self.dir == 'W':
            self.x += self.speed

    def draw(self, screen):
        """
        Renders the car as a rectangle.
        Emergency vehicles are colored red; passenger cars are blue.
        """
        color = EMERGENCY_COLOR if self.emergency else BLUE
        pygame.draw.rect(screen, color, (self.x, self.y, CAR_SIZE, CAR_SIZE))


class IntersectionEnv:
    """
    Handles the intersection traffic environment physics, collision logic, 
    and DQN state/reward transitions.
    """
    def __init__(self):
        self.cars = []
        self.prev_total_queue = 0
        self.prev_total_wait = 0
        self.completed_waits = []

    def reset(self):
        """
        Resets wait counters for existing cars and clears history.
        """
        for car in self.cars:
            car.wait = 0
        self.prev_total_queue = 0
        self.prev_total_wait = 0
        self.completed_waits = []

    def spawn_car(self):
        """
        Spawns a new car in a random direction with a predefined probability.
        """
        if random.random() < CAR_SPAWN_PROB:
            direction = random.choice(['N', 'S', 'E', 'W'])
            self.cars.append(Car(direction))

    def get_queues(self):
        """
        Counts the number of active cars in each direction.
        """
        queues = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        for car in self.cars:
            queues[car.dir] += 1
        return queues

    def get_state(self):
        """
        Returns the environment state representation as an array of queue lengths.
        """
        queues = self.get_queues()
        return np.array([queues[d] for d in ['N', 'S', 'E', 'W']], dtype=np.float32)

    def update_lanes(self, active_phase_directions):
        """
        Applies stop-line rules and moves cars forward or increments wait times.
        """
        # Check if any car from an opposing direction is currently inside the intersection
        intersection_blocked = False
        for car in self.cars:
            if car.in_intersection() and car.dir not in active_phase_directions:
                intersection_blocked = True
                break

        # Group cars by direction
        lanes = {d: [] for d in ['N', 'S', 'E', 'W']}
        for car in self.cars:
            lanes[car.dir].append(car)

        for d in lanes:
            # Sort cars to prioritize the ones closest to the intersection stop line
            if d == 'N':
                lanes[d].sort(key=lambda c: -c.y)
                stop, axis, sign = CENTER - STOP - CAR_SIZE, 'y', 1
            elif d == 'S':
                lanes[d].sort(key=lambda c: c.y)
                stop, axis, sign = CENTER + STOP, 'y', -1
            elif d == 'E':
                lanes[d].sort(key=lambda c: c.x)
                stop, axis, sign = CENTER + STOP, 'x', -1
            else:  # 'W'
                lanes[d].sort(key=lambda c: -c.x)
                stop, axis, sign = CENTER - STOP - CAR_SIZE, 'x', 1

            front_pos = None
            green = d in active_phase_directions

            for c in lanes[d]:
                # If already committed to intersection, just keep moving
                if c.committed:
                    c.forward()
                    continue

                if c.in_intersection():
                    c.committed = True
                    c.forward()
                    continue

                # Collision avoidance limit position
                pos = getattr(c, axis)
                limit = stop if front_pos is None else front_pos - sign * (GAP + CAR_SIZE)

                # Move if green light (and intersection not blocked), emergency vehicle, or has room to advance behind stop line
                is_behind_stop_line = sign * pos < sign * limit
                can_cross = (green and not intersection_blocked) or (c.emergency and not intersection_blocked)

                if can_cross or is_behind_stop_line:
                    c.forward()
                else:
                    c.wait += 1

                front_pos = getattr(c, axis)

        # Collect waiting times of cars being removed
        for c in self.cars:
            if c.is_off_screen():
                self.completed_waits.append(c.wait)

        # Performance Fix: Remove cars that have gone off-screen
        self.cars = [c for c in self.cars if not c.is_off_screen()]

    def calculate_reward(self):
        """
        Calculates the reward for the agent's action based on queue and waiting-time reduction.
        Formula: R_t = (2 * Delta Q_t + 1 * Delta W_t) / max(1, N_t)
        """
        queues = self.get_queues()
        total_queue = sum(queues.values())
        total_wait = sum(c.wait for c in self.cars)

        queue_reduction = self.prev_total_queue - total_queue
        wait_reduction = self.prev_total_wait - total_wait

        # Store values for next calculations
        self.prev_total_queue = total_queue
        self.prev_total_wait = total_wait

        # Calculate reward normalized by number of vehicles present
        reward = (2.0 * queue_reduction + 1.0 * wait_reduction) / max(1, len(self.cars))
        return reward
