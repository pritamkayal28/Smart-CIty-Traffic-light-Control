import sys
import pygame
import numpy as np
from src.config import (
    WIDTH, HEIGHT, FPS, ROAD_WIDTH, CENTER, STOP,
    GRASS, ASPHALT, CURB, LANE_WHITE, STOP_LINE, GREEN, RED, WHITE, PHASES,
    DAY_LENGTH, DECISION_INTERVAL, PANEL_BG, BLUE, EMERGENCY_COLOR
)
from src.environment import IntersectionEnv
from src.agent import DQNAgent

class TrafficSimulation:
    """
    Main simulator class coordinating graphics and learning steps.
    """
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Smart Traffic Control – DQN Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 22)

        # RL Environment and Agent
        self.env = IntersectionEnv()
        self.agent = DQNAgent()

        # Learning Curve Plot Surface
        self.curve_width = 300
        self.curve_height = 120
        self.curve_surface = pygame.Surface((self.curve_width, self.curve_height))
        
        # History lists for metrics plotting
        self.reward_history = []
        self.wait_history = []

        # Simulation metrics
        self.day = 1
        self.step = 0
        self.active_phase = 0
        self.last_reward = 0.0

        # Daily accumulators
        self.daily_reward = 0.0
        self.daily_wait = 0.0
        self.daily_decisions = 0

    def draw_background(self):
        """
        Renders green grass background, asphalt roads, lanes, stop lines, and curbs.
        """
        self.screen.fill(GRASS)

        # Roads
        pygame.draw.rect(self.screen, ASPHALT, (CENTER - ROAD_WIDTH // 2, 0, ROAD_WIDTH, HEIGHT))
        pygame.draw.rect(self.screen, ASPHALT, (0, CENTER - ROAD_WIDTH // 2, WIDTH, ROAD_WIDTH))
        
        # Curbs
        pygame.draw.rect(self.screen, CURB, (CENTER - ROAD_WIDTH // 2 - 6, 0, 6, HEIGHT))
        pygame.draw.rect(self.screen, CURB, (CENTER + ROAD_WIDTH // 2, 0, 6, HEIGHT))
        pygame.draw.rect(self.screen, CURB, (0, CENTER - ROAD_WIDTH // 2 - 6, WIDTH, 6))
        pygame.draw.rect(self.screen, CURB, (0, CENTER + ROAD_WIDTH // 2, WIDTH, 6))

        # Lane dividers
        dash, gap = 20, 15
        for y in range(0, HEIGHT, dash + gap):
            pygame.draw.rect(self.screen, LANE_WHITE, (CENTER - ROAD_WIDTH // 4, y, 4, dash))
            pygame.draw.rect(self.screen, LANE_WHITE, (CENTER + ROAD_WIDTH // 4, y, 4, dash))
        for x in range(0, WIDTH, dash + gap):
            pygame.draw.rect(self.screen, LANE_WHITE, (x, CENTER - ROAD_WIDTH // 4, dash, 4))
            pygame.draw.rect(self.screen, LANE_WHITE, (x, CENTER + ROAD_WIDTH // 4, dash, 4))

        # Stop lines
        pygame.draw.rect(self.screen, STOP_LINE, (CENTER - ROAD_WIDTH // 2, CENTER - STOP - 6, ROAD_WIDTH, 6))
        pygame.draw.rect(self.screen, STOP_LINE, (CENTER - ROAD_WIDTH // 2, CENTER + STOP, ROAD_WIDTH, 6))
        pygame.draw.rect(self.screen, STOP_LINE, (CENTER - STOP - 6, CENTER - ROAD_WIDTH // 2, 6, ROAD_WIDTH))
        pygame.draw.rect(self.screen, STOP_LINE, (CENTER + STOP, CENTER - ROAD_WIDTH // 2, 6, ROAD_WIDTH))
        
        # Intersection center area
        pygame.draw.rect(self.screen, (80, 80, 80), (CENTER - STOP, CENTER - STOP, 2 * STOP, 2 * STOP), 2)

    def draw_signals(self):
        """
        Draws green/red circular indicators reflecting current traffic lights configuration.
        """
        positions = {
            'N': (CENTER - 35, CENTER - STOP - 35),
            'S': (CENTER + 35, CENTER + STOP + 35),
            'E': (CENTER + STOP + 35, CENTER - 35),
            'W': (CENTER - STOP - 35, CENTER + 35)
        }
        active_dirs = PHASES[self.active_phase]
        for direction, pos in positions.items():
            color = GREEN if direction in active_dirs else RED
            pygame.draw.circle(self.screen, color, pos, 10)

    def draw_panel(self):
        """
        Renders text status indicators (HUD) on top-left of the screen.
        """
        cars_count = len(self.env.cars)
        
        # Calculate true average wait time for vehicles in the system
        all_waits = self.env.completed_waits + [c.wait for c in self.env.cars]
        avg_wait = sum(all_waits) / max(1, len(all_waits))
        
        # Determine training mode string
        mode = "RANDOM" if self.agent.epsilon > 0.7 else "MIXED" if self.agent.epsilon > 0.4 else "EXPLOIT"
        phase_names = {
            0: "North Only",
            1: "South Only",
            2: "East Only",
            3: "West Only",
            4: "North & South",
            5: "East & West"
        }
        active_phase_str = phase_names.get(self.active_phase, "Unknown")
        
        panel_items = [
            f"Simulated Day: {self.day}",
            f"Active Phase: {active_phase_str}",
            f"Epsilon (Explore): {self.agent.epsilon:.2f} ({mode})",
            f"Vehicles On Screen: {cars_count}",
            f"Avg Vehicle Wait: {avg_wait:.1f} steps",
            f"Last Reward Feedback: {self.last_reward:.2f}"
        ]

        # Draw a semi-transparent panel background for premium appearance
        bg_surface = pygame.Surface((260, 120))
        bg_surface.fill(PANEL_BG)
        bg_surface.set_alpha(180)
        self.screen.blit(bg_surface, (5, 5))

        # Render status lines
        for i, text in enumerate(panel_items):
            rendered_text = self.font.render(text, True, WHITE)
            self.screen.blit(rendered_text, (12, 10 + i * 18))

    def draw_lane_labels(self):
        """
        Draws N, S, E, W lane direction markers on the road approaches.
        """
        font_large = pygame.font.SysFont(None, 24, bold=True)
        # North Lane (top)
        self.screen.blit(font_large.render("NORTH LANE (↓)", True, WHITE), (CENTER - 150, 30))
        # South Lane (bottom)
        self.screen.blit(font_large.render("SOUTH LANE (↑)", True, WHITE), (CENTER + 30, HEIGHT - 50))
        # East Lane (right)
        self.screen.blit(font_large.render("EAST LANE (←)", True, WHITE), (WIDTH - 170, CENTER - 60))
        # West Lane (left)
        self.screen.blit(font_large.render("WEST LANE (→)", True, WHITE), (30, CENTER + 45))

    def draw_legend(self):
        """
        Renders a user-friendly legend on the bottom-left of the screen.
        """
        # Semi-transparent background
        bg_surface = pygame.Surface((250, 110))
        bg_surface.fill(PANEL_BG)
        bg_surface.set_alpha(180)
        self.screen.blit(bg_surface, (5, HEIGHT - 115))

        # Title
        title = self.font.render("SIMULATION LEGEND", True, WHITE)
        self.screen.blit(title, (10, HEIGHT - 110))

        # Legend items: (color, shape, text)
        # Passenger Car (Blue)
        pygame.draw.rect(self.screen, BLUE, (15, HEIGHT - 90, 10, 10))
        lbl_passenger = self.font.render("Passenger Vehicle", True, WHITE)
        self.screen.blit(lbl_passenger, (35, HEIGHT - 94))

        # Emergency Car (Red)
        pygame.draw.rect(self.screen, EMERGENCY_COLOR, (15, HEIGHT - 70, 10, 10))
        lbl_emergency = self.font.render("Emergency Vehicle (Override)", True, WHITE)
        self.screen.blit(lbl_emergency, (35, HEIGHT - 74))

        # Traffic Light (Green)
        pygame.draw.circle(self.screen, GREEN, (20, HEIGHT - 45), 6)
        lbl_green_light = self.font.render("Green Light (Go)", True, WHITE)
        self.screen.blit(lbl_green_light, (35, HEIGHT - 50))

        # Traffic Light (Red)
        pygame.draw.circle(self.screen, RED, (20, HEIGHT - 25), 6)
        lbl_red_light = self.font.render("Red Light (Stop)", True, WHITE)
        self.screen.blit(lbl_red_light, (35, HEIGHT - 30))

    def draw_learning_curve(self):
        """
        Plots normalized rewards and waiting times over consecutive days.
        """
        self.curve_surface.fill(PANEL_BG)
        
        if len(self.reward_history) > 1:
            # Draw Reward History (Green Curve)
            max_reward = max(self.reward_history)
            min_reward = min(self.reward_history)
            reward_range = max(1e-5, max_reward - min_reward)

            for i in range(1, len(self.reward_history)):
                x1 = int((i - 1) * self.curve_width / (len(self.reward_history) - 1))
                y1 = int(self.curve_height - ((self.reward_history[i - 1] - min_reward) / reward_range) * (self.curve_height - 20) - 10)
                x2 = int(i * self.curve_width / (len(self.reward_history) - 1))
                y2 = int(self.curve_height - ((self.reward_history[i] - min_reward) / reward_range) * (self.curve_height - 20) - 10)
                pygame.draw.line(self.curve_surface, (0, 220, 0), (x1, y1), (x2, y2), 2)
            
            # Draw Wait History (Red Curve)
            max_wait = max(self.wait_history)
            min_wait = min(self.wait_history)
            wait_range = max(1e-5, max_wait - min_wait)

            for i in range(1, len(self.wait_history)):
                x1 = int((i - 1) * self.curve_width / (len(self.wait_history) - 1))
                y1 = int(self.curve_height - ((self.wait_history[i - 1] - min_wait) / wait_range) * (self.curve_height - 20) - 10)
                x2 = int(i * self.curve_width / (len(self.wait_history) - 1))
                y2 = int(self.curve_height - ((self.wait_history[i] - min_wait) / wait_range) * (self.curve_height - 20) - 10)
                pygame.draw.line(self.curve_surface, (220, 0, 0), (x1, y1), (x2, y2), 2)

        # Outline border and labels
        pygame.draw.rect(self.curve_surface, WHITE, (0, 0, self.curve_width, self.curve_height), 1)
        
        # Legend/Labels
        lbl_reward = self.font.render("Reward (Green)", True, (0, 220, 0))
        lbl_wait = self.font.render("Wait Time (Red)", True, (220, 0, 0))
        self.curve_surface.blit(lbl_reward, (5, 5))
        self.curve_surface.blit(lbl_wait, (self.curve_width - 110, 5))

        self.screen.blit(self.curve_surface, (WIDTH - self.curve_width - 10, 10))

    def run(self):
        """
        Executes the primary simulation loop.
        """
        running = True
        state = self.env.get_state()

        while running:
            self.clock.tick(FPS)

            # Handle system close actions
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Draw static road grid
            self.draw_background()

            # Spawn new vehicles
            self.env.spawn_car()

            # Retrieve active phase directions
            active_directions = PHASES[self.active_phase]

            # Physics Update: advance vehicles according to stop-line signals
            self.env.update_lanes(active_directions)

            # Agent Action Interval (Decision is made every 60 frames)
            if self.step % DECISION_INTERVAL == 0:
                self.daily_decisions += 1
                
                # Fetch reward and next state observation
                reward = self.env.calculate_reward()
                next_state = self.env.get_state()
                
                self.last_reward = reward
                self.daily_reward += reward

                # Record transition memory
                self.agent.remember(state, self.active_phase, reward, next_state)
                
                # Advance agent decision
                self.active_phase = self.agent.select_action(next_state)
                state = next_state.copy()

            # Terminal Update every 600 frames (10 seconds of simulation time)
            if self.step % 600 == 0 and self.step > 0:
                queues = self.env.get_queues()
                q_str = f"N:{queues['N']} S:{queues['S']} E:{queues['E']} W:{queues['W']}"
                phase_names = {0: "North", 1: "South", 2: "East", 3: "West", 4: "N-S", 5: "E-W"}
                p_name = phase_names.get(self.active_phase, "Unknown")
                
                all_waits = self.env.completed_waits + [c.wait for c in self.env.cars]
                curr_avg_wait = sum(all_waits) / max(1, len(all_waits))
                
                print(f"[Step {self.step:5d}] Queues: [{q_str}] | Epsilon: {self.agent.epsilon:.2f} | Phase: {p_name} | Avg Wait: {curr_avg_wait:.1f} steps | Last Reward: {self.last_reward:.2f}")

            # DQN Optimization
            self.agent.optimize()

            # Render dynamic items
            for car in self.env.cars:
                car.draw(self.screen)

            # Render extra user friendly information
            self.draw_signals()
            self.draw_lane_labels()
            self.draw_legend()
            self.draw_panel()
            self.draw_learning_curve()

            # End of Episode (Day Reset)
            if self.step % DAY_LENGTH == 0 and self.step > 0:
                all_waits = self.env.completed_waits + [c.wait for c in self.env.cars]
                avg_daily_wait = sum(all_waits) / max(1, len(all_waits))
                avg_daily_reward = self.daily_reward / max(1, self.daily_decisions)
                
                print(f"\n--- Day {self.day} Completed ---")
                print(f"Avg Daily Reward: {avg_daily_reward:.2f}")
                print(f"Avg Wait Time   : {avg_daily_wait:.2f} steps")
                print(f"Epsilon         : {self.agent.epsilon:.2f}\n")

                # Save history metrics
                self.reward_history.append(avg_daily_reward)
                self.wait_history.append(avg_daily_wait)

                # Transition values
                self.day += 1
                self.agent.decay_epsilon()
                self.agent.update_target_network()
                self.env.reset()

                # Reset daily accumulators
                self.daily_reward = 0.0
                self.daily_decisions = 0

            pygame.display.flip()
            self.step += 1

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    sim = TrafficSimulation()
    sim.run()
