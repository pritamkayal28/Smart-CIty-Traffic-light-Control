import torch

# Pygame Window Dimensions
WIDTH = 1000
HEIGHT = 885
FPS = 60

# Road Layout Geometry
ROAD_WIDTH = 160
CAR_SIZE = 10
GAP = 14

CENTER = WIDTH // 2
STOP = ROAD_WIDTH // 2

# Colors (Harmonious Palette for Premium Styling)
GRASS = (40, 130, 40)
ASPHALT = (50, 50, 50)
LANE_WHITE = (220, 220, 220)
STOP_LINE = (255, 255, 255)
CURB = (90, 90, 90)

GREEN = (0, 220, 0)
RED = (220, 0, 0)
WHITE = (240, 240, 240)
BLUE = (0, 150, 255)
EMERGENCY_COLOR = (255, 60, 60)
PANEL_BG = (30, 30, 30)

# Traffic Light Phase Actions
# Actions select which directions receive green signal:
# Action 0: North only
# Action 1: South only
# Action 2: East only
# Action 3: West only
# Action 4: North and South simultaneously
# Action 5: East and West simultaneously
PHASES = [
    ['N'], ['S'], ['E'], ['W'],
    ['N', 'S'], ['E', 'W']
]

# Reinforcement Learning Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Deep Q-Network (DQN) Hyperparameters
LR = 0.001
GAMMA = 0.95
BATCH_SIZE = 64
MEMORY_SIZE = 6000
EPSILON_START = 1.0
EPSILON_MIN = 0.4
EPSILON_DECAY = 0.9

# Episode & Simulation Duration
DAY_LENGTH = FPS * 120  # 7200 steps (equivalent to 120 seconds of simulation time)
DECISION_INTERVAL = 60  # Action is chosen every 60 steps (1 second)
CAR_SPAWN_PROB = 0.05   # Spawning probability per frame
