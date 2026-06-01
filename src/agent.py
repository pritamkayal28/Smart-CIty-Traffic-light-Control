import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from src.config import DEVICE, LR, GAMMA, BATCH_SIZE, MEMORY_SIZE, EPSILON_START, EPSILON_MIN, EPSILON_DECAY, PHASES
from src.model import DQN

class DQNAgent:
    """
    DQN Agent managing training, action selection, and replay memory
    """
    def __init__(self):
        self.policy_net = DQN().to(DEVICE)
        self.target_net = DQN().to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target net to evaluation mode
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        
    def select_action(self, state):
        """
        Selects an action using Epsilon-Greedy Exploration strategy.
        """
        if random.random() < self.epsilon:
            return random.randrange(len(PHASES))
        
        # Expect state to be a numpy array of shape (4,) or tensor
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            return torch.argmax(self.policy_net(state_t)).item()
            
    def remember(self, state, action, reward, next_state):
        """
        Pushes transition experience tuples into the replay buffer.
        """
        self.memory.append((state, action, reward, next_state))
        
    def optimize(self):
        """
        Sample a mini-batch from experience memory and perform one step of DQN optimization.
        """
        if len(self.memory) < BATCH_SIZE:
            return
            
        # Randomly sample transitions
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert list to tensors
        states_t = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(DEVICE)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        
        # Compute Q(s_t, a)
        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze()
        
        # Compute max Q(s_{t+1}, a) using target network
        next_q_values = self.target_net(next_states_t).max(1)[0].detach()
        
        # Compute expected Q value: reward + gamma * max Q(s_t+1)
        expected_q_values = rewards_t + GAMMA * next_q_values
        
        # Loss calculation (Mean Squared Error)
        loss = nn.MSELoss()(q_values, expected_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def decay_epsilon(self):
        """
        Decay epsilon at the end of each day/episode.
        """
        self.epsilon = max(self.epsilon * EPSILON_DECAY, EPSILON_MIN)
        
    def update_target_network(self):
        """
        Synchronize the target network with the policy network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
