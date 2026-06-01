
# Smart City Traffic Light Control using Reinforcement Learning

An adaptive traffic signal control system at a four-way urban intersection using Deep Q-Learning (DQN). This project simulates vehicle queues, handles emergency vehicle overrides, dynamically controls traffic light phases to minimize overall wait times, and renders learning progress curves in real-time.

Developed based on the research report by:
**Sahitya Biswas, Sayan Ghosh, Pritam Koyal**  
*Ramakrishna Mission Vivekananda Educational and Research Institute, West Bengal*

---

##  System Architecture & RL Formulation

Traditional traffic signals rely on fixed-timers which fail to adapt to real-time traffic surges. This project models the traffic control problem as a Markov Decision Process (MDP) and trains an RL agent online using a custom PyGame environment.

### 1. State Space ($S$)
The state at time step $t$, denoted by $s_t \in S$, represents the current congestion level at the intersection. It is encoded as a 4-dimensional vector containing the number of vehicles waiting/approaching in each lane direction:
$$s_t = [q_N(t), q_S(t), q_E(t), q_W(t)]$$
where $q_N(t)$, $q_S(t)$, $q_E(t)$, and $q_W(t)$ denote the vehicle count in the North, South, East, and West lanes respectively.

### 2. Action Space ($A$)
The action space consists of 6 discrete traffic light phases:
1. **Green for North only** (`['N']`)
2. **Green for South only** (`['S']`)
3. **Green for East only** (`['E']`)
4. **Green for West only** (`['W']`)
5. **Green for North–South directions** (`['N', 'S']`)
6. **Green for East–West directions** (`['E', 'W']`)

### 3. Reward Function ($R$)
The reward function encourages the agent to maximize flow efficiency by penalizing queue growth and wait times:
$$R_t = \frac{2 \times \Delta Q_t + 1 \times \Delta W_t}{\max(1, N_t)}$$
Where:
- $\Delta Q_t = Q_{t-1} - Q_t$ is the reduction in total queue length.
- $\Delta W_t = W_{t-1} - W_t$ is the reduction in cumulative wait time.
- $N_t$ is the total number of vehicles currently present in the intersection.

---

##  Repository Structure

The project has been refactored from a single monolithic Jupyter Notebook into a modular, industry-standard structure:

```
Smart-CIty-Traffic-light-Control/
├── src/
│   ├── __init__.py
│   ├── config.py           # Window layouts, color constants, and hyper-parameters
│   ├── model.py            # Deep Q-Network PyTorch definition
│   ├── agent.py            # Replay buffer management, epsilon-greedy action selection, and optimization
│   ├── environment.py      # Car mechanics, lane collision rules, and queue state/reward tracking
│   └── simulation.py       # PyGame graphic grid, panel statistics, and main loop coordinator
├── docs/
│   └── Report_Team_SSP.pdf # Original team research paper
├── notebooks/
│   └── SCTLC.ipynb         # Original notebook code preserved for reference
├── main.py                 # Application entry point script
├── requirements.txt        # Package dependencies list
├── .gitignore              # Ignores byte caches, IDE configs, and virtual environments
└── README.md               # Project documentation
```

---

##  Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Pip package manager

### 1. Clone the Repository
```bash
git clone https://github.com/Sahityabiswas/Smart_CIty_Traffic_light_Control.git
cd Smart_CIty_Traffic_light_Control
```

### 2. Install Dependencies
Install the required packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

##  Running the Simulation

To launch the traffic light controller simulation and start training the DQN agent:
```bash
python main.py
```

### Key Interactive Features:
* **Emergency Override**: Emergency vehicles (marked in **red**) automatically override the traffic signal. They are granted priority and will cross the intersection regardless of the green light status.
* **Real-time Metrics Panel**: The top-left HUD displays the current simulated day, current exploration rate ($\epsilon$), active vehicle counts, average waiting times, and reward feedbacks.
* **Dynamic Learning Curve**: The top-right panel draws real-time training progress curves. The green line shows the **average daily reward** and the red line shows the **average vehicle wait time**.

---

##  Hyperparameters

The learning performance of the DQN agent depends on the following parameters configured in `src/config.py`:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| Learning Rate | `0.001` | Adam Optimizer step size |
| Discount Factor ($\gamma$) | `0.95` | Importance given to long-term rewards |
| Batch Size | `64` | Number of experiences sampled from memory |
| Replay Buffer Size | `6000` | Transitions stored for model updates |
| Epsilon Decay | `0.90` | Factor by which exploration rate decays each day |
| Epsilon Min | `0.40` | Minimum exploration rate floor |

---

