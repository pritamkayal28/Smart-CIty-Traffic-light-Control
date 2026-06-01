"""
Entry point to run the Smart City Traffic Light Control Simulation.
"""
from src.simulation import TrafficSimulation

def main():
    print("Initializing Smart City Traffic Light Control Simulation...")
    print("Press PyGame window close button to exit.")
    sim = TrafficSimulation()
    sim.run()

if __name__ == "__main__":
    main()
