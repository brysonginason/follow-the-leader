# Follow the Leader: Copycat Trading Simulation

A simulation project exploring copycat trading dynamics, imitation behavior, and network evolution among investors.

## Overview

"Follow the Leader" models the phenomenon of copycat trading—where investors mimic successful strategies—using synthetic data and dynamic network simulations. Each investor is modeled as a node with randomly assigned characteristics (e.g., initial capital) and evolving portfolio performance driven by stochastic processes (e.g., Brownian motion). The simulation examines how imitation events and herding behavior may emerge, potentially influencing market trends and risks.

## Project Objectives

- **Understand Copycat Trading Dynamics:** Explore how and when investors choose to imitate others based on performance differences.
- **Investigate Herding Effects:** Analyze the formation of clusters in the investor network and assess potential market manipulation risks.
- **Benchmark Trading Strategies:** While primarily using synthetic data, the framework allows future integration of historical stock market data to compare simulated behavior against real-world trends.
- **Comprehensive Metrics Logging:** Capture detailed simulation data to enable thorough post-run analysis and visualization.

## Key Features

- **Synthetic Data Generation:** 
  - Generate investor profiles with random initial parameters.
  - Simulate portfolio performance using independent Brownian motion.
  
- **Dynamic Simulation Engine:**
  - Update investor performance in discrete time steps.
  - Calculate imitation probabilities using a normalized function based on recent performance and historical size differences.
  
- **Network Representation:**
  - Utilize a graph-based approach (e.g., via NetworkX) to capture directed copycat relationships.
  - Update network structure in real time as investors mimic strategies.
  
- **Logging and Analysis:**
  - Log key metrics such as performance changes, imitation events, and network statistics.
  - Enable both real-time monitoring and post-simulation data analysis.

## Repository Structure

```
follow-the-leader/
├── README.md            # Project overview and instructions
├── config/              # Configuration files (YAML/JSON) for simulation parameters
├── data/                # Synthetic data generators or historical data files
├── docs/                # Documentation and design details
├── notebooks/           # Jupyter notebooks for data analysis and visualization
├── src/
│   ├── init.py
|   ├── main.py          # Entry point for the simulation script
│   ├── models.py        # Definitions for investor models and nodes
│   ├── simulation.py    # Main simulation engine and time-step loop
│   ├── network.py       # Functions for network updates and imitation logic
│   ├── visualization.py # Plotting functions for network evolution and performance
│   └── utils.py         # Utility functions (logging, configuration management, etc.)
├── logs/                # Log files capturing simulation metrics and events
└── tests/               # Unit tests for the various modules
```

## Setup and Installation

1. **Clone the Repository:**
   ```
   git clone https://github.com/your-username/follow-the-leader.git
   cd follow-the-leader
   ```

2. **Create a Virtual Environment (Optional but Recommended):**
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Configure Simulation Parameters:**
   - Edit the configuration file in the config/ folder (e.g., simulation.yaml) to adjust parameters like the number of investors, imitation strength (α, β), and simulation duration.


## Running the Simulation

- **Run the Simulation Script:**
  ```
  python src/simulation.py --config config/simulation.yaml
  ```

  This command will:

    - Generate synthetic investor data.
    - Update investor performance over a series of time steps.
    - Compute and log imitation events along with network evolution.

## Logging and Analysis

- Logging:
    A dedicated logging module captures key simulation metrics (performance updates, imitation events, network properties) at each time step. Logs are stored in a designated directory (e.g., logs/) for later review.

- Post-Simulation Analysis:
    Use Jupyter notebooks in the notebooks/ directory to visualize and analyze the data. These notebooks can help uncover patterns such as herding behavior, network clustering, and overall market efficiency.

## Future Enhancements

- Historical Data Integration:
    Incorporate real stock market data for benchmarking simulation outcomes.

- Enhanced Visualization:
    Develop interactive dashboards to monitor simulation dynamics in real time.

- Scalability Improvements:
    Optimize the simulation engine for larger networks and more complex scenarios, possibly exploring parallel processing options.

## Acknowledgments

This project is inspired by our initial project presentation by Bryson, Diego, and Manuel. It aims to shed light on the mechanics of copycat trading and its potential impact on market behavior.

## License

This project is licensed under the MIT License. See the LICENSE file for details.