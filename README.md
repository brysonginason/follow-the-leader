# Follow the Leader: Copycat Trading Simulation

A sophisticated simulation modeling copycat trading dynamics, imitation behavior, and network evolution among investors using advanced matrix-based algorithms and visualization techniques.

## Overview

"Follow the Leader" explores copycat trading phenomena where investors mimic successful strategies. The simulation uses matrix-based probability calculations, Brownian motion for performance modeling, and advanced network visualization to study herding effects and market manipulation risks.

## Key Features

### Advanced Simulation Engine
- **Matrix-Based Processing**: Efficient system solving with diagonal replacement for large-scale simulations
- **Government Intervention**: Automatic clipping at -90% to prevent market crashes  
- **Top-K Filtering**: Smart connection limiting to most probable imitation targets
- **Multi-Scenario Analysis**: Compare different alpha/beta parameter combinations

### Sophisticated Visualization
- **Radial Network Layout**: High performers positioned near center based on combined performance/wealth scores
- **Intelligent Coloring**: Green/red nodes for performance, weighted blue edges for copying strength
- **Dual Colorbars**: Separate scales for node performance and edge weights
- **Influence Tracking**: Real-time monitoring of dominant player influence over time

### Comprehensive Analysis Tools
- **Wealth Distribution Analysis**: Histogram visualization of wealth inequality
- **Drift Coefficient Analysis**: Identify investors with highest performance potential  
- **Scenario Comparison**: Side-by-side analysis of different market conditions
- **Performance Tracking**: Historical performance visualization with trend analysis

## Quick Start

### Installation
```bash
git clone https://github.com/brysonginason/follow-the-leader.git
cd follow-the-leader
pip install -r requirements.txt
```

### Run Simulations
```bash
# Basic demonstration with multiple scenarios
python src/main.py

# Advanced simulation with custom parameters  
python src/simulation.py --n 50 --steps 1000 --alpha 1 --beta 0

# Compare different market scenarios
python src/simulation.py --compare

# Analyze investor drift coefficients
python src/simulation.py --analyze-drift

# Original optimal implementation
jupyter notebook notebooks/sparta.ipynb
```

## Simulation Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|--------|
| `n` | Number of investors | 50 | 10-1000 |
| `steps` | Simulation duration | 1000 | 50-10000 |
| `alpha` | Performance weight | 1 | 0.1-10 |
| `beta` | Wealth weight | 0 | 0-500 |
| `seed` | Random seed | None | Any integer |

## Market Scenarios

The simulation supports comprehensive scenario analysis:

### Performance-Driven Markets (α=1, β=0)
Pure merit-based copying where investors follow top performers regardless of wealth.

### Balanced Markets (α=1, β=1) 
Equal weighting of performance and wealth in imitation decisions.

### Wealth-Dominated Markets (α=1, β=2)
Wealth inequality drives copying behavior more than actual performance.

### Extreme Inequality (α=1, β=500)
Ultra-wealthy investors dominate regardless of performance quality.

## Technical Architecture

### Core Algorithms
- **Copying Probability**: `P(i→j) = α×(perf_j - perf_i) + β×(wealth_j - wealth_i)`
- **System Solving**: Matrix inversion with diagonal replacement for efficient computation  
- **Brownian Motion**: Individual drift/volatility parameters per investor
- **Government Intervention**: Automatic -90% loss clipping to prevent system collapse

### Advanced Features
- **Top-K Filtering**: Limits connections to top 5 or n/100 most probable targets
- **Radial Positioning**: Performance-based layout with high performers centralized
- **Influence Tracking**: Real-time monitoring of copying network concentration
- **Multi-Scale Visualization**: Node sizes, colors, and edge weights convey different metrics

## Project Structure

```
follow-the-leader/
├── README.md                 # This file
├── CLAUDE.md                 # Developer guidance for Claude Code
├── requirements.txt          # Python dependencies  
├── notebooks/
│   └── sparta.ipynb         # Original optimal implementation
├── src/
│   ├── main.py              # Entry point with demonstrations
│   ├── simulation.py        # Advanced simulation engine
│   ├── models.py            # Brownian motion and probability models
│   ├── network.py           # Matrix-based copying algorithms  
│   ├── visualization.py     # Advanced plotting and network visualization
│   └── utils.py             # Logging and utility functions
├── data/
│   └── dummy_data.csv       # Sample dataset
└── .gitignore               # Git ignore rules
```

## Research Applications

### Market Analysis
- Study herding behavior formation and dissolution
- Analyze wealth concentration effects on market dynamics  
- Measure government intervention effectiveness

### Risk Assessment  
- Identify potential market manipulation vulnerabilities
- Quantify systemic risk from copycat trading cascades
- Evaluate regulatory policy impacts

### Strategy Development
- Benchmark trading algorithms against copycat behavior
- Optimize portfolio performance in imitation-heavy markets
- Design anti-herding investment strategies

## Future Enhancements

- **Real Market Data Integration**: Historical stock data for validation
- **Machine Learning Models**: Predictive copying behavior analysis  
- **Interactive Dashboards**: Real-time simulation monitoring
- **Distributed Computing**: Parallel processing for large-scale simulations
- **Advanced Network Metrics**: Centrality measures and community detection

## Acknowledgments

Developed by Bryson, Diego, and Manuel. Inspired by research into copycat trading dynamics and their systemic market impacts.

## License

MIT License - see LICENSE file for details.