#!/usr/bin/env python3
"""
Main entry point for the Follow the Leader copycat trading simulation.
"""
from simulation import run_advanced_simulation, compare_scenarios, analyze_drift_influence
import matplotlib.pyplot as plt


def main():
    """
    Main function demonstrating different simulation capabilities.
    """
    print("Follow the Leader: Copycat Trading Simulation")
    print("=" * 50)
    
    # Run a basic simulation
    print("\n1. Running basic simulation...")
    result = run_advanced_simulation(
        n=20, 
        steps=100, 
        alpha=1, 
        beta=0,
        visualize_at=[0, 24, 49, 99],
        seed=42
    )
    
    print(f"\nSimulation completed!")
    print(f"Final wealth range: {result['wealth'].min():.2f} - {result['wealth'].max():.2f}")
    print(f"Average influence: {sum(result['influences'])/len(result['influences']):.3f}")
    
    # Demonstrate scenario comparison
    print("\n2. Running scenario comparison...")
    compare_results = compare_scenarios()
    
    print("\n3. Analyzing drift influence...")
    drift_coeffs = analyze_drift_influence()
    
    print("\nAll demonstrations completed!")


if __name__ == "__main__":
    main()