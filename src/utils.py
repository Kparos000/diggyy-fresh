"""
Utility functions for Diggyy Fresh
Metrics, visualization helpers, and simulation utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


def calculate_waste_percentage(spoilage_cost: float, revenue: float) -> float:
    """
    Calculate waste percentage from costs

    Args:
        spoilage_cost: Total cost of spoiled items
        revenue: Total revenue from sales

    Returns:
        Waste percentage (0-100)
    """
    if revenue == 0:
        return 0.0
    return (spoilage_cost / revenue) * 100


def calculate_stockout_rate(stockouts: int, total_demand: int) -> float:
    """
    Calculate stockout rate

    Args:
        stockouts: Number of units out of stock
        total_demand: Total demand in units

    Returns:
        Stockout rate (0-100)
    """
    if total_demand == 0:
        return 0.0
    return (stockouts / total_demand) * 100


def calculate_profit(revenue: float, spoilage_cost: float, stockout_penalty: float) -> float:
    """
    Calculate net profit

    Args:
        revenue: Total revenue
        spoilage_cost: Cost of spoiled items
        stockout_penalty: Penalty for stockouts

    Returns:
        Net profit
    """
    return revenue - spoilage_cost - stockout_penalty


def run_simulation(env, policy, n_episodes: int = 1, verbose: bool = False) -> Dict:
    """
    Run simulation with given policy and collect metrics

    Args:
        env: Gymnasium environment
        policy: Policy with predict(observation) method
        n_episodes: Number of episodes to run
        verbose: Print detailed info

    Returns:
        Dictionary with aggregated metrics
    """

    results = {
        'episodes': [],
        'total_revenue': 0,
        'total_spoilage': 0,
        'total_stockouts': 0,
        'total_profit': 0,
    }

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_data = {
            'steps': [],
            'revenue': 0,
            'spoilage': 0,
            'stockouts': 0,
            'rewards': [],
        }

        done = False
        step = 0

        while not done:
            # Get action from policy
            if hasattr(policy, 'predict'):
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action = policy(obs)  # Simple function

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record step data
            step_data = {
                'step': step,
                'action': action.copy() if hasattr(action, 'copy') else action,
                'reward': reward,
                'revenue': info.get('revenue', 0),
                'spoilage_cost': info.get('spoilage_cost', 0),
                'stockout_penalty': info.get('stockout_penalty', 0),
            }

            episode_data['steps'].append(step_data)
            episode_data['rewards'].append(reward)

            step += 1

            if verbose and step % 5 == 0:
                print(f"  Step {step}: Reward={reward:.2f}, "
                      f"Revenue=${info.get('revenue', 0):.2f}, "
                      f"Spoilage=${info.get('spoilage_cost', 0):.2f}")

        # Aggregate episode metrics
        episode_data['revenue'] = info['total_revenue']
        episode_data['spoilage'] = info['total_spoilage']
        episode_data['stockouts'] = info['total_stockouts']
        episode_data['profit'] = calculate_profit(
            info['total_revenue'],
            info['total_spoilage'],
            0  # Stockout penalty already in reward
        )

        results['episodes'].append(episode_data)
        results['total_revenue'] += episode_data['revenue']
        results['total_spoilage'] += episode_data['spoilage']
        results['total_stockouts'] += episode_data['stockouts']
        results['total_profit'] += episode_data['profit']

        if verbose:
            print(f"\nEpisode {episode + 1}: "
                  f"Revenue=${episode_data['revenue']:.2f}, "
                  f"Waste=${episode_data['spoilage']:.2f}, "
                  f"Stockouts={episode_data['stockouts']}")

    # Calculate averages
    results['avg_revenue'] = results['total_revenue'] / n_episodes
    results['avg_spoilage'] = results['total_spoilage'] / n_episodes
    results['avg_stockouts'] = results['total_stockouts'] / n_episodes
    results['avg_profit'] = results['total_profit'] / n_episodes

    # Calculate waste percentage
    if results['total_revenue'] > 0:
        results['waste_percentage'] = (results['total_spoilage'] / results['total_revenue']) * 100
    else:
        results['waste_percentage'] = 0.0

    return results


def compare_policies_detailed(env, policies: Dict, n_episodes: int = 10) -> pd.DataFrame:
    """
    Compare multiple policies and return detailed comparison

    Args:
        env: Gymnasium environment
        policies: Dictionary of {policy_name: policy_object}
        n_episodes: Number of episodes per policy

    Returns:
        DataFrame with comparison metrics
    """

    comparison = []

    for policy_name, policy in policies.items():
        print(f"\nEvaluating {policy_name}...")
        results = run_simulation(env, policy, n_episodes=n_episodes, verbose=False)

        comparison.append({
            'Policy': policy_name,
            'Avg Revenue': results['avg_revenue'],
            'Avg Spoilage': results['avg_spoilage'],
            'Avg Stockouts': results['avg_stockouts'],
            'Avg Profit': results['avg_profit'],
            'Waste %': results['waste_percentage'],
        })

    df = pd.DataFrame(comparison)
    return df


def format_currency(value: float) -> str:
    """Format value as currency"""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value:.1f}%"


def create_daily_schedule(start_date: str = "2024-01-01", days: int = 30) -> List[Dict]:
    """
    Create a daily schedule with dates and day of week

    Args:
        start_date: Start date (YYYY-MM-DD)
        days: Number of days

    Returns:
        List of daily info dicts
    """

    start = datetime.strptime(start_date, "%Y-%m-%d")
    schedule = []

    for i in range(days):
        current_date = start + timedelta(days=i)
        schedule.append({
            'day': i,
            'date': current_date.strftime("%Y-%m-%d"),
            'day_of_week': current_date.weekday(),
            'day_name': current_date.strftime("%A"),
            'is_weekend': current_date.weekday() in [5, 6],
        })

    return schedule


def generate_demand_shock(base_demand: float, shock_type: str = "spike") -> float:
    """
    Generate demand shock for testing

    Args:
        base_demand: Normal demand level
        shock_type: Type of shock ('spike', 'drop', 'volatile')

    Returns:
        Adjusted demand
    """

    if shock_type == "spike":
        return base_demand * np.random.uniform(1.5, 2.0)
    elif shock_type == "drop":
        return base_demand * np.random.uniform(0.3, 0.6)
    elif shock_type == "volatile":
        return base_demand * np.random.uniform(0.5, 2.0)
    else:
        return base_demand


def aggregate_episode_data(episode_steps: List[Dict]) -> Dict:
    """
    Aggregate step-level data into episode summary

    Args:
        episode_steps: List of step dictionaries

    Returns:
        Aggregated episode metrics
    """

    if not episode_steps:
        return {}

    total_reward = sum(step['reward'] for step in episode_steps)
    total_revenue = sum(step.get('revenue', 0) for step in episode_steps)
    total_spoilage = sum(step.get('spoilage_cost', 0) for step in episode_steps)
    total_stockout = sum(step.get('stockout_penalty', 0) for step in episode_steps)

    return {
        'total_steps': len(episode_steps),
        'total_reward': total_reward,
        'avg_reward': total_reward / len(episode_steps),
        'total_revenue': total_revenue,
        'total_spoilage': total_spoilage,
        'total_stockout_penalty': total_stockout,
        'profit': total_revenue - total_spoilage - total_stockout,
        'waste_percentage': (total_spoilage / total_revenue * 100) if total_revenue > 0 else 0,
    }


def get_performance_grade(waste_percentage: float) -> Tuple[str, str]:
    """
    Get performance grade based on waste percentage

    Args:
        waste_percentage: Waste as percentage of revenue

    Returns:
        Tuple of (grade, description)
    """

    if waste_percentage < 10:
        return "A+", "Excellent - Minimal waste"
    elif waste_percentage < 20:
        return "A", "Great - Low waste"
    elif waste_percentage < 30:
        return "B", "Good - Moderate waste"
    elif waste_percentage < 40:
        return "C", "Fair - High waste"
    elif waste_percentage < 50:
        return "D", "Poor - Very high waste"
    else:
        return "F", "Critical - Excessive waste"


def print_summary(results: Dict, policy_name: str = "Policy"):
    """
    Print formatted summary of results

    Args:
        results: Results dictionary from run_simulation
        policy_name: Name of the policy
    """

    print(f"\n{'='*60}")
    print(f"  {policy_name} Performance Summary")
    print(f"{'='*60}")
    print(f"Episodes:          {len(results['episodes'])}")
    print(f"Avg Revenue:       {format_currency(results['avg_revenue'])}")
    print(f"Avg Spoilage:      {format_currency(results['avg_spoilage'])}")
    print(f"Avg Stockouts:     {results['avg_stockouts']:.0f} units")
    print(f"Avg Profit:        {format_currency(results['avg_profit'])}")
    print(f"Waste %:           {format_percentage(results['waste_percentage'])}")

    grade, description = get_performance_grade(results['waste_percentage'])
    print(f"Grade:             {grade} - {description}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")

    # Test metrics
    waste_pct = calculate_waste_percentage(spoilage_cost=50, revenue=500)
    print(f"Waste %: {waste_pct:.1f}%")

    stockout_rate = calculate_stockout_rate(stockouts=20, total_demand=200)
    print(f"Stockout rate: {stockout_rate:.1f}%")

    profit = calculate_profit(revenue=500, spoilage_cost=50, stockout_penalty=30)
    print(f"Profit: ${profit:.2f}")

    # Test schedule
    schedule = create_daily_schedule(days=7)
    print("\nWeekly schedule:")
    for day_info in schedule:
        print(f"  Day {day_info['day']}: {day_info['day_name']} ({day_info['date']})")

    # Test grading
    for waste in [5, 15, 25, 35, 45, 55]:
        grade, desc = get_performance_grade(waste)
        print(f"Waste {waste}%: {grade} - {desc}")

    print("\n✓ Utilities test complete!")
