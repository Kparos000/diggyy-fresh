"""
Synthetic Demand Data Generator for Diggyy Fresh
Generates realistic grocery demand patterns when real Instacart data is unavailable
"""

import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


class SyntheticDemandGenerator:
    """
    Generate realistic synthetic grocery demand data

    Simulates:
    - Day-of-week patterns (weekends higher)
    - Product-specific seasonality
    - Random noise (realistic variability)
    - Correlations between products
    """

    def __init__(self, seed: int = 42):
        """
        Initialize generator with random seed for reproducibility

        Args:
            seed: Random seed for consistent data generation
        """
        np.random.seed(seed)
        self.seed = seed

        # Product catalog matching environment.py
        self.products = {
            'berries': {
                'shelf_life': 3,
                'price': 4.99,
                'cost': 2.50,
                'base_demand': 15,
                # Friday-Sunday peak (weekend snacking, entertaining)
                'day_pattern': {0: 0.9, 1: 0.8, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.3, 6: 1.2},
                'variability': 0.35,  # Higher variability (perishable, discretionary)
            },
            'milk': {
                'shelf_life': 7,
                'price': 3.49,
                'cost': 1.75,
                'base_demand': 20,
                # Steady staple with slight weekend bump
                'day_pattern': {0: 1.0, 1: 0.9, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.2, 6: 1.1},
                'variability': 0.25,  # Lower variability (staple)
            },
            'bread': {
                'shelf_life': 5,
                'price': 2.99,
                'cost': 1.50,
                'base_demand': 18,
                # Weekend peak (baking, sandwiches for week)
                'day_pattern': {0: 1.0, 1: 0.9, 2: 1.0, 3: 1.0, 4: 1.1, 5: 1.3, 6: 1.2},
                'variability': 0.30,
            },
            'leafy_greens': {
                'shelf_life': 4,
                'price': 3.99,
                'cost': 2.00,
                'base_demand': 12,
                # Thu-Sat peak (meal prep for health-conscious shoppers)
                'day_pattern': {0: 0.9, 1: 0.8, 2: 0.9, 3: 1.0, 4: 1.2, 5: 1.3, 6: 1.1},
                'variability': 0.40,  # Higher variability (discretionary)
            },
            'tomatoes': {
                'shelf_life': 5,
                'price': 2.49,
                'cost': 1.25,
                'base_demand': 14,
                # Weekend cooking spike
                'day_pattern': {0: 1.0, 1: 0.9, 2: 1.0, 3: 1.0, 4: 1.1, 5: 1.2, 6: 1.1},
                'variability': 0.35,
            },
        }

    def generate_daily_demand(
        self,
        product: str,
        day_of_week: int,
        seasonal_factor: float = 1.0
    ) -> int:
        """
        Generate demand for a single product on a given day

        Args:
            product: Product name (e.g., 'berries')
            day_of_week: Day of week (0=Mon, 6=Sun)
            seasonal_factor: Multiplier for seasonal effects (e.g., 1.2 for holidays)

        Returns:
            Demand in units (integer)
        """
        if product not in self.products:
            raise ValueError(f"Unknown product: {product}")

        config = self.products[product]

        # Base demand
        base = config['base_demand']

        # Day-of-week pattern
        day_multiplier = config['day_pattern'].get(day_of_week, 1.0)

        # Random variability (normal distribution)
        noise = np.random.normal(1.0, config['variability'] / 2)
        noise = max(0.3, min(1.7, noise))  # Clip to reasonable range

        # Calculate final demand
        demand = base * day_multiplier * seasonal_factor * noise

        return max(0, int(round(demand)))

    def generate_historical_data(
        self,
        num_days: int = 90,
        start_date: str = None
    ) -> pd.DataFrame:
        """
        Generate historical demand data for all products

        Args:
            num_days: Number of days of historical data
            start_date: Start date (YYYY-MM-DD) or None for today - num_days

        Returns:
            DataFrame with columns: date, day_of_week, product, demand, price, revenue
        """
        if start_date is None:
            start = datetime.now() - timedelta(days=num_days)
        else:
            start = datetime.strptime(start_date, '%Y-%m-%d')

        records = []

        for day_offset in range(num_days):
            current_date = start + timedelta(days=day_offset)
            day_of_week = current_date.weekday()  # 0=Monday

            # Simple seasonal factor (slight increase over time for growth)
            seasonal_factor = 1.0 + (day_offset / num_days) * 0.1

            for product in self.products.keys():
                demand = self.generate_daily_demand(product, day_of_week, seasonal_factor)
                price = self.products[product]['price']
                revenue = demand * price

                records.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'day_of_week': day_of_week,
                    'day_name': current_date.strftime('%A'),
                    'product': product,
                    'demand': demand,
                    'price': price,
                    'revenue': round(revenue, 2),
                    'shelf_life': self.products[product]['shelf_life'],
                })

        df = pd.DataFrame(records)
        return df

    def calculate_demand_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics from historical data

        Args:
            df: DataFrame from generate_historical_data()

        Returns:
            Dictionary with statistics per product
        """
        stats = {}

        for product in self.products.keys():
            product_data = df[df['product'] == product]

            stats[product] = {
                'mean_demand': product_data['demand'].mean(),
                'std_demand': product_data['demand'].std(),
                'min_demand': product_data['demand'].min(),
                'max_demand': product_data['demand'].max(),
                'total_revenue': product_data['revenue'].sum(),
                'day_of_week_avg': product_data.groupby('day_of_week')['demand'].mean().to_dict(),
            }

        return stats

    def save_data(
        self,
        df: pd.DataFrame,
        output_dir: str = 'data/processed',
        filename: str = 'synthetic_demand.csv'
    ):
        """
        Save generated data to CSV

        Args:
            df: DataFrame to save
            output_dir: Output directory
            filename: Output filename
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"✓ Saved {len(df)} records to {filepath}")

    def save_demand_profiles(
        self,
        output_dir: str = 'data/processed',
        filename: str = 'demand_profiles.pkl'
    ):
        """
        Save demand profiles for quick loading

        Args:
            output_dir: Output directory
            filename: Output filename
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        profiles = {}
        for product, config in self.products.items():
            profiles[product] = {
                'base_demand': config['base_demand'],
                'day_pattern': config['day_pattern'],
                'shelf_life': config['shelf_life'],
                'price': config['price'],
                'cost': config['cost'],
            }

        with open(filepath, 'wb') as f:
            pickle.dump(profiles, f)

        print(f"✓ Saved demand profiles to {filepath}")

    def generate_and_save_all(
        self,
        num_days: int = 90,
        output_dir: str = 'data/processed'
    ):
        """
        Generate all data and save to files

        Args:
            num_days: Number of days of historical data
            output_dir: Output directory
        """
        print(f"Generating {num_days} days of synthetic demand data...")
        print(f"Random seed: {self.seed} (for reproducibility)")

        # Generate historical data
        df = self.generate_historical_data(num_days)

        # Calculate and print statistics
        stats = self.calculate_demand_statistics(df)

        print("\nDemand Statistics:")
        for product, product_stats in stats.items():
            print(f"\n{product.upper()}:")
            print(f"  Mean demand: {product_stats['mean_demand']:.1f} units/day")
            print(f"  Std dev: {product_stats['std_demand']:.1f}")
            print(f"  Range: {product_stats['min_demand']}-{product_stats['max_demand']}")
            print(f"  Total revenue: ${product_stats['total_revenue']:.2f}")

        # Save files
        self.save_data(df, output_dir)
        self.save_demand_profiles(output_dir)

        print(f"\n✅ Data generation complete!")

        return df, stats


def load_demand_profiles(filepath: str = 'data/processed/demand_profiles.pkl') -> Dict:
    """
    Load pre-generated demand profiles

    Args:
        filepath: Path to pickled profiles

    Returns:
        Dictionary of demand profiles
    """
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found, using defaults")
        # Return default profiles from generator
        generator = SyntheticDemandGenerator()
        return {
            product: {
                'base_demand': config['base_demand'],
                'day_pattern': config['day_pattern'],
            }
            for product, config in generator.products.items()
        }

    with open(filepath, 'rb') as f:
        profiles = pickle.load(f)

    print(f"✓ Loaded demand profiles from {filepath}")
    return profiles


if __name__ == "__main__":
    # Generate synthetic data
    print("="*60)
    print("  Diggyy Fresh - Synthetic Data Generator")
    print("="*60)

    generator = SyntheticDemandGenerator(seed=42)
    df, stats = generator.generate_and_save_all(num_days=90)

    print("\nSample data:")
    print(df.head(10))

    print("\n" + "="*60)
