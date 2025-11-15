"""
Instacart Data Loader and Processor
Processes 3M+ transactions to extract demand patterns
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class InstacartDataLoader:
    """Load and process Instacart dataset for grocery demand patterns"""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.orders = None
        self.order_products = None
        self.products = None
        self.aisles = None
        self.departments = None

        # Processed data
        self.demand_patterns = {}
        self.product_mapping = {}

    def load_raw_data(self):
        """Load raw Instacart CSV files"""
        print("Loading Instacart dataset...")

        try:
            # Load main datasets
            self.orders = pd.read_csv(os.path.join(self.data_dir, "orders.csv"))
            self.order_products = pd.read_csv(
                os.path.join(self.data_dir, "order_products__train.csv")
            )
            self.products = pd.read_csv(os.path.join(self.data_dir, "products.csv"))
            self.aisles = pd.read_csv(os.path.join(self.data_dir, "aisles.csv"))
            self.departments = pd.read_csv(os.path.join(self.data_dir, "departments.csv"))

            print(f"✓ Loaded {len(self.orders):,} orders")
            print(f"✓ Loaded {len(self.order_products):,} order-product pairs")
            print(f"✓ Loaded {len(self.products):,} products")

            return True

        except FileNotFoundError as e:
            print(f"✗ Error: Could not find Instacart CSV files in {self.data_dir}")
            print(f"  Expected files: orders.csv, order_products__train.csv, products.csv, aisles.csv, departments.csv")
            print(f"  Error: {e}")
            return False

    def identify_target_products(self, target_categories: List[str] = None) -> Dict:
        """
        Identify high-waste products matching our simulation

        Args:
            target_categories: List of product categories to focus on
                              e.g., ['berries', 'milk', 'bread', 'salad greens', 'tomatoes']

        Returns:
            Dictionary mapping simulation products to Instacart product IDs
        """

        if target_categories is None:
            target_categories = ['berries', 'milk', 'bread', 'salad', 'tomatoes']

        # Merge products with aisles and departments
        products_full = self.products.merge(self.aisles, on='aisle_id')
        products_full = products_full.merge(self.departments, on='department_id')

        mapping = {}

        for category in target_categories:
            # Search in product names and aisle names
            matches = products_full[
                products_full['product_name'].str.lower().str.contains(category, na=False) |
                products_full['aisle'].str.lower().str.contains(category, na=False)
            ]

            if len(matches) > 0:
                # Get most popular products in this category
                product_ids = matches['product_id'].tolist()

                # Count orders for these products
                product_counts = self.order_products[
                    self.order_products['product_id'].isin(product_ids)
                ]['product_id'].value_counts()

                if len(product_counts) > 0:
                    top_product_id = product_counts.index[0]
                    top_product = matches[matches['product_id'] == top_product_id].iloc[0]

                    mapping[category] = {
                        'product_id': top_product_id,
                        'product_name': top_product['product_name'],
                        'aisle': top_product['aisle'],
                        'order_count': int(product_counts.iloc[0]),
                    }

        print(f"\nIdentified {len(mapping)} target products:")
        for cat, info in mapping.items():
            print(f"  {cat}: {info['product_name']} ({info['order_count']} orders)")

        self.product_mapping = mapping
        return mapping

    def extract_demand_patterns(self, product_ids: List[int]) -> Dict:
        """
        Extract demand patterns from Instacart data

        Args:
            product_ids: List of product IDs to analyze

        Returns:
            Dictionary with demand patterns by day of week
        """

        print("\nExtracting demand patterns...")

        # Merge orders with products
        data = self.order_products[
            self.order_products['product_id'].isin(product_ids)
        ].merge(
            self.orders[['order_id', 'order_dow', 'order_hour_of_day']],
            on='order_id'
        )

        demand_patterns = {}

        for product_id in product_ids:
            product_data = data[data['product_id'] == product_id]

            if len(product_data) == 0:
                continue

            # Group by day of week
            dow_counts = product_data.groupby('order_dow').size()

            # Normalize to get relative demand
            total = dow_counts.sum()
            dow_pattern = (dow_counts / total * 7).to_dict()  # Scale to avg=1.0

            # Fill missing days
            for dow in range(7):
                if dow not in dow_pattern:
                    dow_pattern[dow] = 1.0

            demand_patterns[product_id] = {
                'day_of_week_pattern': dow_pattern,
                'total_orders': int(total),
                'avg_daily_demand': total / 7,
            }

        self.demand_patterns = demand_patterns
        print(f"✓ Extracted patterns for {len(demand_patterns)} products")

        return demand_patterns

    def get_demand_multiplier(self, product_id: int, day_of_week: int) -> float:
        """
        Get demand multiplier for a product on a given day

        Args:
            product_id: Instacart product ID
            day_of_week: Day of week (0=Monday, 6=Sunday)

        Returns:
            Demand multiplier (1.0 = average)
        """

        if product_id not in self.demand_patterns:
            return 1.0

        pattern = self.demand_patterns[product_id]['day_of_week_pattern']
        return pattern.get(day_of_week, 1.0)

    def get_base_demand(self, product_id: int, scale: float = 1.0) -> float:
        """
        Get base daily demand for a product

        Args:
            product_id: Instacart product ID
            scale: Scaling factor to adjust to simulation size

        Returns:
            Base daily demand in units
        """

        if product_id not in self.demand_patterns:
            return 15.0  # Default

        avg_demand = self.demand_patterns[product_id]['avg_daily_demand']
        return avg_demand * scale

    def create_demand_profile(self, simulation_product: str, scale: float = 0.01) -> Dict:
        """
        Create complete demand profile for simulation product

        Args:
            simulation_product: Product name in simulation (e.g., 'berries')
            scale: Scale factor to convert Instacart demand to simulation units

        Returns:
            Demand profile with base demand and day-of-week patterns
        """

        if simulation_product not in self.product_mapping:
            print(f"Warning: No mapping for {simulation_product}, using defaults")
            return {
                'base_demand': 15.0,
                'day_pattern': {i: 1.0 for i in range(7)},
            }

        product_id = self.product_mapping[simulation_product]['product_id']

        if product_id not in self.demand_patterns:
            print(f"Warning: No demand pattern for {simulation_product}, using defaults")
            return {
                'base_demand': 15.0,
                'day_pattern': {i: 1.0 for i in range(7)},
            }

        pattern = self.demand_patterns[product_id]

        return {
            'base_demand': pattern['avg_daily_demand'] * scale,
            'day_pattern': pattern['day_of_week_pattern'],
            'instacart_product': self.product_mapping[simulation_product]['product_name'],
        }

    def process_all(self, target_categories: List[str] = None) -> Dict:
        """
        Full pipeline: load data, identify products, extract patterns

        Args:
            target_categories: Categories to focus on

        Returns:
            Complete demand profiles for all simulation products
        """

        if not self.load_raw_data():
            print("\nUsing placeholder demand patterns (Instacart data not available)")
            return self._get_placeholder_patterns()

        # Identify target products
        self.identify_target_products(target_categories)

        # Extract demand patterns
        product_ids = [info['product_id'] for info in self.product_mapping.values()]
        self.extract_demand_patterns(product_ids)

        # Create demand profiles
        profiles = {}
        for category in self.product_mapping.keys():
            profiles[category] = self.create_demand_profile(category)

        print("\n✓ Demand profiles created for all products")
        return profiles

    def _get_placeholder_patterns(self) -> Dict:
        """Fallback placeholder patterns if Instacart data not available"""
        return {
            'berries': {
                'base_demand': 15,
                'day_pattern': {0: 0.9, 1: 0.8, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.3, 6: 1.2},
            },
            'milk': {
                'base_demand': 20,
                'day_pattern': {0: 1.0, 1: 0.9, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.2, 6: 1.1},
            },
            'bread': {
                'base_demand': 18,
                'day_pattern': {0: 1.0, 1: 0.9, 2: 1.0, 3: 1.0, 4: 1.1, 5: 1.3, 6: 1.2},
            },
            'leafy_greens': {
                'base_demand': 12,
                'day_pattern': {0: 0.9, 1: 0.8, 2: 0.9, 3: 1.0, 4: 1.2, 5: 1.3, 6: 1.1},
            },
            'tomatoes': {
                'base_demand': 14,
                'day_pattern': {0: 1.0, 1: 0.9, 2: 1.0, 3: 1.0, 4: 1.1, 5: 1.2, 6: 1.1},
            },
        }


if __name__ == "__main__":
    # Test data loader
    print("Testing Instacart data loader...")

    loader = InstacartDataLoader()
    profiles = loader.process_all()

    print("\nDemand Profiles:")
    for product, profile in profiles.items():
        print(f"\n{product}:")
        print(f"  Base demand: {profile['base_demand']:.1f} units/day")
        print(f"  Day patterns: {profile['day_pattern']}")
