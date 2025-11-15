"""
Gymnasium Environment for Grocery Inventory Management
Simulates multi-product inventory with spoilage, demand, and ordering decisions
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, List, Optional


class GroceryInventoryEnv(gym.Env):
    """
    Grocery inventory environment with RL-based ordering

    State: [inventory_age_bins (4 per product), day_of_week, sales_velocity, weather_signal]
    Action: order_quantity per product (continuous 0-100 units)
    Reward: revenue - 2.5*spoilage_cost - 3*stockout_penalty + freshness_bonus
    """

    metadata = {'render_modes': ['human']}

    # Product catalog with shelf life (days) and prices
    PRODUCTS = {
        'berries': {'shelf_life': 3, 'price': 4.99, 'cost': 2.50},
        'milk': {'shelf_life': 7, 'price': 3.49, 'cost': 1.75},
        'bread': {'shelf_life': 5, 'price': 2.99, 'cost': 1.50},
        'leafy_greens': {'shelf_life': 4, 'price': 3.99, 'cost': 2.00},
        'tomatoes': {'shelf_life': 5, 'price': 2.49, 'cost': 1.25},
    }

    def __init__(
        self,
        num_products: int = 5,
        max_inventory: int = 100,
        episode_length: int = 30,
        demand_profiles: Optional[Dict] = None
    ):
        super().__init__()

        self.num_products = num_products
        self.max_inventory = max_inventory
        self.episode_length = episode_length
        self.product_names = list(self.PRODUCTS.keys())[:num_products]

        # State space: 4 age bins per product + day_of_week + sales_velocity + weather
        # Age bins: [0-25%, 25-50%, 50-75%, 75-100%] of shelf life
        state_dim = num_products * 4 + 3  # 4 bins per product + 3 scalar features
        self.observation_space = spaces.Box(
            low=0, high=max_inventory, shape=(state_dim,), dtype=np.float32
        )

        # Action space: order quantity for each product (continuous 0-100)
        self.action_space = spaces.Box(
            low=0, high=100, shape=(num_products,), dtype=np.float32
        )

        # Internal state
        self.current_step = 0
        self.day_of_week = 0
        self.inventory = {}  # {product: [(quantity, age), ...]}
        self.sales_history = []
        self.total_revenue = 0
        self.total_spoilage = 0
        self.total_stockouts = 0

        # Load demand patterns (synthetic or from data loader)
        self.demand_profiles = demand_profiles or self._get_default_profiles()
        self.base_demand = {
            product: self.demand_profiles[product]['base_demand']
            for product in self.product_names
        }
        self.demand_variability = 0.3

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)

        self.current_step = 0
        self.day_of_week = 0
        self.total_revenue = 0
        self.total_spoilage = 0
        self.total_stockouts = 0
        self.sales_history = []

        # Initialize inventory with some starting stock
        self.inventory = {
            product: [(np.random.randint(10, 30), 0)]
            for product in self.product_names
        }

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step"""
        # Clip actions to valid range
        action = np.clip(action, 0, 100)

        # 1. Receive new inventory (ordered yesterday)
        self._receive_order(action)

        # 2. Age existing inventory
        self._age_inventory()

        # 3. Remove spoiled items (FIFO)
        spoilage_cost = self._remove_spoiled()

        # 4. Simulate customer demand
        revenue, stockout_penalty = self._simulate_demand()

        # 5. Calculate freshness bonus
        freshness_bonus = self._calculate_freshness_bonus()

        # 6. Calculate reward
        reward = revenue - 2.5 * spoilage_cost - 3 * stockout_penalty + freshness_bonus

        # 7. Update state
        self.current_step += 1
        self.day_of_week = (self.day_of_week + 1) % 7
        self.total_revenue += revenue
        self.total_spoilage += spoilage_cost

        # Check termination
        terminated = self.current_step >= self.episode_length
        truncated = False

        observation = self._get_observation()
        info = self._get_info()
        info['revenue'] = revenue
        info['spoilage_cost'] = spoilage_cost
        info['stockout_penalty'] = stockout_penalty
        info['freshness_bonus'] = freshness_bonus

        return observation, reward, terminated, truncated, info

    def _receive_order(self, action: np.ndarray):
        """Add new inventory from order (age 0)"""
        for i, product in enumerate(self.product_names):
            quantity = int(action[i])
            if quantity > 0:
                if product not in self.inventory:
                    self.inventory[product] = []
                self.inventory[product].append((quantity, 0))

    def _age_inventory(self):
        """Age all inventory by 1 day"""
        for product in self.product_names:
            if product in self.inventory:
                self.inventory[product] = [
                    (qty, age + 1) for qty, age in self.inventory[product]
                ]

    def _remove_spoiled(self) -> float:
        """Remove spoiled inventory (FIFO) and return cost"""
        spoilage_cost = 0

        for product in self.product_names:
            if product not in self.inventory:
                continue

            shelf_life = self.PRODUCTS[product]['shelf_life']
            cost = self.PRODUCTS[product]['cost']

            # Remove items that exceeded shelf life
            non_spoiled = []
            for qty, age in self.inventory[product]:
                if age > shelf_life:
                    spoilage_cost += qty * cost
                else:
                    non_spoiled.append((qty, age))

            self.inventory[product] = non_spoiled

        return spoilage_cost

    def _simulate_demand(self) -> Tuple[float, float]:
        """Simulate customer demand and calculate revenue & stockout penalty"""
        revenue = 0
        stockout_penalty = 0
        daily_sales = {}

        for product in self.product_names:
            # Generate demand using realistic patterns from demand profiles
            base = self.base_demand[product]

            # Day-of-week pattern (product-specific from demand profiles)
            day_pattern = self.demand_profiles[product]['day_pattern']
            day_multiplier = day_pattern.get(self.day_of_week, 1.0)

            # Add randomness to simulate real-world variability
            noise = np.random.uniform(1 - self.demand_variability, 1 + self.demand_variability)
            demand = int(base * day_multiplier * noise)

            # Fulfill demand using FIFO
            fulfilled = 0
            remaining_demand = demand

            if product in self.inventory and self.inventory[product]:
                updated_inventory = []

                for qty, age in self.inventory[product]:
                    if remaining_demand <= 0:
                        updated_inventory.append((qty, age))
                    elif qty <= remaining_demand:
                        # Consume entire batch
                        fulfilled += qty
                        remaining_demand -= qty
                    else:
                        # Partial consumption
                        fulfilled += remaining_demand
                        updated_inventory.append((qty - remaining_demand, age))
                        remaining_demand = 0

                self.inventory[product] = updated_inventory

            # Calculate revenue and stockout penalty
            price = self.PRODUCTS[product]['price']
            revenue += fulfilled * price

            if remaining_demand > 0:
                # Penalty for lost sales
                stockout_penalty += remaining_demand * price * 0.5
                self.total_stockouts += remaining_demand

            daily_sales[product] = fulfilled

        self.sales_history.append(daily_sales)
        return revenue, stockout_penalty

    def _calculate_freshness_bonus(self) -> float:
        """Reward for maintaining fresh inventory"""
        bonus = 0

        for product in self.product_names:
            if product not in self.inventory or not self.inventory[product]:
                continue

            shelf_life = self.PRODUCTS[product]['shelf_life']
            total_qty = sum(qty for qty, _ in self.inventory[product])

            if total_qty == 0:
                continue

            # Calculate average freshness (0-1, where 1 is brand new)
            avg_freshness = sum(
                qty * max(0, 1 - age / shelf_life)
                for qty, age in self.inventory[product]
            ) / total_qty

            bonus += avg_freshness * 2  # Small bonus for freshness

        return bonus

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector"""
        obs = []

        # Inventory age bins for each product (4 bins per product)
        for product in self.product_names:
            shelf_life = self.PRODUCTS[product]['shelf_life']
            bins = [0, 0, 0, 0]  # [0-25%, 25-50%, 50-75%, 75-100%]

            if product in self.inventory:
                for qty, age in self.inventory[product]:
                    freshness = max(0, 1 - age / shelf_life)
                    bin_idx = min(3, int(freshness * 4))
                    bins[3 - bin_idx] += qty  # Reverse so older items in higher bins

            obs.extend(bins)

        # Day of week (normalized 0-1)
        obs.append(self.day_of_week / 7.0)

        # Recent sales velocity (last 7 days average)
        if len(self.sales_history) > 0:
            recent_sales = self.sales_history[-7:]
            avg_sales = np.mean([sum(day.values()) for day in recent_sales])
            obs.append(avg_sales / 100.0)  # Normalize
        else:
            obs.append(0.5)

        # Weather signal (placeholder - random for now)
        obs.append(np.random.uniform(0, 1))

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict:
        """Return additional info"""
        total_inventory = sum(
            sum(qty for qty, _ in items)
            for items in self.inventory.values()
        )

        return {
            'step': self.current_step,
            'day_of_week': self.day_of_week,
            'total_inventory': total_inventory,
            'total_revenue': self.total_revenue,
            'total_spoilage': self.total_spoilage,
            'total_stockouts': self.total_stockouts,
        }

    def _get_default_profiles(self) -> Dict:
        """
        Get default synthetic demand profiles with realistic patterns

        These patterns are based on typical grocery shopping behavior:
        - Weekends have higher demand (people shop for the week)
        - Different products have different patterns
        - Berries: Peak Friday-Sunday (weekend consumption)
        - Milk: Steady with weekend spike (staple item)
        - Bread: Strong weekend pattern (baking, sandwiches)
        - Leafy greens: Thu-Sat peak (health-conscious, meal prep)
        - Tomatoes: Weekend cooking spike
        """
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

    def render(self):
        """Render environment state"""
        if self.render_mode == 'human':
            print(f"\n=== Day {self.current_step} (DoW: {self.day_of_week}) ===")
            for product in self.product_names:
                if product in self.inventory:
                    total = sum(qty for qty, _ in self.inventory[product])
                    ages = [f"{qty}@{age}d" for qty, age in self.inventory[product]]
                    print(f"{product}: {total} units ({', '.join(ages)})")
                else:
                    print(f"{product}: 0 units")
            print(f"Revenue: ${self.total_revenue:.2f}, Spoilage: ${self.total_spoilage:.2f}, Stockouts: {self.total_stockouts}")
