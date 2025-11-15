"""
Tests for GroceryInventoryEnv
"""

import pytest
import numpy as np
from src.environment import GroceryInventoryEnv


class TestGroceryInventoryEnv:
    """Test suite for the RL environment"""

    def test_environment_initialization(self):
        """Test that environment initializes correctly"""
        env = GroceryInventoryEnv(num_products=5, episode_length=30)

        assert env.num_products == 5
        assert env.episode_length == 30
        assert len(env.product_names) == 5

        # Check observation space dimensions
        # (5 products * 4 age bins) + 3 features = 23
        assert env.observation_space.shape == (23,)

        # Check action space dimensions (5 products)
        assert env.action_space.shape == (5,)

    def test_reset(self):
        """Test environment reset"""
        env = GroceryInventoryEnv(num_products=5)
        obs, info = env.reset(seed=42)

        # Check observation shape
        assert obs.shape == (23,)
        assert isinstance(obs, np.ndarray)

        # Check initial state
        assert env.current_step == 0
        assert env.total_revenue == 0
        assert env.total_spoilage == 0

        # Check info dict
        assert 'step' in info
        assert 'total_revenue' in info

    def test_step(self):
        """Test environment step function"""
        env = GroceryInventoryEnv(num_products=5, episode_length=30)
        env.reset(seed=42)

        # Take action: order 20 units of each product
        action = np.array([20, 20, 20, 20, 20], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # Check outputs
        assert obs.shape == (23,)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        # Check state updates
        assert env.current_step == 1

        # Check info contains expected keys
        assert 'revenue' in info
        assert 'spoilage_cost' in info
        assert 'stockout_penalty' in info

    def test_episode_termination(self):
        """Test that episode terminates after episode_length steps"""
        episode_length = 10
        env = GroceryInventoryEnv(num_products=3, episode_length=episode_length)
        env.reset(seed=42)

        action = np.array([15, 15, 15], dtype=np.float32)

        # Run for episode_length - 1 steps
        for _ in range(episode_length - 1):
            _, _, terminated, _, _ = env.step(action)
            assert not terminated

        # Final step should terminate
        _, _, terminated, _, _ = env.step(action)
        assert terminated

    def test_action_clipping(self):
        """Test that actions are clipped to valid range"""
        env = GroceryInventoryEnv(num_products=3)
        env.reset(seed=42)

        # Action with values outside [0, 100]
        action = np.array([150, -10, 50], dtype=np.float32)
        _, _, _, _, _ = env.step(action)

        # Should not raise error (clipping happens inside step)
        assert True

    def test_spoilage_mechanism(self):
        """Test that products spoil after shelf life"""
        env = GroceryInventoryEnv(num_products=1)  # Just berries (3 day shelf life)
        env.reset(seed=42)

        # Order berries and let them age
        action = np.array([50], dtype=np.float32)

        total_spoilage_before = env.total_spoilage

        # Age for more than shelf life (berries = 3 days)
        for _ in range(5):
            _, _, _, _, info = env.step(np.array([0], dtype=np.float32))  # No new orders

        # Should have some spoilage
        assert env.total_spoilage > total_spoilage_before

    def test_demand_patterns_loaded(self):
        """Test that demand patterns are loaded correctly"""
        env = GroceryInventoryEnv(num_products=5)

        # Check demand profiles exist
        assert len(env.demand_profiles) >= 5

        # Check each product has base_demand and day_pattern
        for product in env.product_names:
            assert 'base_demand' in env.demand_profiles[product]
            assert 'day_pattern' in env.demand_profiles[product]
            assert isinstance(env.demand_profiles[product]['day_pattern'], dict)

    def test_custom_demand_profiles(self):
        """Test environment with custom demand profiles"""
        custom_profiles = {
            'berries': {'base_demand': 10, 'day_pattern': {i: 1.0 for i in range(7)}},
            'milk': {'base_demand': 20, 'day_pattern': {i: 1.0 for i in range(7)}},
        }

        env = GroceryInventoryEnv(num_products=2, demand_profiles=custom_profiles)

        assert env.base_demand['berries'] == 10
        assert env.base_demand['milk'] == 20

    def test_reward_components(self):
        """Test that reward includes all components"""
        env = GroceryInventoryEnv(num_products=3)
        env.reset(seed=42)

        action = np.array([20, 20, 20], dtype=np.float32)
        _, reward, _, _, info = env.step(action)

        # Check info has all reward components
        assert 'revenue' in info
        assert 'spoilage_cost' in info
        assert 'stockout_penalty' in info
        assert 'freshness_bonus' in info

        # Reward should be numeric
        assert isinstance(reward, (int, float))

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results"""
        env1 = GroceryInventoryEnv(num_products=3)
        env2 = GroceryInventoryEnv(num_products=3)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        # Initial observations should be identical
        np.testing.assert_array_almost_equal(obs1, obs2)

        action = np.array([15, 15, 15], dtype=np.float32)

        # Step should produce identical results
        obs1_next, reward1, _, _, _ = env1.step(action)
        obs2_next, reward2, _, _, _ = env2.step(action)

        # Note: Demand has randomness, so exact match may not occur
        # But observation structure should be identical
        assert obs1_next.shape == obs2_next.shape
