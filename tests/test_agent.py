"""
Tests for GroceryAgent and baseline policy
"""

import pytest
import numpy as np
import os
import tempfile
from src.environment import GroceryInventoryEnv
from src.agent import GroceryAgent, BaselinePolicy


class TestBaselinePolicy:
    """Test suite for baseline policy"""

    def test_baseline_initialization(self):
        """Test baseline policy initialization"""
        baseline = BaselinePolicy(order_quantity=25)
        assert baseline.order_quantity == 25

    def test_baseline_prediction(self):
        """Test that baseline returns fixed quantities"""
        baseline = BaselinePolicy(order_quantity=20)

        # Create dummy observation for 5 products
        # State: 5*4 + 3 = 23 dimensions
        obs = np.random.rand(23)

        action, _ = baseline.predict(obs)

        # Should return 5 actions (one per product)
        assert len(action) == 5

        # All actions should be 20
        np.testing.assert_array_equal(action, np.array([20, 20, 20, 20, 20]))

    def test_baseline_different_sizes(self):
        """Test baseline with different numbers of products"""
        baseline = BaselinePolicy(order_quantity=15)

        # 3 products: 3*4 + 3 = 15 dimensions
        obs_3_products = np.random.rand(15)
        action, _ = baseline.predict(obs_3_products)
        assert len(action) == 3
        np.testing.assert_array_equal(action, np.array([15, 15, 15]))


class TestGroceryAgent:
    """Test suite for RL agent"""

    def test_agent_initialization(self):
        """Test agent initialization"""
        env = GroceryInventoryEnv(num_products=3)
        agent = GroceryAgent(env)

        assert agent.env == env
        assert agent.model is None  # Model not created yet

    def test_model_creation(self):
        """Test PPO model creation"""
        env = GroceryInventoryEnv(num_products=3)
        agent = GroceryAgent(env)

        agent.create_model()

        assert agent.model is not None
        assert hasattr(agent.model, 'learn')
        assert hasattr(agent.model, 'predict')

    def test_short_training(self):
        """Test that agent can train for a few steps"""
        env = GroceryInventoryEnv(num_products=3, episode_length=10)
        agent = GroceryAgent(env)

        # Very short training (just to verify it doesn't crash)
        results = agent.train(total_timesteps=100)

        assert 'episode_rewards' in results
        assert 'revenues' in results
        assert 'spoilage_costs' in results

    def test_model_save_and_load(self):
        """Test model saving and loading"""
        env = GroceryInventoryEnv(num_products=3, episode_length=10)
        agent = GroceryAgent(env)

        # Train briefly
        agent.train(total_timesteps=100)

        # Save to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model")
            agent.save_model(model_path)

            # Check file exists
            assert os.path.exists(f"{model_path}.zip")

            # Load in new agent
            new_agent = GroceryAgent(env)
            new_agent.load_model(model_path)

            assert new_agent.model is not None

    def test_prediction_after_training(self):
        """Test that agent can make predictions after training"""
        env = GroceryInventoryEnv(num_products=3, episode_length=10)
        agent = GroceryAgent(env)

        # Train briefly
        agent.train(total_timesteps=100)

        # Reset environment and get observation
        obs, _ = env.reset(seed=42)

        # Make prediction
        action = agent.predict(obs, deterministic=True)

        # Check action shape and range
        assert action.shape == (3,)
        assert np.all(action >= 0)
        assert np.all(action <= 100)

    def test_prediction_without_training_raises_error(self):
        """Test that prediction without training raises error"""
        env = GroceryInventoryEnv(num_products=3)
        agent = GroceryAgent(env)

        obs, _ = env.reset()

        with pytest.raises(ValueError, match="Model not trained or loaded"):
            agent.predict(obs)

    def test_evaluation(self):
        """Test agent evaluation"""
        env = GroceryInventoryEnv(num_products=3, episode_length=10)
        agent = GroceryAgent(env)

        # Train briefly
        agent.train(total_timesteps=100)

        # Evaluate
        results = agent.evaluate(n_episodes=2)

        assert 'mean_reward' in results
        assert 'std_reward' in results
        assert isinstance(results['mean_reward'], (int, float))

    def test_custom_hyperparameters(self):
        """Test agent with custom hyperparameters"""
        env = GroceryInventoryEnv(num_products=3)
        agent = GroceryAgent(
            env,
            learning_rate=1e-3,
            n_steps=128,
            batch_size=32
        )

        assert agent.learning_rate == 1e-3
        assert agent.n_steps == 128
        assert agent.batch_size == 32

    def test_agent_environment_compatibility(self):
        """Test that agent works with different environment sizes"""
        # Test with 3 products
        env_3 = GroceryInventoryEnv(num_products=3, episode_length=10)
        agent_3 = GroceryAgent(env_3)
        agent_3.train(total_timesteps=100)

        obs_3, _ = env_3.reset()
        action_3 = agent_3.predict(obs_3)
        assert action_3.shape == (3,)

        # Test with 5 products
        env_5 = GroceryInventoryEnv(num_products=5, episode_length=10)
        agent_5 = GroceryAgent(env_5)
        agent_5.train(total_timesteps=100)

        obs_5, _ = env_5.reset()
        action_5 = agent_5.predict(obs_5)
        assert action_5.shape == (5,)
