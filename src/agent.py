"""
RL Agent for Grocery Inventory Management
Implements PPO training and baseline comparison
"""

import os
import numpy as np
from typing import Dict, List, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

from src.environment import GroceryInventoryEnv


class TrainingCallback(BaseCallback):
    """Custom callback for tracking training metrics"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.spoilage_costs = []
        self.revenues = []

    def _on_step(self) -> bool:
        # Check if episode finished
        if self.locals.get('dones'):
            for idx, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][idx]
                    self.episode_rewards.append(self.locals['rewards'][idx])
                    self.revenues.append(info.get('total_revenue', 0))
                    self.spoilage_costs.append(info.get('total_spoilage', 0))

                    if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                        print(f"Episode {len(self.episode_rewards)}: "
                              f"Revenue=${info.get('total_revenue', 0):.2f}, "
                              f"Spoilage=${info.get('total_spoilage', 0):.2f}, "
                              f"Stockouts={info.get('total_stockouts', 0)}")
        return True


class BaselinePolicy:
    """Simple baseline policy using fixed order quantity"""

    def __init__(self, order_quantity: float = 20):
        self.order_quantity = order_quantity

    def predict(self, observation, deterministic=True):
        """Return fixed order quantity for all products"""
        # Observation shape determines number of products
        # State: num_products*4 + 3, so num_products = (obs_dim - 3) / 4
        obs_dim = len(observation) if hasattr(observation, '__len__') else observation.shape[-1]
        num_products = (obs_dim - 3) // 4

        action = np.full(num_products, self.order_quantity, dtype=np.float32)
        return action, None


class GroceryAgent:
    """RL Agent wrapper for training and inference"""

    def __init__(
        self,
        env: gym.Env,
        model_path: str = "models/ppo_grocery",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
    ):
        self.env = env
        self.model_path = model_path
        self.model = None

        # PPO hyperparameters optimized for CPU
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size

    def create_model(self):
        """Initialize PPO model"""
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Encourage exploration
            verbose=1,
            tensorboard_log="./logs/",
        )
        print("PPO model created")

    def train(self, total_timesteps: int = 50000, save_freq: int = 10000):
        """Train the PPO agent"""
        if self.model is None:
            self.create_model()

        print(f"\nTraining PPO agent for {total_timesteps} timesteps...")

        # Setup callbacks
        callback = TrainingCallback(verbose=1)

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )

        # Save final model
        self.save_model()

        print(f"\nTraining complete! Model saved to {self.model_path}")

        return {
            'episode_rewards': callback.episode_rewards,
            'revenues': callback.revenues,
            'spoilage_costs': callback.spoilage_costs,
        }

    def save_model(self, path: str = None):
        """Save trained model"""
        if path is None:
            path = self.model_path

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str = None):
        """Load trained model"""
        if path is None:
            path = self.model_path

        if not os.path.exists(f"{path}.zip"):
            raise FileNotFoundError(f"Model not found at {path}.zip")

        # Wrap environment in VecEnv for compatibility
        vec_env = DummyVecEnv([lambda: self.env])
        self.model = PPO.load(path, env=vec_env)
        print(f"Model loaded from {path}")

    def predict(self, observation, deterministic: bool = True):
        """Make prediction using trained model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        action, *_ = self.model.predict(observation, deterministic=deterministic)

        # Ensure action has the correct shape (num_products,)
        # Remove any batch dimensions if present
        action = np.atleast_1d(action).flatten()

        # Verify action has correct number of elements
        expected_size = self.env.action_space.shape[0]
        if action.size != expected_size:
            raise ValueError(f"Action size mismatch: got {action.size}, expected {expected_size}")

        return action

    def evaluate(self, n_episodes: int = 10) -> Dict:
        """Evaluate agent performance"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.env,
            n_eval_episodes=n_episodes,
            deterministic=True,
        )

        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
        }


def compare_policies(env: gym.Env, rl_agent: GroceryAgent, baseline: BaselinePolicy, n_episodes: int = 10):
    """Compare RL agent vs baseline policy"""

    print(f"\nComparing RL agent vs Baseline over {n_episodes} episodes...")

    results = {
        'rl': {'rewards': [], 'revenues': [], 'spoilage': [], 'stockouts': []},
        'baseline': {'rewards': [], 'revenues': [], 'spoilage': [], 'stockouts': []},
    }

    # Evaluate RL agent
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = rl_agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        results['rl']['rewards'].append(episode_reward)
        results['rl']['revenues'].append(info['total_revenue'])
        results['rl']['spoilage'].append(info['total_spoilage'])
        results['rl']['stockouts'].append(info['total_stockouts'])

    # Evaluate baseline
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = baseline.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        results['baseline']['rewards'].append(episode_reward)
        results['baseline']['revenues'].append(info['total_revenue'])
        results['baseline']['spoilage'].append(info['total_spoilage'])
        results['baseline']['stockouts'].append(info['total_stockouts'])

    # Calculate statistics
    print("\n=== Performance Comparison ===")
    for policy_name in ['rl', 'baseline']:
        data = results[policy_name]
        print(f"\n{policy_name.upper()}:")
        print(f"  Avg Reward: {np.mean(data['rewards']):.2f} ± {np.std(data['rewards']):.2f}")
        print(f"  Avg Revenue: ${np.mean(data['revenues']):.2f}")
        print(f"  Avg Spoilage: ${np.mean(data['spoilage']):.2f}")
        print(f"  Avg Stockouts: {np.mean(data['stockouts']):.0f} units")

    # Calculate waste reduction
    rl_spoilage = np.mean(results['rl']['spoilage'])
    baseline_spoilage = np.mean(results['baseline']['spoilage'])
    waste_reduction = (baseline_spoilage - rl_spoilage) / baseline_spoilage * 100

    print(f"\nWaste Reduction: {waste_reduction:.1f}%")

    return results


if __name__ == "__main__":
    # Quick training example
    print("Creating environment...")
    env = GroceryInventoryEnv(num_products=5, episode_length=30)

    print("Initializing RL agent...")
    agent = GroceryAgent(env)

    print("Training agent...")
    training_stats = agent.train(total_timesteps=50000)

    print("\nEvaluating agent...")
    eval_stats = agent.evaluate(n_episodes=10)
    print(f"Evaluation: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")

    print("\nComparing with baseline...")
    baseline = BaselinePolicy(order_quantity=20)
    comparison_results = compare_policies(env, agent, baseline, n_episodes=10)

    print("\nDone! Model saved and ready for deployment.")
