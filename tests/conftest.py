"""
Pytest configuration and shared fixtures
"""

import pytest
import numpy as np


@pytest.fixture
def seed_random():
    """Fixture to set random seeds for reproducibility"""
    np.random.seed(42)
    yield
    # Reset after test
    np.random.seed(None)


@pytest.fixture
def sample_demand_profiles():
    """Fixture providing sample demand profiles for testing"""
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
    }
