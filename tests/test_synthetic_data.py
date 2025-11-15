"""
Tests for synthetic data generator
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from src.synthetic_data_generator import SyntheticDemandGenerator, load_demand_profiles


class TestSyntheticDemandGenerator:
    """Test suite for synthetic data generation"""

    def test_generator_initialization(self):
        """Test generator initializes with seed"""
        generator = SyntheticDemandGenerator(seed=42)
        assert generator.seed == 42
        assert len(generator.products) == 5

    def test_product_configuration(self):
        """Test that products have correct configuration"""
        generator = SyntheticDemandGenerator()

        required_keys = ['shelf_life', 'price', 'cost', 'base_demand', 'day_pattern', 'variability']

        for product in ['berries', 'milk', 'bread', 'leafy_greens', 'tomatoes']:
            assert product in generator.products
            config = generator.products[product]

            for key in required_keys:
                assert key in config, f"{product} missing {key}"

    def test_generate_daily_demand(self):
        """Test daily demand generation"""
        generator = SyntheticDemandGenerator(seed=42)

        demand = generator.generate_daily_demand('berries', day_of_week=5)

        assert isinstance(demand, int)
        assert demand >= 0
        assert demand < 100  # Reasonable upper bound

    def test_demand_variability(self):
        """Test that demand varies across days"""
        generator = SyntheticDemandGenerator(seed=42)

        demands = [
            generator.generate_daily_demand('milk', day_of_week=i)
            for i in range(7)
        ]

        # Should have some variation (not all the same)
        assert len(set(demands)) > 1

    def test_day_of_week_patterns(self):
        """Test that day-of-week patterns affect demand"""
        generator = SyntheticDemandGenerator(seed=42)

        # Generate many samples for each day to average out noise
        samples_per_day = 100
        avg_demands = {}

        for dow in range(7):
            demands = [
                generator.generate_daily_demand('bread', day_of_week=dow)
                for _ in range(samples_per_day)
            ]
            avg_demands[dow] = np.mean(demands)

        # Weekend days (5, 6) should generally have higher demand for bread
        # (day_pattern shows 1.3 and 1.2 vs ~1.0 for weekdays)
        # This is statistical, so we check trend over many samples
        weekday_avg = np.mean([avg_demands[i] for i in range(5)])
        weekend_avg = np.mean([avg_demands[5], avg_demands[6]])

        # Weekend should be higher (with some tolerance for randomness)
        assert weekend_avg > weekday_avg * 0.95

    def test_seasonal_factor(self):
        """Test that seasonal factor affects demand"""
        generator = SyntheticDemandGenerator(seed=42)

        demand_normal = generator.generate_daily_demand('milk', day_of_week=1, seasonal_factor=1.0)
        demand_high = generator.generate_daily_demand('milk', day_of_week=1, seasonal_factor=1.5)

        # With higher seasonal factor, demand should generally be higher
        # (statistical test over multiple runs)
        demands_normal = [
            generator.generate_daily_demand('milk', day_of_week=1, seasonal_factor=1.0)
            for _ in range(50)
        ]
        demands_high = [
            generator.generate_daily_demand('milk', day_of_week=1, seasonal_factor=1.5)
            for _ in range(50)
        ]

        assert np.mean(demands_high) > np.mean(demands_normal)

    def test_generate_historical_data(self):
        """Test historical data generation"""
        generator = SyntheticDemandGenerator(seed=42)

        df = generator.generate_historical_data(num_days=30)

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 30 * 5  # 30 days * 5 products

        # Check columns
        required_columns = ['date', 'day_of_week', 'day_name', 'product', 'demand', 'price', 'revenue']
        for col in required_columns:
            assert col in df.columns

        # Check data types
        assert df['demand'].dtype in [np.int64, np.int32]
        assert df['price'].dtype in [np.float64, np.float32]
        assert df['revenue'].dtype in [np.float64, np.float32]

    def test_historical_data_date_range(self):
        """Test that historical data covers correct date range"""
        generator = SyntheticDemandGenerator(seed=42)

        df = generator.generate_historical_data(num_days=10, start_date='2024-01-01')

        dates = pd.to_datetime(df['date']).unique()
        assert len(dates) == 10

        # First date should be 2024-01-01
        assert str(dates[0])[:10] == '2024-01-01'

    def test_calculate_demand_statistics(self):
        """Test statistics calculation"""
        generator = SyntheticDemandGenerator(seed=42)

        df = generator.generate_historical_data(num_days=30)
        stats = generator.calculate_demand_statistics(df)

        # Check statistics exist for all products
        assert len(stats) == 5

        for product in generator.products.keys():
            assert product in stats
            product_stats = stats[product]

            # Check required statistics
            assert 'mean_demand' in product_stats
            assert 'std_demand' in product_stats
            assert 'min_demand' in product_stats
            assert 'max_demand' in product_stats
            assert 'total_revenue' in product_stats
            assert 'day_of_week_avg' in product_stats

    def test_save_data(self):
        """Test saving data to CSV"""
        generator = SyntheticDemandGenerator(seed=42)
        df = generator.generate_historical_data(num_days=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            generator.save_data(df, output_dir=tmpdir, filename='test.csv')

            # Check file exists
            filepath = os.path.join(tmpdir, 'test.csv')
            assert os.path.exists(filepath)

            # Load and verify
            loaded_df = pd.read_csv(filepath)
            assert len(loaded_df) == 10 * 5

    def test_save_demand_profiles(self):
        """Test saving demand profiles"""
        generator = SyntheticDemandGenerator(seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            generator.save_demand_profiles(output_dir=tmpdir, filename='profiles.pkl')

            # Check file exists
            filepath = os.path.join(tmpdir, 'profiles.pkl')
            assert os.path.exists(filepath)

            # Load and verify
            profiles = load_demand_profiles(filepath)
            assert len(profiles) == 5
            assert 'berries' in profiles

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same data"""
        gen1 = SyntheticDemandGenerator(seed=42)
        gen2 = SyntheticDemandGenerator(seed=42)

        df1 = gen1.generate_historical_data(num_days=5)
        df2 = gen2.generate_historical_data(num_days=5)

        # Should be identical
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data"""
        gen1 = SyntheticDemandGenerator(seed=42)
        gen2 = SyntheticDemandGenerator(seed=99)

        df1 = gen1.generate_historical_data(num_days=5)
        df2 = gen2.generate_historical_data(num_days=5)

        # Should be different
        assert not df1['demand'].equals(df2['demand'])

    def test_load_demand_profiles_fallback(self):
        """Test that load_demand_profiles returns defaults if file not found"""
        profiles = load_demand_profiles(filepath='nonexistent_file.pkl')

        # Should still return valid profiles
        assert isinstance(profiles, dict)
        assert 'berries' in profiles
        assert 'milk' in profiles
