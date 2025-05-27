import unittest
import numpy as np
import sys
import os

# Adjust the path to import from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):

    def test_initialization(self):
        processor = DataProcessor(num_categorical_features=2, num_numerical_features=3)
        self.assertEqual(len(processor.coder), 2) # Check number of LabelEncoders
        self.assertIsNotNone(processor.scaler) # StandardScaler should be initialized
        self.assertEqual(processor.num_categorical_features, 2)
        self.assertEqual(processor.num_numerical_features, 3)
        self.assertFalse(processor.fitted)

        processor_no_scale = DataProcessor(num_numerical_features=0)
        self.assertIsNone(processor_no_scale.scaler)

    def test_observation_processor_basic(self):
        processor = DataProcessor(num_numerical_features=3) # Only numerical for simplicity here
        
        # Test with list
        obs_list = [1.0, 2.0, 3.0]
        processed_obs_list = processor.observation_processor(obs_list)
        self.assertIsInstance(processed_obs_list, np.ndarray)
        self.assertEqual(processed_obs_list.dtype, np.float32)
        np.testing.assert_array_equal(processed_obs_list, np.array(obs_list, dtype=np.float32))

        # Test with NumPy array
        obs_np = np.array([4.0, 5.0, 6.0])
        processed_obs_np = processor.observation_processor(obs_np)
        self.assertIsInstance(processed_obs_np, np.ndarray)
        self.assertEqual(processed_obs_np.dtype, np.float32)
        np.testing.assert_array_equal(processed_obs_np, obs_np.astype(np.float32))
        
        # Test with no scaler (should just convert to float32 numpy array)
        processor_no_scale = DataProcessor(num_numerical_features=0)
        processed_obs_no_scale = processor_no_scale.observation_processor(obs_list)
        self.assertIsInstance(processed_obs_no_scale, np.ndarray)
        np.testing.assert_array_equal(processed_obs_no_scale, np.array(obs_list, dtype=np.float32))


    def test_fit_scaler_and_process(self):
        num_feats = 2
        processor = DataProcessor(num_numerical_features=num_feats)
        
        # Sample data for fitting
        sample_data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        processor.fit_scaler(sample_data)
        self.assertTrue(processor.fitted)

        # Test processing a single observation
        obs_single = np.array([1.5, 2.5])
        processed_single = processor.observation_processor(obs_single)
        # Expected: (obs_single - sample_data.mean(axis=0)) / sample_data.std(axis=0)
        expected_mean = np.mean(sample_data, axis=0)
        expected_std = np.std(sample_data, axis=0)
        expected_processed_single = (obs_single - expected_mean) / expected_std
        np.testing.assert_array_almost_equal(processed_single, expected_processed_single, decimal=6)

        # Test processing a batch
        obs_batch = np.array([[0.5, 1.5], [3.5, 4.5]])
        processed_batch = processor.observation_processor(obs_batch)
        expected_processed_batch = (obs_batch - expected_mean) / expected_std
        np.testing.assert_array_almost_equal(processed_batch, expected_processed_batch, decimal=6)

        # Test processing with wrong feature number (should skip scaling)
        obs_wrong_feats = np.array([1.0, 2.0, 3.0])
        processed_wrong_feats = processor.observation_processor(obs_wrong_feats)
        np.testing.assert_array_equal(processed_wrong_feats, obs_wrong_feats.astype(np.float32)) # Should be unchanged other than type

    def test_gnn_generator(self):
        processor = DataProcessor()
        processed_obs = np.array([[1.0, 0.5], [0.3, 0.8]])
        
        # Placeholder GNN generator just returns the input
        gnn_data = processor.GNN_generator(processed_obs)
        np.testing.assert_array_equal(gnn_data, processed_obs)

    def test_scaler_not_fitted_warning(self):
        # This test would ideally capture a log warning or specific behavior.
        # For now, just ensure it runs and doesn't scale.
        processor = DataProcessor(num_numerical_features=2)
        obs = np.array([1.0, 2.0])
        processed_obs = processor.observation_processor(obs)
        np.testing.assert_array_equal(processed_obs, obs.astype(np.float32)) # Not scaled

    def test_fit_scaler_with_no_scaler(self):
        processor = DataProcessor(num_numerical_features=0)
        sample_data = np.array([[1.0, 2.0]])
        processor.fit_scaler(sample_data) # Should do nothing and not error
        self.assertFalse(processor.fitted)


if __name__ == '__main__':
    unittest.main()
