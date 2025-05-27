import unittest
import numpy as np
import sys
import os

# Adjust the path to import from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import ActorCriticModel, PlaceholderTransformerEncoder, PolicyNetworkHead

class TestModelComponents(unittest.TestCase):

    def setUp(self):
        self.num_c_feats = 5
        self.num_m_feats = 10
        self.num_w_feats = 3
        self.internal_dim = 32 # Smaller for tests
        self.transformer_heads = 1
        self.transformer_ffn_dim = 64
        self.transformer_layers = 1
        self.action_dims_example = [4, 4, 6] # start, end, task

        self.model_params_actor = {
            "num_categorical_features": self.num_c_feats,
            "num_numerical_features": self.num_m_feats,
            "num_work_process_features": self.num_w_feats,
            "internal_embedding_dim": self.internal_dim,
            "transformer_heads": self.transformer_heads,
            "transformer_ffn_dim": self.transformer_ffn_dim,
            "transformer_layers": self.transformer_layers,
            "action_dims": self.action_dims_example,
            "is_critic": False
        }
        self.model_params_critic = {
            "num_categorical_features": self.num_c_feats,
            "num_numerical_features": self.num_m_feats,
            "num_work_process_features": self.num_w_feats,
            "internal_embedding_dim": self.internal_dim,
            "transformer_heads": self.transformer_heads,
            "transformer_ffn_dim": self.transformer_ffn_dim,
            "transformer_layers": self.transformer_layers,
            "is_critic": True
        }

        # Dummy input data (batch size 1 for simplicity in some checks)
        self.batch_size = 1 # Can also test with > 1
        self.seq_len_c, self.seq_len_m, self.seq_len_w = 2, 2, 3 

        self.dummy_obs_C = np.random.rand(self.batch_size, self.seq_len_c, self.num_c_feats).astype(np.float32)
        self.dummy_obs_M = np.random.rand(self.batch_size, self.seq_len_m, self.num_m_feats).astype(np.float32)
        self.dummy_obs_W = np.random.rand(self.batch_size, self.seq_len_w, self.num_w_feats).astype(np.float32)
        self.dummy_direct_state_critic = np.random.rand(self.batch_size, self.internal_dim).astype(np.float32)


    def test_placeholder_transformer_encoder_init_forward(self):
        encoder = PlaceholderTransformerEncoder(input_dim=self.internal_dim, num_heads=2, feedforward_dim=64, num_layers=1)
        self.assertIn("layer_0_attention_w", encoder.state_dict())
        test_input = np.random.rand(self.batch_size, 5, self.internal_dim).astype(np.float32) # (batch, seq_len, input_dim)
        output = encoder.forward(test_input)
        self.assertEqual(output.shape, test_input.shape)

    def test_policy_network_head_init_forward(self):
        head = PolicyNetworkHead(input_dim=self.internal_dim, output_dim=self.action_dims_example[0])
        self.assertIn("linear_w", head.state_dict())
        test_input = np.random.rand(self.batch_size, self.internal_dim).astype(np.float32) # (batch, input_dim)
        output = head.forward(test_input)
        self.assertEqual(output.shape, (self.batch_size, self.action_dims_example[0]))


    def test_actor_model_initialization(self):
        actor_model = ActorCriticModel(**self.model_params_actor)
        self.assertIsNotNone(actor_model.transformer_C)
        self.assertIsNotNone(actor_model.projection_M)
        self.assertIsNotNone(actor_model.transformer_CM)
        self.assertIsNotNone(actor_model.transformer_W)
        self.assertIsNotNone(actor_model.transformer_final)
        self.assertTrue(hasattr(actor_model, 'head1_start_id'))
        self.assertTrue(hasattr(actor_model, 'head2_end_id'))
        self.assertTrue(hasattr(actor_model, 'head3_task_id'))
        self.assertFalse(hasattr(actor_model, 'critic_value_head') or actor_model.is_critic) # Should not have critic value head

    def test_critic_model_initialization(self):
        critic_model = ActorCriticModel(**self.model_params_critic)
        self.assertIsNotNone(critic_model.transformer_C) # Shared body
        self.assertTrue(hasattr(critic_model, 'critic_value_head'))
        self.assertFalse(hasattr(critic_model, 'head1_start_id')) # Should not have actor heads

    def test_actor_model_forward_pass(self):
        actor_model = ActorCriticModel(**self.model_params_actor)
        # ActorCriticModel.forward returns tuple: (head1_output, head2_output, head3_output)
        outputs = actor_model.forward(self.dummy_obs_C, self.dummy_obs_M, self.dummy_obs_W)
        self.assertEqual(len(outputs), 3)
        # Check output shapes based on action_dims and batch_size
        # Placeholder PolicyNetworkHead returns random data of shape (batch_size, output_dim)
        self.assertEqual(outputs[0].shape, (self.batch_size, self.action_dims_example[0]))
        self.assertEqual(outputs[1].shape, (self.batch_size, self.action_dims_example[1]))
        self.assertEqual(outputs[2].shape, (self.batch_size, self.action_dims_example[2]))

    def test_critic_model_forward_pass(self):
        critic_model = ActorCriticModel(**self.model_params_critic)
        # Critic model's forward returns a tuple: (value_output,)
        # Test with direct state input for critic (conceptual V_prime or similar)
        value_output_tuple = critic_model.forward(self.dummy_obs_C, self.dummy_obs_M, self.dummy_obs_W, obs_state_for_critic=self.dummy_direct_state_critic)
        self.assertEqual(len(value_output_tuple), 1)
        self.assertEqual(value_output_tuple[0].shape, (self.batch_size, 1)) # Critic head outputs a single value per batch item

        # Test critic forward using the C,M,W pipeline to generate V_prime internally
        value_output_tuple_pipeline = critic_model.forward(self.dummy_obs_C, self.dummy_obs_M, self.dummy_obs_W)
        self.assertEqual(len(value_output_tuple_pipeline), 1)
        self.assertEqual(value_output_tuple_pipeline[0].shape, (self.batch_size, 1))


    def test_model_state_dict_and_load(self):
        actor_model = ActorCriticModel(**self.model_params_actor)
        state_dict_original = actor_model.state_dict()
        self.assertIsInstance(state_dict_original, dict)
        self.assertTrue(len(state_dict_original) > 0) # Check it's not empty
        
        # Check a known key from a sub-module (PlaceholderWeightContainer adds 'initial_param')
        # ActorCriticModel._collect_submodule_weights prefixes them
        expected_key_example = "transformer_C_initial_param" 
        self.assertIn(expected_key_example, state_dict_original)

        new_actor_model = ActorCriticModel(**self.model_params_actor)
        # Ensure weights are different before loading (random init)
        # This is tricky with np.random.rand, but we can check one specific weight if it's predictable or store/compare all
        # For simplicity, we trust random init differs enough or focus on successful load call.
        
        new_actor_model.load_state_dict(state_dict_original)
        
        state_dict_loaded = new_actor_model.state_dict()

        # Check if all keys are present and values are equal
        self.assertEqual(state_dict_original.keys(), state_dict_loaded.keys())
        for key in state_dict_original:
            np.testing.assert_array_equal(state_dict_original[key], state_dict_loaded[key],
                                          err_msg=f"Weight mismatch for key: {key}")

    def test_feature_extractor_model_forward(self):
        # Test model when action_dims is None and is_critic is False (feature extractor mode)
        model_params_fe = self.model_params_actor.copy()
        del model_params_fe["action_dims"] # Remove action_dims
        model_params_fe["is_critic"] = False

        feature_extractor = ActorCriticModel(**model_params_fe)
        outputs = feature_extractor.forward(self.dummy_obs_C, self.dummy_obs_M, self.dummy_obs_W)
        self.assertEqual(len(outputs), 1) # Should be (V_prime,)
        # V_prime shape after mean pooling over sequence length: (batch_size, internal_embedding_dim)
        self.assertEqual(outputs[0].shape, (self.batch_size, self.internal_dim))


if __name__ == '__main__':
    unittest.main()
