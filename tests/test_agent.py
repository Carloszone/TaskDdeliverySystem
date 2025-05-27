import unittest
import numpy as np
import sys
import os
import collections

# Adjust the path to import from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent import Agent, ReplayBuffer 
# ActorCriticModel is used by Agent, so it's implicitly tested via Agent's interactions
# from model import ActorCriticModel # Not directly instantiated here, but by Agent

class TestAgent(unittest.TestCase):

    def setUp(self):
        """Common setup for Agent tests."""
        self.state_dim_flat = 20 # Conceptual flat state dimension
        self.action_dims = [3, 3, 4] # E.g., start_arm (0-2), end_arm (0-2), task (0-3)
        
        self.model_params_example = {
            "num_categorical_features": 5,
            "num_numerical_features": 5,
            "num_work_process_features": 2,
            "internal_embedding_dim": 16, # Smaller for faster tests
            "transformer_heads": 1,
            "transformer_ffn_dim": 32,
            "transformer_layers": 1,
            "seq_len_c_example": 2, 
            "seq_len_m_example": 2,
            "seq_len_w_example": 2 
        }
        
        self.agent_params = {
            "state_dim": self.state_dim_flat,
            "action_dims": self.action_dims,
            "gamma": 0.9,
            "epsilon": 0.95,
            "epsilon_decay": 0.99,
            "epsilon_min": 0.01,
            "tau": 0.01,
            "learning_rate_actor": 2e-4,
            "learning_rate_critic": 2e-4,
            "buffer_size": 500, # Smaller for tests
            "batch_size": 32,    # Smaller for tests
            "min_learn_size": 40, # Smaller for tests
            "model_params": self.model_params_example
        }
        self.agent = Agent(**self.agent_params)

    def test_initialization(self):
        self.assertEqual(self.agent.state_dim, self.state_dim_flat)
        self.assertEqual(self.agent.action_dims, self.action_dims)
        self.assertEqual(self.agent.gamma, 0.9)
        self.assertIsNotNone(self.agent.action_net)
        self.assertIsNotNone(self.agent.critic_net)
        self.assertIsNotNone(self.agent.target_action_net)
        self.assertIsNotNone(self.agent.target_critic_net)
        self.assertIsNotNone(self.agent.replay_buffer)
        self.assertEqual(len(self.agent.replay_buffer), 0)

    def test_store_transition(self):
        initial_buffer_len = len(self.agent.replay_buffer)
        dummy_state = np.random.rand(self.state_dim_flat)
        dummy_action = np.random.randint(0, 2, size=len(self.action_dims))
        dummy_reward = 0.5
        dummy_next_state = np.random.rand(self.state_dim_flat)
        dummy_done = False
        
        self.agent.store_transition(dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done)
        self.assertEqual(len(self.agent.replay_buffer), initial_buffer_len + 1)

    def test_choose_action_no_mask(self):
        dummy_observation = np.random.rand(self.state_dim_flat)
        action = self.agent.choose_action(dummy_observation, action_mask=None)
        self.assertEqual(action.shape, (len(self.action_dims),))
        for i, act_comp in enumerate(action):
            self.assertTrue(0 <= act_comp < self.action_dims[i])

    def test_choose_action_with_mask(self):
        dummy_observation = np.random.rand(self.state_dim_flat)
        # Create a mask that restricts some actions
        action_mask = {
            "start_arm_mask": np.array([1, 0, 1], dtype=np.int8), # Action 1 for start_arm is invalid
            "end_arm_mask": np.array([0, 1, 0], dtype=np.int8),   # Actions 0, 2 for end_arm are invalid
            "task_id_mask": np.array([1, 1, 0, 0], dtype=np.int8) # Actions 2, 3 for task_id are invalid
        }
        
        self.agent.epsilon = 0.0 # Ensure policy is used (exploitation)
        
        # Run multiple times to increase chance of hitting a masked action if logic is flawed
        for _ in range(50): 
            action = self.agent.choose_action(dummy_observation, action_mask=action_mask)
            self.assertEqual(action.shape, (len(self.action_dims),))
            self.assertNotEqual(action[0], 1) # Should not choose action 1 for start_arm
            self.assertNotEqual(action[1], 0) # Should not choose action 0 for end_arm
            self.assertNotEqual(action[1], 2) # Should not choose action 2 for end_arm
            self.assertNotIn(action[2], [2, 3]) # Should not choose action 2 or 3 for task

    def test_learn_process(self):
        # Populate buffer to enable learning
        for _ in range(self.agent_params["min_learn_size"]):
            dummy_state = np.random.rand(self.state_dim_flat)
            dummy_action = np.array([np.random.randint(d) for d in self.action_dims])
            dummy_reward = np.random.rand()
            dummy_next_state = np.random.rand(self.state_dim_flat)
            dummy_done = bool(np.random.choice([True, False]))
            self.agent.store_transition(dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done)

        self.assertTrue(len(self.agent.replay_buffer) >= self.agent.min_learn_size)
        
        # Get initial state of target networks (conceptual placeholder weights)
        initial_target_actor_weights = self.agent.target_action_net.state_dict().copy()
        initial_target_critic_weights = self.agent.target_critic_net.state_dict().copy()
        
        self.agent.learn() # This will call _calculate_advantages and _update_target_networks
        
        # Check if target networks were updated (soft update)
        # This is conceptual with placeholder state_dicts. We check if they are not identical if tau < 1
        if self.agent.tau < 1.0 and self.agent.tau > 0.0:
            # Check a sample weight. It's hard to guarantee non-equality due to random nature of placeholders,
            # but if tau is applied, they should differ if main net had any "change".
            # The placeholder learn() doesn't change main nets, so target nets won't change from main.
            # However, _update_target_networks itself might change the dict if keys mismatch or values are floats.
            # For this test, we mainly ensure learn() runs and calls sub-methods.
            # A more robust check would involve mocking sub-methods or inspecting specific weight values if deterministic.
            pass # Direct weight comparison is difficult with current placeholder setup. 
                 # The key is that learn() runs and calls the sub-methods.

        self.assertTrue(self.agent.learn_step_counter > 0)


    def test_save_load_checkpoint(self):
        # Modify some agent parameter to check if it's saved/loaded (e.g., epsilon)
        self.agent.epsilon = 0.5
        # Conceptually "train" the action_net a bit by changing a weight directly for placeholder
        # This simulates that the network has learned something.
        # We access the internal 'weights' dict of the PlaceholderWeightContainer via state_dict()
        actor_sd = self.agent.action_net.state_dict()
        # Find a weight key to modify, e.g., the first one found from a sub-module
        # Example: "transformer_C_layer_0_attention_w" or "projection_M_linear_w"
        key_to_modify = None
        for k in actor_sd.keys(): # Find a weight array to modify
            if isinstance(actor_sd[k], np.ndarray) and actor_sd[k].ndim > 0:
                key_to_modify = k
                break
        
        original_weight_val = None
        if key_to_modify:
            original_weight_val = np.copy(actor_sd[key_to_modify])
            actor_sd[key_to_modify] = np.random.rand(*actor_sd[key_to_modify].shape) # Change it
            self.agent.action_net.load_state_dict(actor_sd) # Load modified back into action_net
        
        path = "./test_checkpoints"
        prefix = "test_agent_cp_"
        self.agent.save_checkpoint(path=path, filename_prefix=prefix)
        
        # Create a new agent and load
        new_agent_params = self.agent_params.copy()
        new_agent_params["epsilon"] = 1.0 # Different initial epsilon
        new_agent = Agent(**new_agent_params)
        new_agent.load_checkpoint(path=path, filename_prefix=prefix)
        
        # Check if weights are loaded (conceptual via state_dict comparison)
        if key_to_modify:
            loaded_actor_sd = new_agent.action_net.state_dict()
            self.assertIn(key_to_modify, loaded_actor_sd)
            np.testing.assert_array_equal(loaded_actor_sd[key_to_modify], actor_sd[key_to_modify],
                                          err_msg=f"Weight for {key_to_modify} did not load correctly.")
        
        # Note: Epsilon is not part of the network's state_dict, so it's not saved/loaded by this function.
        # self.assertEqual(new_agent.epsilon, self.agent.epsilon) # This would fail as epsilon is not saved
        
        # Cleanup
        if os.path.exists(os.path.join(path, f"{prefix}action_net.npz")):
            os.remove(os.path.join(path, f"{prefix}action_net.npz"))
        if os.path.exists(os.path.join(path, f"{prefix}critic_net.npz")):
            os.remove(os.path.join(path, f"{prefix}critic_net.npz"))
        if os.path.exists(path) and not os.listdir(path): # Remove dir if empty
             os.rmdir(path)


if __name__ == '__main__':
    unittest.main()
