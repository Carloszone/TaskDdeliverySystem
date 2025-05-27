import unittest
import numpy as np
import sys
import os
import gymnasium as gym # Import gymnasium
from gymnasium import spaces


# Adjust the path to import from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from industrial_env import IndustrialEnv
# We might not need to import RobotArm and RewardCalculator explicitly if
# we are only testing the environment's interaction with them at a high level.

class TestIndustrialEnv(unittest.TestCase):

    def setUp(self):
        """Common setup for tests."""
        self.env_config_example = {
            "num_robot_arms": 2,
            "robot_arm_configs": [
                {"id": "R1", "task_list": ["task_1", "task_2"], "target_types": {}, "target_list": [], "location": 0, 
                 "task_info_mapping": {"task_1": {"time":3, "switch_time":1, "output_type":"A"}, "task_2": {"time":2, "switch_time":1, "output_type":"B"}}},
                {"id": "R2", "task_list": ["task_1", "task_3"], "target_types": {}, "target_list": [], "location": 5, 
                 "task_info_mapping": {"task_1": {"time":3, "switch_time":1, "output_type":"A"}, "task_3": {"time":4, "switch_time":1, "output_type":"C"}}},
            ],
            "reward_calculator_config": {}, # Use default reward calculator settings
            "max_steps_per_episode": 50 
        }
        self.env = IndustrialEnv(env_config=self.env_config_example)

    def test_initialization(self):
        self.assertEqual(len(self.env.robot_arm_list), self.env_config_example["num_robot_arms"])
        self.assertIsNotNone(self.env.reward_calculator)
        self.assertEqual(self.env.max_steps_per_episode, self.env_config_example["max_steps_per_episode"])
        self.assertIsInstance(self.env.action_space, spaces.Space)
        self.assertIsInstance(self.env.observation_space, spaces.Space)
        self.assertTrue(hasattr(self.env, 'metadata')) # From gym.Env

    def test_define_action_space(self):
        action_space = self.env.define_action_space()
        self.assertIsInstance(action_space, spaces.MultiDiscrete)
        # Expected shape based on num_arms + R0 and num_tasks + wait_task
        num_arms = self.env_config_example["num_robot_arms"]
        num_tasks_total = 10 + 1 # 10 tasks + 1 wait task
        expected_dims = [num_arms + 1, num_arms + 1, num_tasks_total]
        self.assertListEqual(list(action_space.nvec), expected_dims)

    def test_define_observation_space(self):
        observation_space = self.env.define_observation_space()
        self.assertIsInstance(observation_space, spaces.Box)
        # Expected shape based on num_robot_arms * features_per_arm
        num_features_per_arm = 5 # state, current_task_id, task_time_remaining, is_occupied, work_output
        max_arms_for_space = self.env_config_example.get("num_robot_arms", 2)
        expected_shape = (max_arms_for_space * num_features_per_arm,)
        self.assertEqual(observation_space.shape, expected_shape)


    def test_reset(self):
        obs, info = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertTrue(self.env.observation_space.contains(obs), "Observation from reset not in observation space.")
        self.assertIsInstance(info, dict)
        self.assertEqual(self.env.step_count, 0)
        self.assertEqual(self.env.total_rewards, 0.0)
        # Check if robot arms were reset (e.g., state is idle)
        for arm in self.env.robot_arm_list:
            self.assertEqual(arm.state, 0) # Assuming 0 is idle state

    def test_step(self):
        self.env.reset()
        # Sample a random valid action from the action space
        action = self.env.action_space.sample() 
        
        initial_step_count = self.env.step_count
        
        obs, reward, done, truncated, info = self.env.step(action)
        
        self.assertIsInstance(obs, np.ndarray)
        self.assertTrue(self.env.observation_space.contains(obs), "Observation from step not in observation space.")
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        
        self.assertEqual(self.env.step_count, initial_step_count + 1)
        # Further checks could involve conceptual state changes of robot arms,
        # but this depends on the detailed (placeholder) logic within RobotArm and the action parsing.
        # For now, ensuring the step runs and returns correct types is key.

    def test_step_truncation(self):
        self.env.reset()
        for _ in range(self.env.max_steps_per_episode -1):
            action = self.env.action_space.sample()
            _, _, _, truncated, _ = self.env.step(action)
            self.assertFalse(truncated)

        action = self.env.action_space.sample()
        _, _, _, truncated, _ = self.env.step(action) # This should be the max_steps_per_episode step
        self.assertTrue(truncated)


    def test_get_action_mask(self):
        self.env.reset()
        action_mask = self.env.get_action_mask() # Added in a previous subtask
        
        self.assertIsInstance(action_mask, dict)
        self.assertIn("start_arm_mask", action_mask)
        self.assertIn("end_arm_mask", action_mask)
        self.assertIn("task_id_mask", action_mask)
        
        self.assertIsInstance(action_mask["start_arm_mask"], np.ndarray)
        self.assertIsInstance(action_mask["end_arm_mask"], np.ndarray)
        self.assertIsInstance(action_mask["task_id_mask"], np.ndarray)
        
        # Check if mask dimensions match action space dimensions
        action_space_dims = self.env.action_space.nvec
        self.assertEqual(len(action_mask["start_arm_mask"]), action_space_dims[0])
        self.assertEqual(len(action_mask["end_arm_mask"]), action_space_dims[1])
        self.assertEqual(len(action_mask["task_id_mask"]), action_space_dims[2])

    def test_add_remove_robot_arm(self):
        # This test is more conceptual as action/observation spaces are fixed at init in current setup
        initial_num_arms = len(self.env.robot_arm_list)
        new_arm_config = {"id": "R_Test", "task_list": ["task1"], "target_types": {}, "target_list": [], 
                          "location": 10, "task_info_mapping": {"task1": {"time":5}}}
        self.env.add_robot_arm(new_arm_config)
        self.assertEqual(len(self.env.robot_arm_list), initial_num_arms + 1)
        
        self.env.remove_robot_arm("R_Test")
        self.assertEqual(len(self.env.robot_arm_list), initial_num_arms)
        
        # Try removing a non-existent arm
        self.env.remove_robot_arm("R_NonExistent")
        self.assertEqual(len(self.env.robot_arm_list), initial_num_arms)


if __name__ == '__main__':
    unittest.main()
