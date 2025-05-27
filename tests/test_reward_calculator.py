import unittest
import sys
import os

# Adjust the path to import from the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from reward_calculator import RewardCalculator

class TestRewardCalculator(unittest.TestCase):

    def setUp(self):
        """Common setup for tests."""
        self.reward_config = {
            "product_completion_reward": 100.0,
            "step_correct_reward": 20.0,
            "invalid_robot_arm_id_penalty": -20.0,
            "invalid_task_id_penalty": -10.0,
            "move_cost_per_unit_distance": -1.0,
            "task_switch_cost": -5.0,
            "task_execution_cost_per_time_unit": -0.1
        }
        self.calculator = RewardCalculator(**self.reward_config)

    def test_initialization(self):
        self.assertEqual(self.calculator.product_completion_reward, self.reward_config["product_completion_reward"])
        self.assertEqual(self.calculator.step_correct_reward, self.reward_config["step_correct_reward"])
        self.assertEqual(self.calculator.invalid_robot_arm_id_penalty, self.reward_config["invalid_robot_arm_id_penalty"])
        self.assertEqual(self.calculator.invalid_task_id_penalty, self.reward_config["invalid_task_id_penalty"])
        self.assertEqual(self.calculator.move_cost_per_unit_distance, self.reward_config["move_cost_per_unit_distance"])
        self.assertEqual(self.calculator.task_switch_cost, self.reward_config["task_switch_cost"])
        self.assertEqual(self.calculator.task_execution_cost_per_time_unit, self.reward_config["task_execution_cost_per_time_unit"])

    def test_calculate_product_completion_reward(self):
        # Test final step completion
        reward_final_step = self.calculator.calculate_product_completion_reward(is_final_step=True)
        self.assertEqual(reward_final_step, self.reward_config["product_completion_reward"])

        # Test non-final step
        reward_non_final_step = self.calculator.calculate_product_completion_reward(is_final_step=False)
        self.assertEqual(reward_non_final_step, 0.0)

    def test_calculate_error_penalty(self):
        # Test invalid task penalty
        penalty_invalid_task = self.calculator.calculate_error_penalty(error_type="invalid_task")
        self.assertEqual(penalty_invalid_task, self.reward_config["invalid_task_id_penalty"])

        # Test invalid robot arm penalty
        penalty_invalid_arm = self.calculator.calculate_error_penalty(error_type="invalid_robot_arm")
        self.assertEqual(penalty_invalid_arm, self.reward_config["invalid_robot_arm_id_penalty"])

        # Test unknown error type
        penalty_unknown_error = self.calculator.calculate_error_penalty(error_type="unknown_error")
        self.assertEqual(penalty_unknown_error, 0.0)

    def test_calculate_reward_scenarios(self):
        # Scenario 1: Successful final step, no errors, some costs
        reward1 = self.calculator.calculate_reward(
            is_final_step=True, step_successful=True,
            invalid_arm_assignment=False, invalid_task_assignment=False,
            distance_moved=5, task_switched=True, task_execution_time=10
        )
        expected_reward1 = (self.reward_config["product_completion_reward"] + # 100 (final step includes base step reward implicitly by design in example)
                            5 * self.reward_config["move_cost_per_unit_distance"] + # -5
                            self.reward_config["task_switch_cost"] + # -5
                            10 * self.reward_config["task_execution_cost_per_time_unit"]) # -1
                            # = 100 - 5 - 5 - 1 = 89
        self.assertEqual(reward1, expected_reward1)

        # Scenario 2: Successful intermediate step, no errors, some costs
        reward2 = self.calculator.calculate_reward(
            is_final_step=False, step_successful=True,
            invalid_arm_assignment=False, invalid_task_assignment=False,
            distance_moved=2, task_switched=False, task_execution_time=5
        )
        expected_reward2 = (self.reward_config["step_correct_reward"] + # 20
                             2 * self.reward_config["move_cost_per_unit_distance"] + # -2
                             # 0 for task_switched=False
                             5 * self.reward_config["task_execution_cost_per_time_unit"]) # -0.5
                             # = 20 - 2 - 0.5 = 17.5
        self.assertEqual(reward2, expected_reward2)


        # Scenario 3: Failed step due to invalid arm assignment
        reward3 = self.calculator.calculate_reward(
            is_final_step=False, step_successful=False, # Step not successful
            invalid_arm_assignment=True, invalid_task_assignment=False,
            distance_moved=0, task_switched=False, task_execution_time=0
        )
        expected_reward3 = self.reward_config["invalid_robot_arm_id_penalty"] # -20
        self.assertEqual(reward3, expected_reward3)


        # Scenario 4: Failed step due to invalid task assignment, plus other costs
        reward4 = self.calculator.calculate_reward(
            is_final_step=False, step_successful=False, # Step not successful
            invalid_arm_assignment=False, invalid_task_assignment=True,
            distance_moved=3, task_switched=False, task_execution_time=0
        )
        expected_reward4 = (self.reward_config["invalid_task_id_penalty"] + # -10
                             3 * self.reward_config["move_cost_per_unit_distance"]) # -3
                             # = -13
        self.assertEqual(reward4, expected_reward4)

        # Scenario 5: Successful step but high costs leading to small positive reward
        reward5 = self.calculator.calculate_reward(
            is_final_step=False, step_successful=True,
            invalid_arm_assignment=False, invalid_task_assignment=False,
            distance_moved=10, task_switched=True, task_execution_time=20
        )
        expected_reward5 = (self.reward_config["step_correct_reward"] + # 20
                             10 * self.reward_config["move_cost_per_unit_distance"] + # -10
                             self.reward_config["task_switch_cost"] + # -5
                             20 * self.reward_config["task_execution_cost_per_time_unit"]) # -2
                             # = 20 - 10 - 5 - 2 = 3
        self.assertEqual(reward5, expected_reward5)
        
        # Scenario 6: All penalties and costs applied
        # Step not successful, so no positive rewards for completion/step.
        reward6 = self.calculator.calculate_reward(
            is_final_step=False, step_successful=False, 
            invalid_arm_assignment=True, invalid_task_assignment=True,
            distance_moved=5, task_switched=True, task_execution_time=2
        )
        expected_reward6 = (self.reward_config["invalid_robot_arm_id_penalty"] + # -20
                             self.reward_config["invalid_task_id_penalty"] +    # -10
                             5 * self.reward_config["move_cost_per_unit_distance"] + # -5
                             self.reward_config["task_switch_cost"] + # -5
                             2 * self.reward_config["task_execution_cost_per_time_unit"]) # -0.2
                             # = -20 -10 -5 -5 -0.2 = -40.2
        self.assertAlmostEqual(reward6, expected_reward6) # Use assertAlmostEqual for float comparisons

        # Scenario 7: Step successful, final step, but also errors (should not happen logically, but test reward calc)
        # If step_successful is True, error penalties are applied, but also success rewards.
        reward7 = self.calculator.calculate_reward(
            is_final_step=True, step_successful=True,
            invalid_arm_assignment=True, invalid_task_assignment=True,
            distance_moved=1, task_switched=False, task_execution_time=1
        )
        expected_reward7 = (self.reward_config["product_completion_reward"] + # 100
                            self.reward_config["invalid_robot_arm_id_penalty"] + # -20
                            self.reward_config["invalid_task_id_penalty"] +    # -10
                            1 * self.reward_config["move_cost_per_unit_distance"] + # -1
                            1 * self.reward_config["task_execution_cost_per_time_unit"]) # -0.1
                            # = 100 - 20 - 10 - 1 - 0.1 = 68.9
        self.assertAlmostEqual(reward7, expected_reward7)


if __name__ == '__main__':
    unittest.main()
