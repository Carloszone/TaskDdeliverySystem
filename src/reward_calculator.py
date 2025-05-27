class RewardCalculator:
    """
    Calculates rewards based on task execution logs and defined reward rules.
    """
    def __init__(self,
                 product_completion_reward: float = 100.0,
                 step_correct_reward: float = 20.0,
                 invalid_robot_arm_id_penalty: float = -20.0,
                 invalid_task_id_penalty: float = -10.0,
                 move_cost_per_unit_distance: float = -1.0,
                 task_switch_cost: float = -5.0,
                 task_execution_cost_per_time_unit: float = -0.1):
        """
        Initializes the RewardCalculator.

        Args:
            product_completion_reward: Reward for completing the final step of a product.
            step_correct_reward: Reward for successfully completing any single process step.
            invalid_robot_arm_id_penalty: Penalty for assigning a task to an invalid/wrong robot arm.
            invalid_task_id_penalty: Penalty for assigning an invalid/wrong task.
            move_cost_per_unit_distance: Cost for moving a mover per unit of distance.
            task_switch_cost: Cost associated with a robot arm switching its current task.
            task_execution_cost_per_time_unit: Cost per unit of time for task execution.
        """
        self.product_completion_reward: float = product_completion_reward
        self.step_correct_reward: float = step_correct_reward
        self.invalid_robot_arm_id_penalty: float = invalid_robot_arm_id_penalty
        self.invalid_task_id_penalty: float = invalid_task_id_penalty
        # Costs related to "常规任务成本"
        self.move_cost_per_unit_distance: float = move_cost_per_unit_distance
        self.task_switch_cost: float = task_switch_cost
        self.task_execution_cost_per_time_unit: float = task_execution_cost_per_time_unit

    def calculate_reward(self,
                         is_final_step: bool,
                         step_successful: bool,
                         invalid_arm_assignment: bool,
                         invalid_task_assignment: bool,
                         distance_moved: int = 0,
                         task_switched: bool = False,
                         task_execution_time: int = 0) -> float:
        """
        Calculates the total reward for a given step/action.

        Args:
            is_final_step: True if the completed step was the final one for a product.
            step_successful: True if the current process step was completed successfully.
            invalid_arm_assignment: True if the task was assigned to a wrong robot arm.
            invalid_task_assignment: True if a wrong task was assigned.
            distance_moved: The distance the mover traveled for this task.
            task_switched: True if the robot arm had to switch tasks.
            task_execution_time: The time taken for the robot arm to execute the task.

        Returns:
            The calculated reward value.

        Reward Rules (from README.md to be implemented in detail):
        1. 产品完成工序最后一步：+100 (self.product_completion_reward)
        2. 成功完成任一工序步骤：+20 (self.step_correct_reward)
        3. 任务分配失败（任务错误）：-10 (self.invalid_task_id_penalty)
        4. 任务分配失败（机械臂错误）：-20 (self.invalid_robot_arm_id_penalty)
        5. 常规任务成本 =
           - 移动成本：|location_end - location_start| (distance_moved * self.move_cost_per_unit_distance)
           - 任务切换成本（若发生） (task_switched * self.task_switch_cost)
           - 任务执行耗时（根据类型） (task_execution_time * self.task_execution_cost_per_time_unit)
        """
        reward = 0.0

        # Penalties for errors
        if invalid_arm_assignment:
            reward += self.calculate_error_penalty(error_type="invalid_robot_arm")
        if invalid_task_assignment:
            reward += self.calculate_error_penalty(error_type="invalid_task")

        # Rewards for successful operations
        if step_successful:
            reward += self.calculate_product_completion_reward(is_final_step)
            if not is_final_step : # Add step_correct_reward if it's not the final step (final step already includes a larger reward)
                 reward += self.step_correct_reward
        
        # Costs
        reward += distance_moved * self.move_cost_per_unit_distance
        if task_switched:
            reward += self.task_switch_cost
        reward += task_execution_time * self.task_execution_cost_per_time_unit
        
        print(f"Calculated reward: {reward} (Final: {is_final_step}, Success: {step_successful}, InvalidArm: {invalid_arm_assignment}, InvalidTask: {invalid_task_assignment}, Dist: {distance_moved}, Switch: {task_switched}, Time: {task_execution_time})")
        return reward

    def calculate_product_completion_reward(self, is_final_step: bool) -> float:
        """
        Calculates reward related to product completion.
        - Product complete (final step): +product_completion_reward
        """
        if is_final_step:
            print(f"Reward: Product final step completed (+{self.product_completion_reward})")
            return self.product_completion_reward
        return 0.0

    def calculate_error_penalty(self, error_type: str) -> float:
        """
        Calculates penalties for errors.
        - Invalid task assignment: invalid_task_id_penalty
        - Invalid robot arm assignment: invalid_robot_arm_id_penalty
        """
        if error_type == "invalid_task":
            print(f"Penalty: Invalid task assignment ({self.invalid_task_id_penalty})")
            return self.invalid_task_id_penalty
        elif error_type == "invalid_robot_arm":
            print(f"Penalty: Invalid robot arm assignment ({self.invalid_robot_arm_id_penalty})")
            return self.invalid_robot_arm_id_penalty
        return 0.0

if __name__ == '__main__':
    calculator = RewardCalculator()

    print("\n--- Example Calculations ---")

    # Scenario 1: Successful final step, no errors, some costs
    reward1 = calculator.calculate_reward(
        is_final_step=True,
        step_successful=True,
        invalid_arm_assignment=False,
        invalid_task_assignment=False,
        distance_moved=5,
        task_switched=True,
        task_execution_time=10
    )
    # Expected: 100 (product_completion) - 5*1 (move_cost) - 5 (switch_cost) - 10*0.1 (exec_cost) = 100 - 5 - 5 - 1 = 89
    print(f"Scenario 1 Reward: {reward1} (Expected: 89)")

    # Scenario 2: Successful intermediate step, no errors, some costs
    reward2 = calculator.calculate_reward(
        is_final_step=False,
        step_successful=True,
        invalid_arm_assignment=False,
        invalid_task_assignment=False,
        distance_moved=2,
        task_switched=False,
        task_execution_time=5
    )
    # Expected: 20 (step_correct) - 2*1 (move_cost) - 0 (switch_cost) - 5*0.1 (exec_cost) = 20 - 2 - 0 - 0.5 = 17.5
    print(f"Scenario 2 Reward: {reward2} (Expected: 17.5)")

    # Scenario 3: Failed step due to invalid arm assignment
    reward3 = calculator.calculate_reward(
        is_final_step=False,
        step_successful=False,
        invalid_arm_assignment=True,
        invalid_task_assignment=False,
        distance_moved=0,
        task_switched=False,
        task_execution_time=0
    )
    # Expected: -20 (invalid_arm_penalty)
    print(f"Scenario 3 Reward: {reward3} (Expected: -20)")

    # Scenario 4: Failed step due to invalid task assignment, plus other costs that might still apply if action was attempted
    reward4 = calculator.calculate_reward(
        is_final_step=False,
        step_successful=False, # Assuming step is not successful if there's an assignment error
        invalid_arm_assignment=False,
        invalid_task_assignment=True,
        distance_moved=3, # e.g. mover went to a location before error was identified
        task_switched=False,
        task_execution_time=0
    )
    # Expected: -10 (invalid_task_penalty) - 3*1 (move_cost) = -10 - 3 = -13
    print(f"Scenario 4 Reward: {reward4} (Expected: -13)")

    # Scenario 5: Successful step but with high costs
    reward5 = calculator.calculate_reward(
        is_final_step=False,
        step_successful=True,
        invalid_arm_assignment=False,
        invalid_task_assignment=False,
        distance_moved=10,
        task_switched=True,
        task_execution_time=20
    )
    # Expected: 20 (step_correct) - 10*1 (move_cost) - 5 (switch_cost) - 20*0.1 (exec_cost) = 20 - 10 - 5 - 2 = 3
    print(f"Scenario 5 Reward: {reward5} (Expected: 3)")

    # Scenario 6: All penalties and costs
    reward6 = calculator.calculate_reward(
        is_final_step=False, # Does not matter if step_successful is False
        step_successful=False,
        invalid_arm_assignment=True,
        invalid_task_assignment=True,
        distance_moved=5,
        task_switched=True, # Assume switch was attempted before error fully registered
        task_execution_time=2 # Assume some minimal time before error stopped process
    )
    # Expected: -20 (invalid_arm) -10 (invalid_task) -5*1 (move) -5 (switch) -2*0.1 (exec) = -20-10-5-5-0.2 = -40.2
    print(f"Scenario 6 Reward: {reward6} (Expected: -40.2)")

    print("\nRewardCalculator class implementation complete with example usage.")
