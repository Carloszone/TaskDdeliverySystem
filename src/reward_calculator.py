from src.robot_arm import RobotArm
from config_manager import config
import numpy as np


class RewardCalculator:
    """
    Calculates rewards based on task execution logs and defined reward rules.
    """
    def __init__(self,
                 product_completion_reward: float = 200.0,
                 step_correct_reward: float = 30.0,
                 invalid_robot_arm_id_penalty: float = -30.0,
                 invalid_task_id_penalty: float = -20.0,
                 move_cost_per_unit_distance: float = -1.0):
        """
        初始化奖励计算器.

        Args:
            product_completion_reward: 完成全部工序的奖励（不考虑成品NG或OK情况）.
            step_correct_reward: 当前任务分配合理的奖励.
            invalid_robot_arm_id_penalty: 分配错误机械臂的惩罚.
            invalid_task_id_penalty: 分配错误任务ID的惩罚.
            move_cost_per_unit_distance: 每行动一个单位距离的惩罚系数.
        """
        self.product_completion_reward: float = product_completion_reward
        self.step_correct_reward: float = step_correct_reward
        self.invalid_robot_arm_id_penalty: float = invalid_robot_arm_id_penalty
        self.invalid_task_id_penalty: float = invalid_task_id_penalty
        self.move_cost_per_unit_distance: float = move_cost_per_unit_distance

    def calculate_reward(self, start_robot_arm: RobotArm, end_robot_arm: RobotArm, task_id: str, task_check_result: bool,
                         task_execute_cost: float) -> float:
        """
        计算当前任务分配行动的奖励值。
        步骤：
        1. 计算动子移动的奖励值
        2. 计算任务分配结果的奖励值
        3. 计算完成全部工序的奖励值
        4. 计算任务执行的成本消耗
        Args:
            start_robot_arm: 起点的机械臂
            end_robot_arm: 终点的机械臂
            task_id: 分配的任务id
            task_check_result：任务检测结果
            task_execute_cost：任务执行耗时
        Returns:
            本次行动的总奖励.
        """
        # 计算动子移动成本
        move_cost = self.calculate_move_cost(start_robot_arm, end_robot_arm)

        # 计算任务分配的奖励/惩罚
        delivery_reward = self.calculate_task_delivery_reward(start_robot_arm, end_robot_arm, task_check_result, task_id)

        # 计算完成全部工序的奖励
        all_completion_reward = self.calculate_completion_reward(task_check_result, task_id)

        # 计算总奖励
        total_reward = move_cost + delivery_reward + all_completion_reward + task_execute_cost
        return total_reward

    def calculate_move_cost(self, start_robot_arm: RobotArm, end_robot_arm: RobotArm):
        """
        计算动子在行动中的移动成本。计算公式为 移动成本 = 移动距离 * 每单位距离的惩罚系数
        """
        distance = np.abs(end_robot_arm.location - start_robot_arm.location)
        return distance * self.move_cost_per_unit_distance

    def calculate_task_delivery_reward(self, start_robot_arm: RobotArm, end_robot_arm: RobotArm, check_result: bool,
                                       task_id: str):
        """
        计算任务分配的奖励和惩罚。如果任务检查通过，则获得奖励；如果不通过，分类讨论惩罚情况
        """
        loading_action_list = config.get_setting("loading_action_list")  # 上料动作列表

        if check_result:
            return self.step_correct_reward
        else:
            if task_id not in end_robot_arm.task_list:
                return self.invalid_robot_arm_id_penalty
            if end_robot_arm.state != 0:
                return self.invalid_robot_arm_id_penalty
            if end_robot_arm.is_occupied is True:
                return self.invalid_robot_arm_id_penalty
            if task_id in loading_action_list:
                if end_robot_arm.have_output is False:
                    return self.invalid_task_id_penalty
            else:
                if (end_robot_arm.target_types is not None) and end_robot_arm.target_types[start_robot_arm.output_type] == 0:
                    return self.invalid_robot_arm_id_penalty
                if start_robot_arm.have_output is False:
                    return self.invalid_robot_arm_id_penalty
                if start_robot_arm.is_ng == 1:
                    return self.invalid_robot_arm_id_penalty
                return self.invalid_task_id_penalty

    def calculate_completion_reward(self, check_result, task_id):
        """
        计算是否完成全部生产工序的函数。如果任务检查通过，且任务类型为下料，则认为完成全部工序
        """
        unloading_task_list = config.get_setting("unloading_task_list")
        if check_result:
            if task_id in unloading_task_list:
                return self.product_completion_reward
        return 0