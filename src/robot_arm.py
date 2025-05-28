import random
import logging
from config_manager import config


class RobotArm:
    """
    模拟工业场景下机械臂工作和交互的类.
    """

    def __init__(self, robot_id: str, task_id: str, task_ids: list, task_list: list, target_types: list[bool],
                 target_list: list[bool], location: float, state: int = 0, is_occupied: bool = False,
                 have_output: bool = True, is_ng: int = -1, output_type: int = None, available_outputs: list = None,
                 task_time: float = 0.0, task_switch_time: float = 1.0, timer: int = 0, failure_rate: float = 0.0,
                 recovery_rate: float = 0.0, ng_rate: float = 0.01, task_info_mapping=None):
        """
        初始化一个机械臂实例.

        notes:
        1. 假设机械臂切换所有任务的耗时一致，均为1.0
        2. 工件在动子和机械臂之间的移动（上下料）耗时设置为0
        3. 不考虑动子的运动过程，只计算运动时间作为成本
        4. 对于非专门上下料的机械臂，其主要任务中包含下料操作，但是上料操作独立。以质检流程为例： 机械臂可以执行的两个任务为： （下料）质检， 上料
        5. 每次请求状态视为时间流逝1s，AKA 上位系统以每秒一次的频率请求状态信息并进行任务分配
        """
        if task_info_mapping is None:
            task_info_mapping = {}
        self.id: str = robot_id  # 机械臂的唯一标识
        self.task_id: str = task_id  # 当前执行的任务类型id
        self.init_task_id: str = task_id  # 初始的任务id，用于重置信息
        self.available_task_ids: list = task_ids  # 当前可以执行的任务类型（不需要切换机械臂功能）
        self.task_list: list = task_list  # 机械臂的所有可执行的任务类型id
        self.target_types: list = target_types  # 处理对象类别列表，基于target_list更新,长度等于所有需处理工件的总类型，如果不需要处理对象，则为None。
        self.target_list: list = target_list  # 可处理对象列表,长度等于所有需处理工件的总类型。1表示需要，0表示不需要。，如果不需要处理对象，则为None。
        self.location: float = location  # 机械臂所在的位置（离动子起点）的距离值，用于标识机械臂的空间次序
        self.state: int = state  # 0: 空闲, 1: 工作中, 2: 故障, 3: 离线
        self.is_occupied: bool = is_occupied  # 机械臂的动子位是否被占用。0：未被占用， 1：被占用
        self.have_output: bool = have_output  # 机械臂是否有输出
        self.output_type: int = output_type  # 输出的工件id，没有则为None。
        self.available_outputs: list = available_outputs  # 可以输出的工件id，近上料机械臂有值，其他为None
        self.is_ng: int = is_ng  # 当前任务的输出状态。 1: OK, 0: NG, -1: 无输出
        self.task_time: float = task_time  # 任务耗时
        self.feature_switch_time: float = task_switch_time  # 切换功能的用时
        self.timer: int = timer  # 任务计时器
        self.failure_rate: float = failure_rate  # 机械臂的故障率，用于模拟设备故障
        self.recovery_rate: float = recovery_rate  # 机械臂的恢复率，用于模拟故障后的设备恢复
        self.ng_rate: float = ng_rate  # 质检时的ng率，用于模拟质检过程中的瑕疵品检出
        self.task_info_mapping: dict = task_info_mapping  # 任务映射表，用于切换任务后的参数更新

        # target_list, target_types 示例：
        # 假设存在两类工件，id为0， 1，当前机械臂需要将两种工件进行拼装，则：
        # target_list=[1, 1], index代表工件id， 0-1代表是否需要. 当任务类型不变时，target_list也不会改变。
        # 如果此时机械臂上已经拥有id=0的工件，仍然需要id=1的工件，则此时target_types = [0, 1]。target_type会随着机械臂上工件的不同而变化

        # task_info_mapping变量的结构示例:
        # {
        #     "task_id_1": {
        #          "available_task_ids": [1,2,3],
        #         "target_types", "[1, 0, 1]",
        #         "target_list": "[1, 0, 1",
        #         "have_output": True,
        #         "output_type": 1
        #         "available_outputs": [1,2],
        #         "is_ng": -1,
        #         "task_time": 10.0,
        #         "feature_switch_time": 1.0
        #     }，
        #     “task_id_2”: {...}
        # }

    def reset(self):
        """
        重置机械臂的状态
        """
        logging.info(f'开始重置机械臂(id={self.id})的状态')
        self.switch_task(new_task_id=self.init_task_id)

    def execute_task_check(self, task_id, start_robot_arm) -> bool:
        """
        检查机械臂是否可以执行当前任务。如果可以执行，返回True，否则返回False
        检查流程如下：
        0. 判断起始机械臂是否为虚拟机械臂，虚拟机械臂不进行判定，直接返回True
        1. 检查任务id是否在task_list中(self.task_list)；
        2. 检查机械臂状态是否为空闲(self.state)；
        3. 检查机械臂是否被动子占用(self.is_occupied)
        4. 检查任务类型是否为上料动作
        5. 对于上料料动作：
            5.1 检测机械臂是否有输出（self.have_output）
        6. 对于下料动作：
            6.1 检查起点机械臂是否有输出（start_robot_arm.have_output）
            6.2 检测起点机械臂的输出是否是终点机械臂需要的类型（self.target_types）
            6.3 检测起点机械臂的输出是否有瑕疵（start_robot_arm.is_ng）
        7. 返回结果
        """
        loading_action_list = config.get_setting("loading_action_list")  # 上料动作列表
        visual_robot_arm_ids = config.get_setting("visual_robot_arm_ids")

        if start_robot_arm.id in visual_robot_arm_ids:
            return True
        else:
            logging.info(f'开始检查机械臂(id={self.id})是否可以执行当前任务(id={self.task_id})')
            if task_id not in self.task_list:  # step 1
                logging.info('机械臂任务类型不匹配，无法执行该任务')
                return False
            if self.state != 0:  # step 2
                logging.info('机械臂正忙，无法执行该任务')
                return False
            if self.is_occupied is True:  # step 3
                logging.info('机械臂被其他动子占用，无法执行该任务')
                return False
            if task_id in loading_action_list:  # step 4
                logging.info('当前任务含上料动作')
                if self.have_output is False:  # step 5
                    logging.info('机械臂没有输出信息，无法执行该任务')
                    return False
            else:
                logging.info('当前任务含下料动作')
                if (self.target_types is not None) and self.target_types[start_robot_arm.output_type] == 0:  # step 6
                    logging.info('传入的工件与机械臂的需求不符，无法执行该任务')
                    return False
                if start_robot_arm.have_otuput is False:
                    logging.info('没有传入的工件，无法执行该任务')
                    return False
                if start_robot_arm.is_ng == 1:
                    logging.info('传入的工件存在瑕疵，无法执行该任务')
                    return False
            logging.info('检查通过，机械臂可以执行该任务')
            return True

    def execute_task(self, task_id: str, start_robot_arm: "RobotArm"):
        """
        执行任务，更新机械臂状态，计算和返回总成本：时间成本+任务切换成本（如有）。
        工作步骤：
        1. 检查是否需要切换任务，如切换任务，记录切换成本
        2. 一般属性更新：
            2.1 动子改为占用状态
            2.2 状态改为工作中
            2.3 如果需要多个处理对象，更新处理对象的需求信息
            2.4 激活timer计数
        3. 基于任务类型的属性更新：
            3.1 上料任务
                3.1.1 机械臂从外部夹取工件，确保have_output一致为True
            3.2 质检任务
                3.2.1 对于上料动作，模拟质检过程，更新输出相关属性（have_output, output_type, is_ng）
                3.2.2 对于下料动作，更新输出相关属性（have_output, output_type, is_ng）
            3.3 拼装任务
                3.3.1 对于上料动作，更新输出相关属性（have_output, output_type, is_ng）
                3.3.2 对于下料动作，更新输出相关属性（have_output, output_type, is_ng）
            3.4 下料任务
                无任务属性更新
        4. 计算时间成本并返回
        """
        loading_task_list = config.get_setting("loading_task_list")  # 上料任务列表
        checking_task_list = config.get_setting("checking_task_list")  # 质检任务列表
        assembly_task_list = config.get_setting("assembly_task_list")  # 拼装任务列表
        loading_action_list = config.get_setting("loading_action_list")  # 上料动作列表

        # 检查是否需要切换任务
        switch_cost = 0
        if task_id not in self.available_task_ids:
            logging.info(f'开始切换任务。当前任务类型：(id={self.task_id}) -> 目标任务类型：(id={task_id})')
            self.switch_task(new_task_id=task_id)
            switch_cost = self.feature_switch_time

        # 执行任务，更新状态
        self.is_occupied = True  # 动子设为占用状态
        self.state = 1  # 状态更改为工作中
        if self.target_types is not None:
            self.target_types[start_robot_arm.output_type] -= 1  # 对应处理对象更新
        self.timer = 1  # 计数器开始工作

        # 上料类任务的状态更新
        if task_id in loading_task_list:
            self.have_output = True

        # 质检类任务的状态更新
        if task_id in checking_task_list:
            if task_id in loading_action_list:  # 如果是上料动作（指机械臂将其处理工件放置到动子上）
                self.have_output = False  # 更新输出状态（动子上料后，机械臂为空，无法再次上料）
                self.output_type = None
                self.is_ng = -1
            else:
                self.have_output = True
                self.output_type = start_robot_arm.output_type  # 任务输出的工件id取决于动子传入的结果
                # 模拟质检结果
                if random.random() < self.ng_rate:
                    self.is_ng = 1
                else:
                    self.is_ng = 0

        # 组装类任务的状态更新
        if task_id in assembly_task_list:
            if task_id in loading_action_list:
                self.have_output = False
                self.output_type = None
                self.is_ng = -1
            else:
                self.have_output = True
                self.output_type = start_robot_arm.output_type  # 任务输出的工件id取决于动子传入的结果
                self.is_ng = -1

        # 返回结果
        task_cost = self.task_time  # 记录执行结果
        return switch_cost + task_cost

    def update(self):
        """
        更新机械臂状态的函数，通过本函数来进行机械臂状态的退出。
        步骤：
        1. 检测timer是否为0，如果不为0，每次请求后数值+1
        2. 模拟故障恢复。如果机械臂处于故障或离线状态，基于内置概率判定是否恢复
        3. 模拟故障。如果机械臂状态不是故障或者离线，基于内置概率判定是否发生故障
        4. 检测机械臂的状态
            4.1 如果机械臂为空闲状态，且timer不为0，将timer重置为0
            4.2 如果机械臂为工作中状态，比较timer和task_time的数字大小，timer大于等于task_time时，退出工作状态并更新其他状态
        """
        failure_states = config.get_setting("failure_states")  #
        normal_states = config.get_setting("normal_states")
        loading_task_list = config.get_setting("loading_task_list")  # 上料任务列表


        # 更新timer状态,如果timer不为0，开始计数
        if self.timer != 0:
            self.timer += 1

        # 模拟故障恢复
        if self.state in failure_states:
            if random.random() < self.recovery_rate:
                self.state = 0
                self.reset()
                logging.info(f'机械臂（id={self.id}）已经从故障中恢复')

        # 模拟故障
        if self.state in normal_states:
            if random.random() < self.failure_rate:
                failure_id = random.choice(failure_states)
                self.state = failure_id
                logging.info(f'机械臂（id={self.id}）发生了故障，故障代码为：{failure_id}')

        # 正常状态的更新和退出
        if self.state == 0 and self.timer != 0:
            self.timer = 0
        if self.state == 1 and self.timer >= self.task_time:
            self.state = 0
            if self.task_id not in loading_task_list:
                self.have_output = False
                self.is_ng = 0
                self.output_type = -1
            else:
                self.output_type = random.choice(self.available_outputs)
            logging.info(f'机械臂（id={self.id}）已经完成了任务（id={self.task_id}）')

    def switch_task(self, new_task_id: str):
        """
        模拟任务切换并更新机械臂状态.
                #          "available_task_ids": [1,2,3],
        #         "target_types", "[1, 0, 1]",
        #         "target_list": "[1, 0, 1",
        #         "have_output": True,
        #         "output_type": 1
        #         "available_outputs": [1,2],
        #         "task_time": 10.0,
        #         "task_switch_time": 1.0
        """
        self.task_id = new_task_id
        self.state = 0

        task_details = self.task_info_mapping.get(self.task_id, {})
        self.available_task_ids = task_details.get('available_task_ids', None)
        self.target_types = task_details.get("target_types", None)
        self.target_list = task_details.get("target_list", None)
        self.have_output = task_details.get("have_output", False)
        self.output_type = task_details.get("output_type", None)
        self.available_outputs = task_details.get("available_outputs", None)
        self.is_ng = task_details.get("is_ng", -1)
        self.task_time = task_details.get("task_time", 0)
        self.feature_switch_time = task_details.get("feature_switch_time", 1)

    def get_state(self) -> dict:
        """
        返回机械臂的当前状态
        """
        return {
            "id": self.id,
            "available_task_ids": self.available_task_ids,
            "task_list": self.task_list,
            "target_types": self.target_types,
            "target_list": self.target_list,
            "location": self.location,
            "state": self.state,
            "is_occupied": self.is_occupied,
            "have_output": self.have_output,
            "failure_rate": self.failure_rate,
            "output_type": self.output_type,
            "is_ng": self.is_ng,
            "task_time": self.task_time,
            "timer": self.timer
        }

