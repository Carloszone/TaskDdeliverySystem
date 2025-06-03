import random
import logging
from config_manager import config
import sys


class RobotArm:
    """
    模拟工业场景下机械臂工作和交互的类.
    """

    def __init__(self, robot_id: str, role_id, role_list, location: float, timer: int = 0, failure_rate: float = 0.0,
                 recovery_rate: float = 0.0, ng_rate: float = 0.01, role_data_mapping=None):
        """
        初始化一个机械臂实例.

        notes:
        1. 假设机械臂切换所有任务的耗时一致，均为1.0
        2. 工件在动子和机械臂之间的移动（上下料）耗时设置为0
        3. 不考虑动子的运动过程，只计算运动时间作为成本
        4. 对于非专门上下料的机械臂，其主要任务中包含下料操作，但是上料操作独立。以质检流程为例： 机械臂可以执行的两个任务为： （下料）质检， 上料
        5. 每次请求状态视为时间流逝1s，AKA 上位系统以每秒一次的频率请求状态信息并进行任务分配
        6. 初始上料机械臂只能进行一种产品的上料操作
        7. 机械臂的角色切换仅在质检/拼装两者之间
        8. 动子下料后，空载动子会自动回去等候区
        9. 机台的视觉系统可以分别和切换下料/质检/拼装的模式
        """
        # 固有属性
        self.id: str = robot_id  # 机械臂的唯一标识
        self.role_id: str = role_id  # 机械臂分配的角色id
        self.init_role_id: str = role_id  # 机械臂的初始角色id
        self.role_list: list = role_list  # 机械臂可承接的角色id
        self.location: float = location  # 机械臂所在的位置（离动子起点）的距离值，用于标识机械臂的空间次序
        self.timer: int = timer  # 任务计时器
        self.failure_rate: float = failure_rate  # 机械臂的故障率，用于模拟设备故障
        self.recovery_rate: float = recovery_rate  # 机械臂的恢复率，用于模拟故障后的设备恢复
        self.ng_rate: float = ng_rate  # 质检时的ng率，用于模拟质检过程中的瑕疵品检出
        self.role_data_mapping: dict = role_data_mapping  # 任务映射表，用于切换任务后的参数更新
        self.all_task_ids: list = []  # 机械臂可以承接的所有任务id，由role_list决定
        for role in role_list:
            self.all_task_ids += role_data_mapping[role]['task_list']

        # 基于role更新属性
        _role_data = role_data_mapping[role_id]
        self.task_list: list = _role_data['task_list']  # 机械臂在当前角色下的所有可执行的任务类型id
        self.target_list: list = _role_data['target_list']  # 可处理对象列表,长度等于所有需处理工件的总类型。1表示需要，0表示不需要。如果不需要处理对象，则全为0。
        self.available_outputs: list = _role_data['available_outputs']  # 可以输出的工件ids
        self.feature_switch_time: float = _role_data['feature_switch_time']  # 切换功能的用时
        self.main_task_id = _role_data["main_task_id"]  # 机械臂的主要任务id，如果没有，值为None
        self.accept_ng = 0  # 机械臂是否接受ng输入 0：不接受， 1：接受

        # 基于task更新的属性
        _mian_task_id = role_data_mapping[role_id]['main_task_id']
        _main_task_data = role_data_mapping[role_id]['task_data_mapping'][_mian_task_id]
        self.task_id: str = _mian_task_id  # 当前执行的任务类型id
        self.task_time: float = _main_task_data['task_time']  # 任务耗时
        self.default_output_id = _main_task_data['default_output_id']  # 仅拼装和上料角色的机台，输出拼装后的产品id，其他其他的值为-1

        # 基于任务开始/完成更新的属性
        self.state: int = 0  # 0: 空闲, 1: 工作中, 2: 故障, 3: 离线
        self.have_output: int = _main_task_data['have_output']  # 机台是否有输出 0：没有； 1：有
        self.is_occupied: int = 0  # 机械臂的动子位是否被占用。0：未被占用， 1：被占用
        self.target_ids: list = _role_data['target_list']  # 机械臂需要的工件ids，基于target_list更新,长度等于所有需处理工件的总类型数
        self.is_ng: int = 0  # 当前任务的输出状态。 1: NG, 0: OK, 默认为OK
        self.output_id: int = _main_task_data['output_id']  # 输出的工件id，没有则为-1。

        # target_list, target_ids 示例：
        # 假设存在两类工件，id为0， 1，当前机械臂需要将两种工件进行拼装，则：
        # target_list=[1, 1], index代表工件id， 0-1代表是否需要. 当机械臂角色类型不变时，target_list也不会改变。
        # 如果此时机械臂上已经拥有id=0的工件，仍然需要id=1的工件，target_ids = [0, 1]。target_ids会随着机械臂上工件的不同而变化

        # role_data_mapping 示例:
        # {
        #     "role_id_1": {
        #          "task_list": [1,2,3],
        #         "target_list", [1, 0, 1],
        #         "available_outputs": [1, 0, 1],
        #         "feature_switch_time": 2.0
        #         "main_task_id": 123,
        #         "accept_ng" 0,
        #         "task_data_mapping": {
        #             "task_id_1": {
        #                  "task_time": 10.0,
        #                  "target_ids" : [],
        #                  "have_output": -1,
        #                  "output_id": -1,
        #                  "is_ng": -1,
        #                  "output_id": -1
        #             },
        #             "task_id_2": {},
        #         }

        #     }，
        #     “role_id_2”: {...}
        # }

    def reset(self):
        """
        重置机械臂的状态
        """
        print(f'开始重置机械臂(id={self.id})的状态')
        self.switch_role(new_role_id=self.init_role_id)

    def execute_task_check(self, task_id, start_robot_arm) -> bool:
        """
        检查机械臂是否可以执行当前任务，并返回检查信息。
        检查流程如下：
        1. 对于起点机械臂：
            1.1 检查是否为虚拟机械臂
                1.1.1 虚拟机械臂一定可以执行
                1.1.2 非虚拟机械臂，执行任务必须满足：（1）空闲状态（state）；（2）被动子占用(is_occupied)；（3）有输出id(output_id)
        2. 对于终点机械臂：
            2.1 检查终点机械臂是否为虚拟机械臂
                2.1.1 虚拟机械臂任务无法执行
            2.2 对于非虚拟机械臂,必须满足以下条件：
                2.2.1 上料任务：任务id正确，机台空闲， 机台有输出，机台没有被占用
                2.2.2 其他任务（质检，拼装等）：不直接和动子任务交互
                2.2.3 下料任务：任务id正确，机台空闲，机台没有输出，机台没有被占用,有需要的输出对象
        3. 返回结果

        return格式： bool
        """
        # 变量准备
        start_robot_arm_id = start_robot_arm.id
        visual_robot_arm_id = config.get_setting("visual_robot_arm_id")  # 虚拟机台编号
        loading_task_list = config.get_setting("loading_task_list")  # 上料动作列表
        unloading_task_list = config.get_setting("unloading_task_list")  # 下料动作列表
        check_task_list = config.get_setting("check_task_list")  # 质检任务列表
        assembly_task_list = config.get_setting("assembly_task_list")  # 组装任务列表

        # 有效性检测
        if self.id == start_robot_arm_id:  # 如果是起点机械臂
            if self.id != visual_robot_arm_id:  # 如果是非虚拟机械臂
                if self.state != 0:  # 起点机台仍在工作中，无法执行
                    print(f'起点机台处于非空闲状态（状态id={self.state}），起点无效')
                    return False
                if self.is_occupied == 0:  # 起点机台未被动子占用，无法执行
                    print(f'起点机台已被其他动子占用，起点无效')
                    return False
                if self.output_id == -1:  # 起点机台没有输出id，无法执行
                    print(f'起点机台没有输出信息，起点无效')
                    return False
            print('起点机台有效')
            return True
        else:  # 如果是终点机械臂
            if self.id == visual_robot_arm_id:  # 如果终点机台是虚拟机械臂，无法执行
                print('终点机台不能为虚拟机台')
                return False
            else:  # 如果是非虚拟机械臂
                if task_id not in self.all_task_ids:  # 机械臂无法完成该任务
                    print(f'机台无法处理该任务')
                    return False
                if self.state != 0:  # 终点机台不是空闲状态，无法承接新任务
                    print(f'机台处于非空闲状态（状态id={self.state}），无法接受新任务')
                    return False
                if self.is_occupied == 1:  # 终点机台被动子占用，无法承接新任务
                    print(f'机台已被其他动子占用，无法接受新任务')
                    return False
                if task_id in loading_task_list:
                    if self.have_output == 0:  # 对于上料任务，没有输出的机台无法执行该任务
                        print(f'机台没有输出信息，无法承接该任务')
                        return False
                if task_id in unloading_task_list:
                    if self.have_output == 1:  # 对于下料任务，有输出意味者机台等待动子搬运走输出，无法执行下料任务
                        print(f'机台有输出对象等待运走，无法承接该任务')
                        return False
                    if sum(self.target_ids) == 0:  # 机械臂不需要输出对象，无法执行下料任务
                        print(f'机台暂时不需要其他对象，无法承接该任务')
                        return False
                    if start_robot_arm.is_ng == 1 and self.accept_ng == 0:  # 检测到NG输入，无法执行后续任务
                        print(f'动子上的对象为NG状态，机台不接受')
                        return False
                    self.output_id = start_robot_arm.output_id  # 下料任务需要记录起点机台的output_id
                if task_id in check_task_list or task_id in assembly_task_list:  # 动子与机械臂的交互只有上下料
                    print(f'检测到不合法的任务id')
                    return False
        print("终点机台有效")
        return True

    def execute_task(self, task_id: str):
        """
        执行任务，更新机械臂状态，计算和返回总成本：时间成本+任务切换成本（如有）。
        工作步骤：
        1. 检查是否需要切换角色
        2. 执行任务，基于角色类型更新状态
        3. 计算时间成本并返回
        """
        loading_role_id = config.get_setting("loading_role_id")  # 上料角色编号
        check_role_id = config.get_setting("check_role_id")  # 质检角色编号
        assembly_role_id = config.get_setting("assembly_role_id")  # 组装角色编号
        loading_task_list = config.get_setting("loading_task_list")  # 上料任务列表
        check_task_list = config.get_setting("check_task_list")  # 质检任务列表
        assembly_task_list = config.get_setting("assembly_task_list")  # 组装任务列表
        unloading_task_list = config.get_setting("unloading_task_list")  # 下料动作列表
        switch_role_cost = 0

        # 检测是否切换任务，如果需要，切换角色
        if task_id not in self.task_list:
            print(f'需要切换机械臂的任务角色')
            for role_id in self.role_list:
                if role_id != self.role_id and task_id in self.role_data_mapping[role_id]["task_list"]:
                    print(f"执行角色切换：role_id={self.role_id} --> role_id={role_id}")
                    self.switch_role(role_id)
                    switch_role_cost = self.feature_switch_time

        # 通用状态修改
        self.task_id = task_id
        self.state = 1
        self.have_output = 0
        self.is_occupied = 1
        self.timer = 1

        if task_id in loading_task_list:
            self.output_id = random.choice(self.available_outputs)  # 如果是混检上料角色机台，随机生成产品id
            if self.role_id == loading_role_id:  # 对于上料机台，have_output永远为1
                self.have_output = 1

        if task_id in check_task_list:
            if random.random() < self.ng_rate:
                self.is_ng = 1
            else:
                self.is_ng = 0

        if task_id in assembly_task_list:
            pass

        if task_id in unloading_task_list:
            if self.role_id == check_role_id:
                self.output_id = -1

        # 返回结果
        task_cost = self.task_time  # 记录执行结果
        return switch_role_cost + task_cost

    def update(self):
        """
        更新机械臂状态的函数，通过本函数来进行机械臂状态的退出。
        步骤：
        1. 检测timer是否为0，如果不为0，每次请求后数值+1
        2. 模拟故障恢复。如果机械臂处于故障或离线状态，基于内置概率判定是否恢复
        3. 模拟故障。如果机械臂状态不是故障或者离线，基于内置概率判定是否发生故障
        4. 模拟机台状态退出，更新状态
        """
        failure_states = config.get_setting("failure_states")
        normal_states = config.get_setting("normal_states")
        loading_task_list = config.get_setting("loading_task_list")  # 上料任务列表
        check_task_list = config.get_setting("check_task_list")  # 质检任务列表
        assembly_task_list = config.get_setting("assembly_task_list")  # 组装任务列表
        unloading_task_list = config.get_setting("unloading_task_list")  # 下料任务列表
        loading_role_id = config.get_setting("loading_role_id")  # 上料角色id
        check_role_id = config.get_setting("check_role_id")  # 质检角色id
        assembly_role_id = config.get_setting("assembly_role_id")  # 拼装角色id
        product_mapping_index = config.get_setting("product_mapping_index")  # 产品id的编码映射表

        # 更新timer状态,如果timer不为0，开始计数
        if self.timer != 0:
            self.timer += 1

        # 模拟故障恢复
        if self.state in failure_states:
            if random.random() < self.recovery_rate:
                self.state = 0
                self.reset()
                print(f'机械臂（id={self.id}）已经从故障中恢复')

        # 模拟故障
        if self.state in normal_states:
            if random.random() < self.failure_rate:
                failure_id = random.choice(failure_states)
                self.state = failure_id
                print(f'机械臂（id={self.id}）发生了故障，故障代码为：{failure_id}')

        # 正常状态的更新和退出
        if self.state == 1 and self.timer >= self.task_time:
            print(f'机台(id={self.id})完成了任务(id={self.task_id})')
            self.state = 0  # 更新状态信息
            self.timer = 0  # 更新计数信息
            if self.task_id in loading_task_list:  # 上料任务，动子上料后：（1）除上料角色机台外，其他机台的输出为空；（2）拼装机台的target_ids重置
                self.is_occupied = 1
                if self.role_id == loading_role_id:
                    self.have_output = 1
                else:
                    self.have_output = 0
                if self.role_id == assembly_role_id or self.role_id == check_role_id:
                    self.target_ids = self.target_list

            if self.task_id in unloading_task_list:  # 下料任务，动子下料后：（1）拼装和质检角色机台的target_ids基于output_id更新；（2）拼装机台的output_id更新值
                self.is_occupied = 0
                product_mapping_index = product_mapping_index[self.output_id]
                if self.role_id == check_role_id:
                    self.target_ids[product_mapping_index] = 0
                    self.execute_task(task_id=check_task_list[0])
                if self.role_id == assembly_role_id:
                    self.target_ids[product_mapping_index] = 0
                    self.output_id = self.default_output_id  # 对于质检类机台，其输出output_id由动子传入的output_id决定；而拼装机台有特殊的output_id
                    self.execute_task(task_id=assembly_task_list[0])

    def switch_role(self, new_role_id: str):
        """
        模拟机械臂角色切换并更新机械臂状态.
        """
        #
        self.role_id = new_role_id

        # 基于role更新属性
        role_data = self.role_data_mapping[new_role_id]
        self.task_list: list = role_data['task_list']  # 机械臂在当前角色下的所有可执行的任务类型id
        self.target_list: list = role_data['target_list']  # 可处理对象列表,长度等于所有需处理工件的总类型。1表示需要，0表示不需要。如果不需要处理对象，则全为0。
        self.available_outputs: list = role_data['available_outputs']  # 可以输出的工件ids
        self.feature_switch_time: float = role_data['feature_switch_time']  # 切换功能的用时
        self.main_task_id = role_data["main_task_id"]  # 机械臂的主要任务id，如果没有，值为None
        self.accept_ng = 0  # 机械臂是否接受ng输入 0：不接受， 1：接受

        # 基于task更新的属性
        mian_task_id = self.role_data_mapping[new_role_id]['main_task_id']
        main_task_data = self.role_data_mapping[new_role_id]['task_data_mapping'][mian_task_id]
        self.task_id: str = mian_task_id  # 当前执行的任务类型id
        self.task_time: float = main_task_data['task_time']  # 任务耗时
        self.default_output_id = main_task_data['default_output_id']  # 仅拼装和上料角色的机台，输出拼装后的产品id，其他其他的值为-1

        # 基于任务开始/完成更新的属性
        self.state: int = 0  # 0: 空闲, 1: 工作中, 2: 故障, 3: 离线
        self.have_output: int = main_task_data['have_output']  # 机台是否有输出 0：没有； 1：有
        self.is_occupied: int = 0  # 机械臂的动子位是否被占用。0：未被占用， 1：被占用
        self.target_ids: list = main_task_data['target_ids']  # 机械臂需要的工件ids，基于target_list更新,长度等于所有需处理工件的总类型数
        self.is_ng: int = 0  # 当前任务的输出状态。 1: NG, 0: OK, 默认为OK
        self.output_id: int = main_task_data['output_id']  # 输出的工件id，没有则为-1。

    def get_state(self) -> dict:
        """
        返回机械臂的当前状态
                # 固有属性
        self.id: str = robot_id  # 机械臂的唯一标识
        self.role_id: str = role_id  # 机械臂分配的角色id
        self.init_role_id: str = role_id  # 机械臂的初始角色id
        self.role_list: list = role_list  # 机械臂可承接的角色id
        self.location: float = location  # 机械臂所在的位置（离动子起点）的距离值，用于标识机械臂的空间次序
        self.timer: int = timer  # 任务计时器
        self.failure_rate: float = failure_rate  # 机械臂的故障率，用于模拟设备故障
        self.recovery_rate: float = recovery_rate  # 机械臂的恢复率，用于模拟故障后的设备恢复
        self.ng_rate: float = ng_rate  # 质检时的ng率，用于模拟质检过程中的瑕疵品检出
        self.role_data_mapping: dict = role_data_mapping  # 任务映射表，用于切换任务后的参数更新
        self.all_task_ids: list = []  # 机械臂可以承接的所有任务id，由role_list决定
        for role in role_list:
            self.all_task_ids += role_data_mapping[role]['task_list']

        # 基于role更新属性
        role_data = role_data_mapping[role_id]
        self.task_list: list = role_data['task_list']  # 机械臂在当前角色下的所有可执行的任务类型id
        self.target_list: list = role_data['target_list']  # 可处理对象列表,长度等于所有需处理工件的总类型。1表示需要，0表示不需要。如果不需要处理对象，则全为0。
        self.available_outputs: list = role_data['available_outputs']  # 可以输出的工件ids
        self.feature_switch_time: float = role_data['feature_switch_time']  # 切换功能的用时
        self.main_task_id = role_data["main_task_id"]  # 机械臂的主要任务id，如果没有，值为None
        self.accept_ng = 0  # 机械臂是否接受ng输入 0：不接受， 1：接受

        # 基于task更新的属性
        mian_task_id = role_data_mapping[role_id]['main_task_id']
        main_task_data = role_data_mapping[role_id]['task_data_mapping'][mian_task_id]
        self.task_id: str = mian_task_id  # 当前执行的任务类型id
        self.task_time: float = main_task_data['task_time']  # 任务耗时
        self.default_output_id = main_task_data['default_output_id']  # 仅拼装和上料角色的机台，输出拼装后的产品id，其他其他的值为-1

        # 基于任务开始/完成更新的属性
        self.state: int = 0  # 0: 空闲, 1: 工作中, 2: 故障, 3: 离线
        self.have_output: int = main_task_data['have_output']  # 机台是否有输出 0：没有； 1：有
        self.is_occupied: int = 0  # 机械臂的动子位是否被占用。0：未被占用， 1：被占用
        self.target_ids: list = main_task_data['target_ids']  # 机械臂需要的工件ids，基于target_list更新,长度等于所有需处理工件的总类型数
        self.is_ng: int = 0  # 当前任务的输出状态。 1: NG, 0: OK, 默认为OK
        self.output_id: int = main_task_data['output_id']  # 输出的工件id，没有则为-1。
        """
        return {
            "id": self.id,  # str
            "role_id": self.role_id,  # str
            "role_list": self.role_list,  # list[str]
            "location": self.location,  # float
            "all_task_ids": self.all_task_ids,  # list[str]
            "accept_ng": self.accept_ng,  # bool
            "task_id": self.task_id,  # str
            "state": self.state,  # int
            "have_output": self.have_output,  # bool
            "is_occupied": self.is_occupied,  # bool
            "target_ids": self.target_ids,   # list[str]
            "is_ng": self.is_ng,  # bool
            "output_id": self.output_id  # str
        }


if __name__ == '__main__':
    # 生成五类机械臂

    role_data_mapping = {
        "VR": {
            "task_list": [],
            "target_list": [],
            "available_outputs": [],
            "feature_switch_time": 0,
            "main_task_id": "0",
            "accept_ng": 0,
            "task_data_mapping": {
                "0": {
                    "task_time": 0,
                    "default_output_id": -1,
                    "have_output": 0,
                    "output_id": -1,
                }
            }
        },
        "LR": {
            "task_list": ["1"],
            "target_list": [0, 0, 0],
            "available_outputs": ["A"],
            "feature_switch_time": 2.0,
            "main_task_id": "1",
            "accept_ng": 0,
            "task_data_mapping": {
                "1": {
                    "task_time": 5.0,
                    "default_output_id": "A",
                    "have_output": 1,
                    "output_id": "A",
                }
            }
        },
        "CR": {
            "task_list": ["2"],
            "target_list": [1, 0, 0],
            "available_outputs": ["A"],
            "feature_switch_time": 2.0,
            "main_task_id": "2",
            "accept_ng": 0,
            "task_data_mapping": {
                "2": {
                    "task_time": 10.0,
                    "default_output_id": -1,
                    "have_output": 1,
                    "output_id": -1,
                }
            }
        },
        "AR": {
            "task_list": ["3"],
            "target_list": [1, 0, 0],
            "available_outputs": ["AB"],
            "feature_switch_time": 2.0,
            "main_task_id": "3",
            "accept_ng": 0,
            "task_data_mapping": {
                "3": {
                    "task_time": 15.0,
                    "default_output_id": "AB",
                    "have_output": 1,
                    "output_id": "AB",
                }
            }
        },
        "UR": {
            "task_list": ["4"],
            "target_list": [1, 0, 1],
            "available_outputs": [],
            "feature_switch_time": 2.0,
            "main_task_id": "4",
            "accept_ng": 0,
            "task_data_mapping": {
                "4": {
                    "task_time": 15.0,
                    "default_output_id": -1,
                    "have_output": 0,
                    "output_id": -1,
                }
            }
        }
    }

    r0 = RobotArm(robot_id="R0", role_id="VR", role_list=["VR"], location=0, role_data_mapping=role_data_mapping)
    r1 = RobotArm(robot_id="R1", role_id="LR", role_list=["LR"], location=0, role_data_mapping=role_data_mapping)
    r2 = RobotArm(robot_id="R2", role_id="CR", role_list=["CR", "AR"], location=0, role_data_mapping=role_data_mapping)
    r3 = RobotArm(robot_id="R3", role_id="AR", role_list=["CR", "AR"], location=0, role_data_mapping=role_data_mapping)
    r4 = RobotArm(robot_id="R4", role_id="UR", role_list=["UR"], location=0, role_data_mapping=role_data_mapping)

    # 获取R1状态信息
    print('初始R1信息', r1.get_state())

    # 执行任务 R0-R1
    res_1 = r0.execute_task_check(task_id="1", start_robot_arm=r0)
    res_2 = r1.execute_task_check(task_id="1", start_robot_arm=r0)
    print(res_1, res_2)

    if res_1 and res_2:
        r1.execute_task(task_id="1")
    print("R1执行上料后的信息", r1.get_state())

    # 执行任务R1-R2
    print("初始R2信息：", r2.get_state())
    res_3 = r1.execute_task_check(task_id="2", start_robot_arm=r1)
    res_4 = r2.execute_task_check(task_id="2", start_robot_arm=r1)

    print(res_3, res_4)
    if res_3 and res_4:
        r2.execute_task(task_id="2")
    print("R2执行上料后的信息", r2.get_state())

    # 执行任务R2-R3
    print("初始R3信息：", r3.get_state())
    res_5 = r2.execute_task_check(task_id="3", start_robot_arm=r2)
    res_6 = r3.execute_task_check(task_id="3", start_robot_arm=r2)

    print(res_5, res_6)
    if res_5 and res_6:
        r3.execute_task(task_id="3")
    print("R3执行上料后的信息", r3.get_state())

    # 执行任务R3-R4
    print("初始R4信息：", r4.get_state())
    res_7 = r3.execute_task_check(task_id="4", start_robot_arm=r3)
    res_8 = r4.execute_task_check(task_id="4", start_robot_arm=r3)

    print(res_7, res_8)
    if res_7 and res_8:
        r4.execute_task(task_id="4")
    print("R4执行上料后的信息", r4.get_state())

