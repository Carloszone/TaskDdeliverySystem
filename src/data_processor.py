import numpy as np
from config_manager import config

class DataProcessor:
    """
    用于处理原始数据，将其转换为agent可接受数据的处理器
    """
    def __init__(self):
        """
        初始化数据处理器
        """
        self.config = config["feature_info"]

    def onehot_encoder(self, original_data: [list, int, str, float, None], feature_name: str) -> list:
        """
        基于类型值对输出的数据进行0-1转码。

        original_data: 需要被转码的原始数据
        feature_name: 变量名称，用于在配置文件中检索对应信息
        """
        # 将 original_data 转换为列表，以便统一处理
        if not isinstance(original_data, list):
            original_data = [original_data]

        # 提取变量的映射字典
        mapping_dict = self.config[feature_name]
        unique_value_lens = len(mapping_dict)

        # 生成一个和all_element长度一致的列表
        one_hot_vector = [0] * unique_value_lens

        # 生成0-1向量
        for item in original_data:  # 遍历原始数据，将对应位置设置为 1
            try:
                index = mapping_dict(item)
                one_hot_vector[index] = 1
            except ValueError:
                print(f"警告:发现异常值'{item}' 不在设定的可能值列表中，将被忽略。")
        return one_hot_vector

    def flow_to_adjacency_matrix(self, process_flows: list) -> np.ndarray:
        # 从配置中提取信息
        flow_dict = self.config.get_setting("flow_mapping_dict")
        flow_step_num = self.config.get_setting("flow_step_num")

        # 构建空邻接矩阵
        adj_matrix = np.zeros((flow_step_num, flow_step_num))

        # 填充邻接矩阵
        for flow in process_flows:
            for index, step in enumerate(flow):
                current_index = flow_dict[step]
                if index < len(flow) - 1:
                    value_index = flow_dict[flow[index+1]]

                    # 更新邻接矩阵
                    adj_matrix[current_index, value_index] = 1

        # 返回结果
        return adj_matrix

    def robot_arm_state_processor(self, observation: dict) -> (list, list):
        """
        用于处理机台信息的的处理器
        """
        # 信息分离
        num_features = self.config["num_feature_names"]
        cat_features = self.config["cat_feature_names"]

        # 信息处理:数值型变量
        num_obs = []
        for feature in num_features:
            num_obs = num_obs.append(feature)

        # 信息处理：类别变量
        cat_obs = []
        for feature in cat_features:
            cat_obs += self.onehot_encoder(observation[feature], feature)

        # 返回结果
        return num_obs, cat_obs

