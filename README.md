# Industrial RL Agent: 项目需求文档

本项目旨在开发一个强化学习系统，用于控制工业流水线中的机械臂调度任务。该系统将基于当前工业环境状态做出智能决策，提高整体生产效率，并支持多任务并发、动态任务分配等实际复杂需求。

## 📌 项目背景与目标

### 🎯 模型的预测目标
本模型需要进行工业流水线上的动子任务委派调度工作。在工业生产领域，动子一共有两种调度委派类型：
- 叫车委派：将一个空置动子调往目标机械臂位置。如（R0 -> R2）意为将动子派往R2，其中R0为虚拟机械臂，仅为保持输出格式一致性，不具有实际意义。
- 派车委派：将停于某个特定机械臂的动子，派往指定的机械臂位置。例如（R4 -> R7）意为将R4处的动子派往R7。
 
本模型的目标是：基于系统中所有机械臂工作站状态信息，输出如下内容：

- 动子的起点机械臂 ID（叫车任务的起点为虚拟点 R0）
- 动子的终点机械臂 ID
- 动子在目标工作站执行的任务 ID

### 🧩 当前生产任务种类

#### 1. 混检任务

- 多种工件随机上料（视为同一工件类）
- 经过视觉识别质检
- 合格品下料 / 瑕疵品下料

#### 2. A、B 组装任务

- 产品 A 上料 → 质检 → 与 B 组装 → 再次质检 → 下料
- 产品 B 不需质检，在组装工序参与拼装
- 合格品 / 不合格品分别下料

#### 3. 任务可能混合：同时处理混检任务与 AB 组装任务

### 🔧 可执行的任务与动作

#### 混检任务动作列表：

1. 产品上料  
2. 产品质检  
3. 合格品下料  
4. 瑕疵品下料  

#### A,B 组装任务动作列表：

1. 产品 A 上料  
2. 产品 A 质检  
3. 产品组装（此时上料产品 B）  
4. 产品 AB 质检  
5. 合格品下料  
6. 瑕疵品下料  

#### 特殊动作：

- 等待任务（所有任务都不可行时执行）

## 🧠 强化学习系统架构

本系统设计遵循主流强化学习框架接口（如 `PettingZoo`, `Gymnasium` 等），便于与稳定库集成。

---

## 🔧 核心类定义

### `RobotArm`（机械臂）

**职责：** 模拟机械臂状态与执行逻辑。

**属性：**

- `id`：机械臂唯一标识符  
- `task`：当前任务类型  
- `task_list`：可承接任务类型列表  
- `target_types`, `target_list`：目标工件需求与支持映射  
- `location`：空间位置距离  
- `state`：0:空闲, 1:工作中, 2:故障, 3:离线  
- `is_occupied`：是否被占用  
- `work_output`：任务执行结果 1:OK, 0:NG, -1:无输出  
- `output_type`：输出工件类型  
- `task_time`, `task_switch_time`：执行/切换任务耗时  
- `failure_rate`, `recovery_rate`, `ng_rate`：异常与质检失败概率  
- `task_info_mapping`：任务映射配置表  

**方法：**

- `reset()`
- `execute_task_check()`
- `execute_task()`
- `update()`
- `switch_task()`
- `get_state()`

---

### `RewardCalculator`（奖励计算器）

**职责：** 根据任务执行日志计算奖励。

**属性：**

- `product_completion_reward`
- `step_correct_reward`
- `invalid_robot_arm_id_penalty`
- `invalid_task_id_penalty`

**方法：**

- `calculate_reward()`
- `calculate_product_completion_reward()`
- `calculate_error_penalty()`

**奖励规则设计：**

1. 产品完成工序最后一步：+100  
2. 成功完成任一工序步骤：+20  
3. 任务分配失败（任务错误）：-10  
4. 任务分配失败（机械臂错误）：-20  
5. 常规任务成本 =  
   - 移动成本：`|location_end - location_start|`  
   - 任务切换成本（若发生）  
   - 任务执行耗时（根据类型）

---

### `IndustrialEnv`（工业环境）

**职责：** 环境状态模拟与交互

**属性：**

- `env_config`
- `robot_arm_list`：RobotArm 列表  
- `reward_calculator`
- `step_count`
- `product_count`
- `total_rewards`
- `action_space`, `observation_space`

**方法：**

- `init_env()`
- `reset()`
- `step(action)`
- `add/remove_robot_arm()`
- `init_robot_arm()`
- `define_observation_space()`
- `define_action_space()`
- `parse_action()`
- `execute_action()`
- `update_robot_arms()`
- `get_observation()`
- `log_step()`
- `close()`

---

### `Agent`（智能体）

**职责：** 强化学习的主导者，策略优化器

**属性：**

- 策略网络与价值网络：`action_net`, `critic_net`  
- 延迟更新网络：`target_action_net`, `target_critic_net`  
- 优化器与参数：`Gamma`, `Epsilon`, `Tau`, `learning_rate_actor`, ...  
- 经验缓存：`replay_buffer`, `buffer_size`, `batch_size`, `min_learn_size`  
- 动作掩码生成器：`action_mask_generator`  
- 设备、状态维度等定义参数  

**方法：**

- `choose_action()`
- `store_transition()`
- `learn()`
- `save_checkpoint()`, `load_checkpoint()`

---

### `DataProcessor`（数据处理器）

**职责：** 将原始观测信息转换为模型输入

**属性：**

- `coder`
- `scaler`
- `mapping`

**方法：**

- `observation_processor()`
- `GNN_generator()`

---

## 🧮 模型结构与数据流

### 🧠 输入特征

1. 机台类别变量（C）
2. 机台数值变量（M）
3. 工序图变量（W）

### 📈 网络处理流程

1. 类别变量经过 Transformer 编码  
2. 数值变量与类别变量拼接 → 单个机台特征向量  
3. 添加位置编码（L），拼接成 `N x (C+M+L)` 矩阵  
4. Transformer 提取上下文，得出特征向量 `V_CM`  
5. 工序图变量（W） → Transformer 编码 → `V_W`  
6. 拼接：`V = concat(V_CM, V_W)`  
7. Transformer → 标准化 + 池化 → 得到 `V'`  
8. 三个 `head` 输出：
   - `head1`: softmax → 起点ID  
   - `head2`: softmax → 终点ID  
   - `head3`: softmax（含掩码）→ 执行任务类型  

---

## 🧰 特殊机制与设计建议

### 🌀 虚拟机台设计

- 虚拟机台 R0 代表“起始/等待”点
- `(R0, R2)`：表示“叫车”  
- `(R4, R6)`：表示“派车”  
- `(R8, R0)`：表示“动子离场”  
- 成本由位置决定，环形流水线结构（不可逆）

| 起点 | 终点 | 运输成本 |
|------|------|----------|
| R0   | R1   | 1        |
| R0   | R8   | 8        |
| R1   | R0   | 8        |
| R8   | R1   | 1        |

---

## 🧠 增强建议与补充机制

### ✅ 动作掩码机制

- 在 `softmax` 前对预测分数向量添加掩码（非法动作位置设置为 `-inf`）避免选择非法动作
- 使策略网络训练更稳定、更高效

### 🔄 使用 GAE（Generalized Advantage Estimation）

- 为了克服“奖励稀疏”或“累积不足”问题，采用 GAE 估算优势函数：
  
  ```math
  Â_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...
