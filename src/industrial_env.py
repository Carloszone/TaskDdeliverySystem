import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# Assuming RobotArm and RewardCalculator are in the same directory or proper PYTHONPATH is set
# For this project structure, using relative imports:
from .robot_arm import RobotArm
from .reward_calculator import RewardCalculator

class IndustrialEnv(gym.Env):
    """
    Industrial Environment for controlling robot arms in a simulated assembly line.
    Follows the Gymnasium API.
    """
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the IndustrialEnv.

        Args:
            env_config: Configuration dictionary for the environment.
                        Expected keys:
                        - 'num_robot_arms': Number of robot arms to initialize.
                        - 'robot_arm_configs': List of configurations for each arm (passed to RobotArm constructor).
                        - 'reward_calculator_config': Config for RewardCalculator.
                        - 'max_steps_per_episode': Maximum steps before an episode is truncated.
        """
        super().__init__()
        self.env_config = env_config if env_config is not None else self._get_default_config()

        self.robot_arm_list: List[RobotArm] = []
        self.reward_calculator: RewardCalculator = RewardCalculator(
            **(self.env_config.get("reward_calculator_config", {}))
        )
        self.step_count: int = 0
        self.product_count: int = 0 # Example metric
        self.total_rewards: float = 0.0
        self.max_steps_per_episode: int = self.env_config.get("max_steps_per_episode", 1000)

        self.init_env() # Initialize robot arms and other components

        # Define action and observation spaces (placeholders, to be refined)
        self.action_space: spaces.Space = self.define_action_space()
        self.observation_space: spaces.Space = self.define_observation_space()
        
        self.render_mode = 'ansi' # Default render mode

        print("IndustrialEnv initialized.")

    def _get_default_config(self) -> Dict[str, Any]:
        """Provides a default configuration if none is given."""
        return {
            "num_robot_arms": 2,
            "robot_arm_configs": [
                {"id": "R1", "task_list": ["task1", "task2"], "target_types": {}, "target_list": [], "location": 0, "task_info_mapping": {"task1": {"time":10, "switch_time":2, "output_type":"A"}, "task2": {"time":5, "switch_time":1, "output_type":"B"}}},
                {"id": "R2", "task_list": ["task1", "task2"], "target_types": {}, "target_list": [], "location": 5, "task_info_mapping": {"task1": {"time":10, "switch_time":2, "output_type":"A"}, "task2": {"time":5, "switch_time":1, "output_type":"B"}}},
            ],
            "reward_calculator_config": {},
            "max_steps_per_episode": 200
        }

    def define_observation_space(self) -> spaces.Space:
        """
        Defines the observation space for the environment.
        Placeholder: Represents states of all robot arms.
        Each arm: [state (int), current_task (int, mapped), time_remaining (int), is_occupied (bool), work_output (int)]
        This needs to be significantly more detailed based on README's "输入特征".
        """
        print("Defining observation space...")
        # Example: max 5 arms, each with 5 features.
        # This is a very basic placeholder.
        # For simplicity, let's assume a fixed number of arms for the Box space definition.
        # A more dynamic approach or a Dict space might be better for varying numbers of arms.
        num_features_per_arm = 5 # state, current_task_id, task_time_remaining, is_occupied, work_output
        max_arms = self.env_config.get("num_robot_arms", 2) # Max number of arms for space definition
        
        # Using a Box space: (num_arms * num_features_per_arm)
        # Values need to be normalized or have defined ranges.
        # Example ranges (need actual values based on RobotArm states):
        # state: 0-3
        # current_task_id: 0-N (0 for None, 1 to N for tasks)
        # task_time_remaining: 0-max_task_time
        # is_occupied: 0-1
        # work_output: -1-1
        low = np.array([0, 0, 0, 0, -1] * max_arms, dtype=np.float32)
        high = np.array([3, 10, 100, 1, 1] * max_arms, dtype=np.float32) # Assuming max 10 tasks, max 100 time
        
        # For now, let's use a Dict space for more clarity, accommodating varied number of arms in the future.
        # However, a simple Box is often easier for many RL algos if the number of entities is fixed.
        # The README implies a complex input structure (C, M, W variables).
        # This will require a much more sophisticated observation space.
        # For now, a flat Box representing concatenated arm states.
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def define_action_space(self) -> spaces.Space:
        """
        Defines the action space for the environment.
        Placeholder: (mover_start_arm_id, mover_end_arm_id, task_id_for_end_arm)
        This should align with "模型的预测目标" from README.
        """
        print("Defining action space...")
        # mover_start_arm_id: 0 to N (0 for R0/virtual, 1-N for actual arms)
        # mover_end_arm_id: 1 to N (actual arms)
        # task_id_for_end_arm: 0 to M (0 for wait, 1-M for actual tasks)
        num_arms = len(self.robot_arm_list)
        num_tasks = 10 # Example: Max 10 tasks as per README (task_ids 1-10, plus 0 for wait)

        # Using MultiDiscrete for [start_arm, end_arm, task_id]
        # start_arm: index from 0 (R0) to num_arms (R_N)
        # end_arm: index from 0 (R0, for '动子离场') to num_arms (R_N)
        # task_id: 0 (wait) to num_tasks
        # This needs to be adjusted to exactly match how R0 is handled.
        # Let's assume arm indices are 0 to num_arms-1 for actual arms.
        # R0 can be represented as num_arms index.
        
        # For simplicity, let's assume the action is a tuple/array of three integers:
        # action[0]: start_robot_arm_index (0 for R0, 1 to N for R1 to RN)
        # action[1]: end_robot_arm_index (0 for R0, 1 to N for R1 to RN) - R0 for '动子离场'
        # action[2]: task_id (from README, e.g., 0-10)
        
        # Number of actual arms + 1 for R0
        n_arms_plus_r0 = num_arms + 1 
        
        # Using Discrete space for a single action choice from a pre-generated list of possible actions
        # Or MultiDiscrete if the agent directly outputs the three components.
        # README: "head1: softmax -> 起点ID", "head2: softmax -> 终点ID", "head3: softmax -> 执行任务类型"
        # This suggests the agent outputs probabilities for each part of the action.
        # The environment typically expects a discrete action or a continuous vector.
        # Let's use MultiDiscrete: [start_idx, end_idx, task_id]
        # start_idx: 0..num_arms (0 is R0, 1..num_arms are R1..RN)
        # end_idx: 0..num_arms (0 is R0, 1..num_arms are R1..RN)
        # task_id: 0..num_tasks (0 is wait, 1..num_tasks are actual tasks)
        
        # Let's adjust: max_arm_id should be the actual max ID, not count.
        # If arms are R1...R_N, then num_arms is N.
        # R0 can be considered a special index.
        # If using indices 0..N-1 for N arms, R0 could be N.
        
        # For now, let's assume a simpler discrete action space for selecting one arm and one task for it.
        # This is a major simplification.
        # action = robot_arm_index_to_assign_task (0 to num_arms-1)
        # action = task_id_to_assign (0 to num_tasks-1)
        # This would be spaces.MultiDiscrete([num_arms, num_tasks])
        
        # Based on README: (动子的起点机械臂 ID, 动子的终点机械臂 ID, 动子在目标工作站执行的任务 ID)
        # Let's use num_arms for the number of physical arms. R0 is an additional entity.
        # Start ID: R0 or R1..R_num_arms. So num_arms + 1 choices. (0 for R0, 1..num_arms for actual arms)
        # End ID: R1..R_num_arms. So num_arms choices. (1..num_arms for actual arms)
        # Task ID: 0..10 (11 choices)
        if num_arms == 0: # Handle case during init before arms are fully setup
            return spaces.MultiDiscrete([1, 1, 1]) 
            
        return spaces.MultiDiscrete([
            num_arms + 1, # Start arm ID (0 for R0, 1..N for R1..RN)
            num_arms + 1, # End arm ID (0 for R0 '离场', 1..N for R1..RN) - Assuming R0 can be a destination for "离场"
            num_tasks + 1   # Task ID (0 for wait, 1..10 for tasks)
        ])


    def init_env(self):
        """Initializes or re-initializes the environment components."""
        print("Initializing environment...")
        self.robot_arm_list = []
        arm_configs = self.env_config.get("robot_arm_configs", [])
        if not arm_configs and self.env_config.get("num_robot_arms", 0) > 0:
            # Create default configs if only num_robot_arms is specified
            print(f"Warning: 'robot_arm_configs' not provided. Creating default arms for num_robot_arms={self.env_config.get('num_robot_arms')}")
            for i in range(self.env_config.get("num_robot_arms")):
                 # Simplified default config, may need adjustment
                default_task_info = {"task1": {"time":10, "switch_time":2, "output_type":"A"}}
                self.robot_arm_list.append(RobotArm(id=f"R{i+1}", task_list=["task1"], target_types={}, target_list=[], location=i*5, task_info_mapping=default_task_info))
        else:
            for config in arm_configs:
                self.robot_arm_list.append(RobotArm(**config))
        
        self.step_count = 0
        self.product_count = 0
        self.total_rewards = 0.0
        print(f"Environment initialized with {len(self.robot_arm_list)} robot arms.")

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed) # Important for seeding in Gym
        print("Resetting environment...")
        self.init_env() # Re-initialize arms, counters

        # Reset all robot arms
        for arm in self.robot_arm_list:
            arm.reset()

        self.step_count = 0
        self.total_rewards = 0.0
        self.product_count = 0 # Reset products completed

        observation = self._get_observation()
        info = self._get_info()
        
        # Ensure observation matches the defined space
        if not self.observation_space.contains(observation):
            print(f"Warning: Initial observation {observation} is not contained in observation space {self.observation_space}")
            # Fallback to a zero observation if it doesn't fit, though this indicates a deeper issue.
            observation = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)


        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one time step in the environment.
        """
        self.step_count += 1
        print(f"\n--- Step {self.step_count} ---")
        print(f"Received action: {action}")

        # 1. Parse Action
        # Action: [start_arm_idx_plus_r0, end_arm_idx_plus_r0, task_id]
        # Indices for start/end: 0 means R0, 1 means R1 (index 0 in list), ..., N means RN (index N-1 in list)
        parsed_action = self._parse_action(action)
        start_arm_id_str, end_arm_id_str, task_id_str = parsed_action
        
        # Placeholder for actual arm objects
        start_arm: Optional[RobotArm] = None
        end_arm: Optional[RobotArm] = None

        if start_arm_id_str != "R0":
            start_arm = next((arm for arm in self.robot_arm_list if arm.id == start_arm_id_str), None)
        
        if end_arm_id_str != "R0": # R0 cannot be a destination for tasks in this simplified logic
            end_arm = next((arm for arm in self.robot_arm_list if arm.id == end_arm_id_str), None)

        print(f"Parsed action: Start={start_arm_id_str}, End={end_arm_id_str}, Task={task_id_str}")

        # 2. Execute Action & Calculate Reward Components
        # This is highly simplified. A real implementation needs complex logic for mover, task assignment, etc.
        step_successful = False
        is_final_step_for_product = False # Needs logic to track product progress
        invalid_arm = False
        invalid_task = False
        distance_moved = 0
        task_switched = False
        task_execution_time_taken = 0

        if end_arm is not None and task_id_str != "task_0": # Assuming task_0 is 'wait'
            if task_id_str in end_arm.task_list:
                # Simulate task execution attempt
                if end_arm.task != task_id_str:
                    end_arm.switch_task(task_id_str) # This method in RobotArm needs to set task_switch_time
                    task_switched = True
                    # task_execution_time_taken += end_arm.task_switch_time # Cost of switching

                if end_arm.execute_task_check(): # Check if arm can start
                    end_arm.execute_task(task_id_str) # This sets arm.state = 1 (working), arm.task_time
                    # task_execution_time_taken += end_arm.task_time # This is total time, not per step.
                                                                  # For reward, cost is per time unit of *this step*.
                                                                  # Let's assume 1 unit of time passes per env.step()
                    step_successful = True # Basic success if task starts
                    # This is where we'd check if this task completion leads to product completion (is_final_step_for_product)
                    # For now, let's assume a task always takes 1 step to "complete" for reward purposes,
                    # but the arm itself tracks multiple steps internally via its update() method.
                else:
                    print(f"Task {task_id_str} could not be started on {end_arm.id} (pre-check failed).")
                    step_successful = False # e.g. arm busy, in fault
            else:
                print(f"Invalid task {task_id_str} for robot arm {end_arm.id}.")
                invalid_task = True
        elif end_arm_id_str == "R0" and start_arm_id_str != "R0": # "动子离场"
            print(f"Mover from {start_arm_id_str} is leaving (to R0).")
            step_successful = True # Action of leaving is successful
        elif task_id_str == "task_0" and end_arm is not None:
             print(f"Arm {end_arm.id} is assigned 'wait' task.")
             step_successful = True # Waiting is a valid action
        else:
            print("Invalid action: No valid end arm or task specified, or R0 to R0.")
            invalid_arm = True # Or a combination

        # Calculate distance for mover (example)
        if start_arm and end_arm:
            distance_moved = abs(end_arm.location - start_arm.location)
        elif start_arm_id_str == "R0" and end_arm: # Called from R0
            # Assume R0 is at a virtual location, e.g., -1, or look up table from README
            # For simplicity, let's use end_arm.location as distance from a virtual R0 at 0
            # README: R0->R1 cost 1, R0->R8 cost 8. This needs a lookup table.
            # Using a simple placeholder:
            distance_moved = end_arm.location 
        elif end_arm_id_str == "R0" and start_arm: # Moving to R0 (离场)
            distance_moved = start_arm.location # Distance to a virtual R0 at 0

        # 3. Update Robot Arms state (progress current tasks, handle failures)
        self._update_robot_arms()
        
        # Check if any arm completed a task that results in a product
        for arm in self.robot_arm_list:
            if arm.work_output == 1 and arm.state == 0 and arm.task is None: # Just finished a task successfully
                # This is a simplification. Need to check if *this specific action* led to completion.
                # is_final_step_for_product = True # Assume any OK output is a final step for now
                # self.product_count +=1
                # arm.work_output = -1 # Reset after processing
                pass # More sophisticated logic needed here

        # 4. Calculate Reward
        # For reward, task_execution_time should be time spent *in this step*.
        # If a task takes multiple steps, it's 1 per step until completion.
        current_step_execution_time = 1 if (end_arm and end_arm.state == 1) else 0

        reward = self.reward_calculator.calculate_reward(
            is_final_step=is_final_step_for_product, # This needs proper tracking
            step_successful=step_successful,
            invalid_arm_assignment=invalid_arm,
            invalid_task_assignment=invalid_task,
            distance_moved=distance_moved,
            task_switched=task_switched,
            task_execution_time=current_step_execution_time # Time spent in this step
        )
        self.total_rewards += reward

        # 5. Get New Observation
        observation = self._get_observation()

        # 6. Check for Termination Conditions
        done = False # Episode ends if a terminal state is reached
        truncated = False # Episode ends if max_steps_per_episode is reached
        
        if self.step_count >= self.max_steps_per_episode:
            print("Episode truncated due to max steps.")
            truncated = True
        
        # Example 'done' condition: a certain number of products completed
        # if self.product_count >= 10:
        #     done = True
        #     print("Episode finished: Product goal reached.")

        info = self._get_info()
        
        if not self.observation_space.contains(observation):
            print(f"Warning: Step observation {observation} is not contained in observation space {self.observation_space}")
            observation = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)


        print(f"Step outcome: Obs={observation.shape}, Reward={reward}, Done={done}, Truncated={truncated}")
        return observation, reward, done, truncated, info

    def _parse_action(self, action: np.ndarray) -> Tuple[str, str, str]:
        """
        Parses the raw action from the agent into a more usable format.
        Action from MultiDiscrete: [start_idx, end_idx, task_id_numeric]
        start_idx/end_idx: 0 for R0, 1 for R1 (list index 0), ..., N for RN (list index N-1)
        task_id_numeric: 0 for "wait", 1 for "task1", ..., 10 for "task10"
        Returns (start_arm_id_str, end_arm_id_str, task_id_str)
        """
        if not isinstance(action, np.ndarray) or action.ndim == 0: # Basic check
             # Default or error action
            print(f"Warning: Received invalid action format: {action}. Defaulting.")
            return "R0", self.robot_arm_list[0].id if self.robot_arm_list else "R1", "task_0"


        start_idx = action[0]
        end_idx = action[1]
        task_numeric = action[2]

        start_arm_id_str = f"R{start_idx}" if start_idx > 0 and start_idx <= len(self.robot_arm_list) else "R0"
        if start_idx > 0 and start_idx <= len(self.robot_arm_list) :
            start_arm_id_str = self.robot_arm_list[start_idx-1].id
        elif start_idx == 0:
             start_arm_id_str = "R0"
        else: # Invalid index
            print(f"Warning: Invalid start_idx {start_idx} in action. Defaulting to R0.")
            start_arm_id_str = "R0"


        end_arm_id_str = f"R{end_idx}" if end_idx > 0 and end_idx <= len(self.robot_arm_list) else "R0"
        if end_idx > 0 and end_idx <= len(self.robot_arm_list) :
            end_arm_id_str = self.robot_arm_list[end_idx-1].id
        elif end_idx == 0: # '动子离场'
             end_arm_id_str = "R0"
        else: # Invalid index
            print(f"Warning: Invalid end_idx {end_idx} in action. Defaulting to R0 (离场).")
            end_arm_id_str = "R0"
            
        # Mapping numeric task_id to string IDs like "task_1", "task_5" etc.
        # task_id=0 is wait, task_id=1 is task1, ..., task_id=10 is task10
        task_id_str = f"task_{task_numeric}" # e.g. task_0, task_1, ... task_10

        return start_arm_id_str, end_arm_id_str, task_id_str


    def _get_observation(self) -> np.ndarray:
        """
        Collects and returns the current environment observation.
        Placeholder: Concatenated states of all robot arms.
        This needs to be carefully designed to match define_observation_space and agent's input.
        """
        obs_list = []
        # Example: [state (int), current_task (mapped int), time_remaining (int), is_occupied (bool), work_output (int)]
        for arm in self.robot_arm_list:
            state_info = arm.get_state()
            # Map task string to int (e.g. "task1" -> 1, None -> 0)
            current_task_numeric = 0
            if state_info["current_task"]:
                try:
                    current_task_numeric = int(state_info["current_task"].split('_')[-1]) if state_info["current_task"] else 0
                except ValueError:
                    current_task_numeric = 0 # Default if parsing fails

            obs_list.extend([
                state_info["state"],
                current_task_numeric,
                state_info["task_time_remaining"],
                1 if state_info["is_occupied"] else 0,
                state_info["work_output"]
            ])
        
        # Ensure the observation list matches the Box space dimensions if it's fixed.
        # Pad with zeros if fewer arms than max_arms used in Box space definition.
        num_features_per_arm = 5 
        max_arms_in_space = int(self.observation_space.shape[0] / num_features_per_arm)
        
        current_num_arms = len(self.robot_arm_list)
        if current_num_arms < max_arms_in_space:
            padding = [0.0] * ( (max_arms_in_space - current_num_arms) * num_features_per_arm )
            obs_list.extend(padding)
        elif current_num_arms > max_arms_in_space: # Should not happen if configured correctly
            obs_list = obs_list[:self.observation_space.shape[0]]


        return np.array(obs_list, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Returns auxiliary information, not used for learning."""
        return {
            "step_count": self.step_count,
            "total_rewards": self.total_rewards,
            "product_count": self.product_count,
            "robot_arm_states": [arm.get_state() for arm in self.robot_arm_list]
        }

    def _update_robot_arms(self):
        """Updates the state of all robot arms in the environment."""
        print("Updating all robot arms...")
        for arm in self.robot_arm_list:
            arm.update()
            # Check for task completion that might increment product_count
            if arm.work_output == 1 and arm.state == 0 and arm.task is None: # Task just finished OK
                # Simple: assume any OK output is a "product" or significant sub-assembly
                # A more complex system would track specific product types and completion stages
                # self.product_count += 1
                # print(f"Arm {arm.id} completed a task successfully. Product count: {self.product_count}")
                # arm.work_output = -1 # Reset after acknowledging
                pass


    def add_robot_arm(self, config: Dict[str, Any]):
        """Adds a new robot arm to the environment."""
        print(f"Adding robot arm with config: {config}")
        new_arm = RobotArm(**config)
        self.robot_arm_list.append(new_arm)
        # Important: Observation and action spaces might need to be redefined or be dynamic (e.g. using Dict spaces)
        # For now, assume a fixed number of arms defined at init for simplicity of spaces.
        # If spaces are Box, this will likely break them.
        print(f"Warning: Adding arms dynamically may invalidate fixed-size action/observation spaces.")
        # self.action_space = self.define_action_space() # Potentially re-define
        # self.observation_space = self.define_observation_space() # Potentially re-define

    def remove_robot_arm(self, arm_id: str):
        """Removes a robot arm from the environment by its ID."""
        print(f"Removing robot arm with ID: {arm_id}")
        original_len = len(self.robot_arm_list)
        self.robot_arm_list = [arm for arm in self.robot_arm_list if arm.id != arm_id]
        if len(self.robot_arm_list) < original_len:
            print(f"Robot arm {arm_id} removed.")
        else:
            print(f"Robot arm {arm_id} not found.")
        # Similar to add_robot_arm, spaces might need redefinition.

    def init_robot_arm(self, arm_id: str, config: Optional[Dict[str, Any]] = None):
        """Re-initializes a specific robot arm or adds if not present."""
        # This method seems redundant if RobotArm.reset() and add_robot_arm() exist.
        # It might be intended to re-configure an existing arm.
        existing_arm = next((arm for arm in self.robot_arm_list if arm.id == arm_id), None)
        if existing_arm:
            print(f"Re-initializing robot arm {arm_id}...")
            if config:
                 # Naive re-init: replace attributes. A proper re-init might need more care.
                for key, value in config.items():
                    if hasattr(existing_arm, key):
                        setattr(existing_arm, key, value)
                    else:
                        print(f"Warning: Attribute {key} not found on RobotArm {arm_id} during re-init.")
            existing_arm.reset()
        elif config:
            print(f"Robot arm {arm_id} not found. Adding it with new config.")
            self.add_robot_arm(config)
        else:
            print(f"Cannot initialize arm {arm_id}: not found and no config provided.")


    def log_step(self):
        """Logs information about the current step. (Placeholder)"""
        # This could involve writing to a file, a database, or a more structured logging system.
        print(f"Logging step {self.step_count}. Total reward: {self.total_rewards}. Products: {self.product_count}")
        # For detailed logging, one might log the full info dict:
        # print(self._get_info())


    def render(self):
        """
        Renders the environment.
        For 'human' mode, this could be a graphical visualization.
        For 'ansi' mode, print to console.
        """
        if self.render_mode == 'ansi':
            print(f"\n--- Environment State (Step: {self.step_count}) ---")
            print(f"Total Reward: {self.total_rewards}, Products: {self.product_count}")
            for arm in self.robot_arm_list:
                state = arm.get_state()
                print(
                    f"  Arm ID: {state['id']}, Loc: {state['location']}, "
                    f"State: {state['state']} ({['Idle', 'Working', 'Fault', 'Offline'][state['state']]}), "
                    f"Task: {state['current_task']}, Time Left: {state['task_time_remaining']}, "
                    f"Occupied: {state['is_occupied']}, Output: {state['work_output']}"
                )
            return self._get_info() # Or a string representation
        elif self.render_mode == 'human':
            # Placeholder for future GUI rendering
            print("Human rendering mode not yet implemented. Use 'ansi'.")
            return None


    def close(self):
        """Performs any necessary cleanup."""
        print("Closing IndustrialEnv.")
        self.robot_arm_list.clear()

    def get_action_mask(self) -> Dict[str, np.ndarray]:
        """
        Generates an action mask indicating valid actions based on the current environment state.
        
        The structure of the mask corresponds to the MultiDiscrete action space:
        (start_arm_id, end_arm_id, task_id_for_end_arm).

        Returns:
            A dictionary with keys 'start_arm_mask', 'end_arm_mask', 'task_id_mask'.
            Each value is a NumPy array of 0s and 1s, where 1 indicates a valid action component.
            
        Note:
            This is currently a placeholder. In a real implementation, this method would
            contain sophisticated logic to determine:
            - Valid start arms (e.g., R0 or arms with available movers/parts).
            - Valid end arms (e.g., arms that are not full, can accept the part, or R0 for '离场').
            - Valid tasks for the chosen end arm (e.g., tasks the arm can perform,
              tasks for which prerequisites are met, tasks not leading to deadlock).
            The masks would be dynamically generated based on:
            - State of each RobotArm (self.robot_arm_list[i].get_state()).
            - Current task assignments and progress.
            - Mover availability and locations (if movers are explicitly modeled).
            - Production goals and material flow constraints.
        """
        if not isinstance(self.action_space, spaces.MultiDiscrete):
            print("Warning: Action space is not MultiDiscrete. Cannot generate a structured mask.")
            # Fallback to a non-structured mask or raise error
            return {"action_mask": np.ones(self.action_space.n) if hasattr(self.action_space, 'n') else np.array([])}

        action_space_dims = self.action_space.nvec

        # Placeholder: Allow all actions
        start_arm_mask = np.ones(action_space_dims[0], dtype=np.int8)
        end_arm_mask = np.ones(action_space_dims[1], dtype=np.int8)
        task_id_mask = np.ones(action_space_dims[2], dtype=np.int8)
        
        # Example of how one might restrict a specific action component:
        # If, for example, arm R1 (index 1 for start_arm_id if R0 is 0) cannot start a task:
        # if len(start_arm_mask) > 1: start_arm_mask[1] = 0 
        
        # If task_id 5 is currently invalid for all arms (hypothetically):
        # if len(task_id_mask) > 5: task_id_mask[5] = 0
        
        print(f"Generated placeholder action mask: Start {start_arm_mask.shape}, End {end_arm_mask.shape}, Task {task_id_mask.shape}")

        return {
            "start_arm_mask": start_arm_mask,
            "end_arm_mask": end_arm_mask,
            "task_id_mask": task_id_mask
        }


if __name__ == '__main__':
    print("Starting IndustrialEnv example usage...")

    # Example configuration
    env_config_example = {
        "num_robot_arms": 2,
        "robot_arm_configs": [
            {"id": "R1", "task_list": ["task_1", "task_2"], "target_types": {}, "target_list": [], "location": 0, 
             "task_info_mapping": {"task_1": {"time":3, "switch_time":1, "output_type":"A"}, "task_2": {"time":2, "switch_time":1, "output_type":"B"}}},
            {"id": "R2", "task_list": ["task_1", "task_3"], "target_types": {}, "target_list": [], "location": 5, 
             "task_info_mapping": {"task_1": {"time":3, "switch_time":1, "output_type":"A"}, "task_3": {"time":4, "switch_time":1, "output_type":"C"}}},
        ],
        "max_steps_per_episode": 50 # Short episode for testing
    }

    env = IndustrialEnv(env_config=env_config_example)
    obs, info = env.reset()

    print(f"Initial Observation: {obs}")
    print(f"Initial Info: {info}")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    # Check if observation is in space after reset
    if not env.observation_space.contains(obs):
         print(f"ERROR: Observation {obs} not in observation space {env.observation_space.shape} after reset.")

    # Test get_action_mask
    action_mask_example = env.get_action_mask()
    print(f"Example action mask from env: {action_mask_example}")
    assert len(action_mask_example["start_arm_mask"]) == env.action_space.nvec[0]
    assert len(action_mask_example["end_arm_mask"]) == env.action_space.nvec[1]
    assert len(action_mask_example["task_id_mask"]) == env.action_space.nvec[2]


    terminated = False
    truncated = False
    current_episode_steps = 0

    for i in range(env.max_steps_per_episode + 5) : # Test truncation and a few steps beyond
        if terminated or truncated:
            print(f"\nEpisode finished after {current_episode_steps} steps. Resetting.")
            obs, info = env.reset()
            terminated = False
            truncated = False
            current_episode_steps = 0
            if not env.observation_space.contains(obs):
                print(f"ERROR: Observation {obs} not in observation space {env.observation_space.shape} after reset on loop.")


        # Sample a random valid action
        # Action: [start_arm_idx_plus_r0, end_arm_idx_plus_r0, task_id]
        # start_idx/end_idx: 0 for R0, 1 for R1, ..., N for RN
        # task_id: 0 for "wait", 1 for "task_1", ..., 10 for "task_10"
        
        # Sample action considering the mask (conceptually)
        # A real agent would use the mask with its policy. Here, we just sample.
        # For a simple random sampler that respects the mask:
        current_mask = env.get_action_mask()
        
        # This is a simplified way to sample respecting mask for MultiDiscrete
        # A more robust way would be to resample if an invalid action is chosen,
        # or to choose only from valid sub-actions.
        sampled_action_parts = []
        for i, dim_name in enumerate(["start_arm_mask", "end_arm_mask", "task_id_mask"]):
            component_mask = current_mask[dim_name]
            valid_indices = np.where(component_mask == 1)[0]
            if len(valid_indices) > 0:
                sampled_action_parts.append(np.random.choice(valid_indices))
            else: # Should not happen with placeholder mask; if it does, fallback
                sampled_action_parts.append(env.action_space.nvec[i] // 2) # Fallback to a midpoint
        action = np.array(sampled_action_parts)

        # Example of a specific action: R0 -> R1, task_1
        # action = np.array([0, 1, 1]) # R0 to R1 (arm at index 0), task_1
        # Example: R1 -> R2, task_3 (on R2)
        # action = np.array([1, 2, 3]) # R1 to R2, task_3 (assuming task_3 is valid for R2)
        
        print(f"\n--- Main Loop: Taking Action {action} ---")
        obs, reward, terminated, truncated, info = env.step(action)
        current_episode_steps +=1

        env.render() # Render in ansi mode
        env.log_step()
        
        # Check observation validity after step
        if not env.observation_space.contains(obs):
            print(f"ERROR: Observation {obs} not in observation space {env.observation_space.shape} after step {current_episode_steps}.")
            # break # Stop if space is violated

        if i > env.max_steps_per_episode + 2 and not (terminated or truncated):
            print("ERROR: Episode did not truncate as expected.")
            # break

    print("\n--- Testing add/remove arm (illustrative) ---")
    new_arm_config = {"id": "R3", "task_list": ["task_1"], "target_types": {}, "target_list": [], "location": 10, 
                      "task_info_mapping": {"task_1": {"time":5, "switch_time":1, "output_type":"D"}}}
    env.add_robot_arm(new_arm_config)
    print(f"Number of arms after add: {len(env.robot_arm_list)}")
    env.remove_robot_arm("R1")
    print(f"Number of arms after remove: {len(env.robot_arm_list)}")
    env.init_robot_arm("R2", {"failure_rate": 0.5}) # Modify R2's failure rate
    r2_details = next(arm for arm in env.robot_arm_list if arm.id == "R2")
    print(f"R2 failure rate after init_robot_arm: {r2_details.failure_rate}")


    env.close()
    print("\nIndustrialEnv example usage finished.")
