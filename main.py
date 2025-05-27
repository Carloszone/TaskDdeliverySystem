import numpy as np
import sys
import os

# Adjust path for imports from src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from industrial_env import IndustrialEnv
from agent import Agent
from data_processor import DataProcessor
# RobotArm and RewardCalculator are used by IndustrialEnv, no direct import needed here.
# Model classes are used by Agent, no direct import needed here.

def main():
    print("--- Starting Industrial RL Agent Simulation ---")

    # 1. Initialize Environment, Agent, DataProcessor
    # Environment Configuration
    env_config = {
        "num_robot_arms": 3, # Example: 3 robot arms
        "robot_arm_configs": [
            {"id": "R1", "task_list": ["task_1", "task_2"], "target_types": {}, "target_list": [], "location": 0, 
             "task_info_mapping": {"task_1": {"time":5, "switch_time":1, "output_type":"P1"}, "task_2": {"time":3, "switch_time":1, "output_type":"P2"}}},
            {"id": "R2", "task_list": ["task_1", "task_3"], "target_types": {}, "target_list": [], "location": 5, 
             "task_info_mapping": {"task_1": {"time":5, "switch_time":1, "output_type":"P1"}, "task_3": {"time":6, "switch_time":1, "output_type":"P3"}}},
            {"id": "R3", "task_list": ["task_2", "task_3"], "target_types": {}, "target_list": [], "location": 10, 
             "task_info_mapping": {"task_2": {"time":3, "switch_time":1, "output_type":"P2"}, "task_3": {"time":6, "switch_time":1, "output_type":"P3"}}},
        ],
        "max_steps_per_episode": 100 # Max steps per episode
    }
    env = IndustrialEnv(env_config=env_config)

    # DataProcessor Configuration (example)
    # Assuming the observation from env is a flat vector that DataProcessor might scale or transform.
    # The current DataProcessor placeholder might not do much if not configured for specific features.
    # For this example, let's assume 5 features per arm from the env's _get_observation()
    num_features_per_arm_obs = 5 
    total_numerical_features_from_env = env_config["num_robot_arms"] * num_features_per_arm_obs
    data_processor = DataProcessor(num_categorical_features=0, num_numerical_features=total_numerical_features_from_env)
    
    # Fit scaler if using it (optional, requires sample data)
    # sample_obs_for_scaler, _ = env.reset() 
    # data_processor.fit_scaler(np.array([sample_obs_for_scaler] * 10)) # Fit with a few samples

    # Agent Configuration
    # The agent's state_dim should conceptually match the output of DataProcessor.
    # For ActorCriticModel, the 'state_dim' in Agent is more of a placeholder if model_params define the true structure.
    # Action_dims from env.action_space.nvec
    
    agent_model_params = {
        # These should be based on how DataProcessor structures C, M, W inputs for the model
        "num_categorical_features": 5,  # Example: Conceptual features per item in C
        "num_numerical_features": 10,   # Example: Conceptual features per item in M
        "num_work_process_features": 3, # Example: Conceptual features per item in W
        "internal_embedding_dim": 32,   
        "transformer_heads": 1,
        "transformer_ffn_dim": 64,
        "transformer_layers": 1,
        "seq_len_c_example": env_config["num_robot_arms"], # Conceptual: C features for each arm
        "seq_len_m_example": env_config["num_robot_arms"], # Conceptual: M features for each arm
        "seq_len_w_example": 5  # Conceptual: 5 nodes in a process graph
    }
    # The agent's state_dim is currently used by PlaceholderNet if ActorCriticModel isn't fully integrated.
    # For ActorCriticModel, the observation is expected as C,M,W dict by agent.choose_action.
    # For simplicity with current agent.choose_action, we pass the flat observation from env.
    agent_state_dim_placeholder = env.observation_space.shape[0] 

    agent = Agent(
        state_dim=agent_state_dim_placeholder, 
        action_dims=list(env.action_space.nvec), # [start_choices, end_choices, task_choices]
        model_params=agent_model_params,
        buffer_size=10000,      # Size of replay buffer
        batch_size=128,         # Batch size for learning
        min_learn_size=500      # Min experiences in buffer before learning starts
    )

    # 2. Simulation Loop
    num_episodes = 10 # Total number of episodes to run
    max_steps = env_config["max_steps_per_episode"] # Max steps per episode from env_config

    for episode in range(num_episodes):
        observation, info = env.reset()
        # The observation from env is currently a flat NumPy array.
        # DataProcessor.observation_processor expects this.
        # Agent.choose_action (with ActorCriticModel) expects a dict of C,M,W or creates dummy ones.
        # For this main loop, we'll pass the flat observation to DataProcessor,
        # then the (potentially) processed flat observation to the Agent. Agent's choose_action
        # will handle creating dummy C,M,W for its internal ActorCriticModel for now.
        
        # Optional: If DataProcessor is actually used for scaling
        # if data_processor.scaler and not data_processor.fitted:
        #     print("Warning: DataProcessor scaler not fitted. Fitting with initial observation.")
        #     data_processor.fit_scaler(np.array([observation] * 10)) # Fit with a few dummy samples

        current_episode_reward = 0.0

        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        for step in range(max_steps):
            # Process observation (optional, based on DataProcessor's role)
            # For current DataProcessor placeholder, it might just convert type or scale if fitted.
            processed_observation = data_processor.observation_processor(observation)
            
            # Get action mask from environment
            action_mask = env.get_action_mask()
            
            # Agent chooses action
            # Agent's choose_action expects flat observation if it internally creates dummy C,M,W.
            # If DataProcessor were to create the C,M,W dict, that would be passed instead.
            action = agent.choose_action(processed_observation, action_mask=action_mask)
            
            # Environment steps
            next_observation, reward, done, truncated, info = env.step(action)
            
            # Store experience
            # Store raw observation from env, or processed_observation if that's the defined state for the agent
            agent.store_transition(observation, action, reward, next_observation, done)
            
            # Agent learns
            agent.learn()
            
            observation = next_observation
            current_episode_reward += reward
            
            print(f"  Ep {episode + 1}, Step {step + 1}: Action={action}, Reward={reward:.2f}, Done={done}, Trunc={truncated}")
            env.render() # Optional: render environment state

            if done or truncated:
                print(f"Episode {episode + 1} finished after {step + 1} steps. Total Reward: {current_episode_reward:.2f}")
                break
        
        # Optional: Save checkpoint periodically
        if (episode + 1) % 5 == 0:
            agent.save_checkpoint(filename_prefix=f"agent_ep{episode+1}_")

    print("\n--- Simulation Finished ---")
    env.close()

if __name__ == '__main__':
    main()
