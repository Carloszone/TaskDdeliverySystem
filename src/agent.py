import numpy as np
import random
import collections
import os # For saving/loading checkpoints
from typing import Optional # Added for type hinting

# --- Placeholder for PyTorch components ---
# If PyTorch were available, we would import torch, torch.nn, torch.optim:
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

class PlaceholderNet:
    """
    Placeholder for a PyTorch nn.Module.
    Simulates a basic network for action or value estimation.
    """
    def __init__(self, input_dim, output_dim, name="Net"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        # Simulate some weights for saving/loading
        self.weights = {f"w{i}": np.random.rand() for i in range(3)} 
        self.biases = {f"b{i}": np.random.rand() for i in range(3)}
        print(f"Initialized PlaceholderNet: {self.name} (Input: {input_dim}, Output: {output_dim})")

    def forward(self, x):
        print(f"{self.name} forward pass with input shape: {x.shape if isinstance(x, np.ndarray) else 'scalar'}")
        if self.output_dim > 0:
            # For multi-component actions, output_dim is sum of component dimensions
            return np.random.rand(self.output_dim) 
        return np.random.rand() # For a critic outputting a single value

    def parameters(self): # Simulate model.parameters() for optimizer
        # Return a list of all weight/bias values for the optimizer to conceptually use
        all_params = list(self.weights.values()) + list(self.biases.values())
        return all_params

    def state_dict(self): # Simulate model.state_dict()
        # Combine weights and biases into a single dictionary for saving
        sd = {}
        for k, v in self.weights.items(): sd[f"{self.name}_{k}"] = v
        for k, v in self.biases.items(): sd[f"{self.name}_{k}"] = v
        return sd

    def load_state_dict(self, state_dict): # Simulate model.load_state_dict()
        self.weights = {}
        self.biases = {}
        for k, v in state_dict.items():
            if f"{self.name}_w" in k :
                self.weights[k.replace(f"{self.name}_", "")] = v
            elif f"{self.name}_b" in k:
                 self.biases[k.replace(f"{self.name}_", "")] = v
        print(f"{self.name} weights and biases loaded from state_dict.")


# Placeholder for PyTorch Optimizer
class PlaceholderOptimizer:
    def __init__(self, params, lr):
        self.params = list(params) # Ensure it's a list of values
        self.lr = lr
        print(f"Initialized PlaceholderOptimizer with learning_rate: {lr} for {len(self.params)} params groups/tensors.")

    def zero_grad(self):
        pass

    def step(self):
        pass

# --- End of Placeholder PyTorch components ---


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)
        print(f"ReplayBuffer initialized with size: {buffer_size}")

    def store_transition(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return []
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self,
                 state_dim: int,
                 action_dims: list, 
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 tau: float = 0.005,
                 learning_rate_actor: float = 1e-4,
                 learning_rate_critic: float = 1e-3,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 min_learn_size: int = 1000,
                 device: str = "cpu"):
        
        self.state_dim = state_dim
        self.action_dims = action_dims 
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min
        self.tau: float = tau
        self.batch_size: int = batch_size
        self.min_learn_size: int = min_learn_size
        self.device = device 

        print(f"Initializing Agent on device: {self.device}")
        print(f"State dim: {state_dim}, Action dims: {action_dims}")

        # Using PlaceholderNet
        # action_net output_dim is sum of individual action component dimensions for MultiDiscrete-like setup
        actor_output_total_dim = sum(self.action_dims)
        self.action_net = PlaceholderNet(state_dim, actor_output_total_dim, "ActionNet")
        self.target_action_net = PlaceholderNet(state_dim, actor_output_total_dim, "TargetActionNet")
        self.target_action_net.load_state_dict(self.action_net.state_dict())

        # Critic network takes state and action (concatenated)
        # For placeholder, sum of action_dims total elements for action part
        critic_input_dim = state_dim + len(action_dims) # Assuming action is represented as a list/array of N components
        self.critic_net = PlaceholderNet(critic_input_dim, 1, "CriticNet")
        self.target_critic_net = PlaceholderNet(critic_input_dim, 1, "TargetCriticNet")
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())

        self.actor_optimizer = PlaceholderOptimizer(self.action_net.parameters(), learning_rate_actor)
        self.critic_optimizer = PlaceholderOptimizer(self.critic_net.parameters(), learning_rate_critic)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.learn_step_counter = 0

    def choose_action(self, observation: np.ndarray, action_mask: Optional[dict] = None) -> np.ndarray:
        """
        Chooses an action based on the current observation.
        Args:
            observation: The current state of the environment.
            action_mask: An optional dictionary of masks for each action component (e.g.,
                         {'start_arm_mask': np.array, 'end_arm_mask': np.array, 'task_id_mask': np.array}).
                         A 1 indicates valid, 0 indicates invalid.
        Returns:
            An action array.
        """
        if random.random() < self.epsilon:
            # Exploration: Choose a random action, try to respect mask if available
            action_parts = []
            action_component_mask_keys = ["start_arm_mask", "end_arm_mask", "task_id_mask"]
            for i, dim_size in enumerate(self.action_dims):
                component_mask_key = action_component_mask_keys[i]
                if action_mask and component_mask_key in action_mask and action_mask[component_mask_key] is not None:
                    valid_indices = np.where(action_mask[component_mask_key] == 1)[0]
                    if len(valid_indices) > 0:
                        action_parts.append(random.choice(valid_indices))
                    else: # No valid action in mask, fallback to random unmasked
                        action_parts.append(random.randint(0, dim_size - 1))
                else: # No mask for this component, choose freely
                    action_parts.append(random.randint(0, dim_size - 1))
            action = action_parts
            print(f"Agent: Choosing random action (exploration, mask-aware): {action}")

        else:
            # Exploitation: action_net outputs a flat array of scores
            raw_scores = self.action_net.forward(observation) # Shape: (sum(action_dims),)
            
            action_parts = []
            current_idx = 0
            action_component_mask_keys = ["start_arm_mask", "end_arm_mask", "task_id_mask"] # Order matters

            for i, dim_size in enumerate(self.action_dims):
                component_scores = np.array(raw_scores[current_idx : current_idx + dim_size], dtype=np.float32)
                
                component_mask_key = action_component_mask_keys[i]
                if action_mask and component_mask_key in action_mask:
                    mask_for_component = action_mask[component_mask_key]
                    if mask_for_component is not None and len(mask_for_component) == len(component_scores):
                        component_scores[mask_for_component == 0] = -1e9 # Apply mask by setting score of invalid actions to very low
                        # print(f"Agent: Applied mask to component {i}. Masked scores (excerpt): {component_scores[:5]}")
                    elif mask_for_component is not None:
                        print(f"Agent: Warning - Mask length mismatch for component {i} ('{component_mask_key}'). Mask len: {len(mask_for_component)}, Scores len: {len(component_scores)}. Mask not applied to this component.")

                chosen_sub_action = np.argmax(component_scores).item() if component_scores.size > 0 else 0
                action_parts.append(chosen_sub_action)
                current_idx += dim_size
            action = action_parts
            print(f"Agent: Choosing action via policy (exploitation): {action} from (masked) scores.")
        return np.array(action, dtype=np.int32)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.store_transition(state, action, reward, next_state, done)

    def _calculate_advantages(self, rewards: np.ndarray, dones: np.ndarray, values: np.ndarray, next_values: np.ndarray) -> np.ndarray:
        """
        Calculates advantages, e.g., for Actor-Critic methods.
        Currently implements one-step TD error (which is equivalent to GAE with lambda=0).

        Args:
            rewards: Array of rewards from the batch. Shape (batch_size,).
            dones: Array of done flags from the batch. Shape (batch_size,).
            values: Array of state values for current states, V(s_t), from critic. Shape (batch_size,).
            next_values: Array of state values for next states, V(s_{t+1}), from critic. Shape (batch_size,).
        Returns:
            Array of calculated advantages. Shape (batch_size,).
        Note:
            This is a placeholder. For more advanced algorithms like PPO or A2C,
            Generalized Advantage Estimation (GAE) would be implemented here.
            The GAE formula is:
            Â_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...
            where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD error,
            γ is the discount factor, and λ is the GAE smoothing parameter (typically between 0.9 and 1.0).
            The current implementation is equivalent to GAE(lambda=0): Â_t = δ_t.
        """
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)
        values = np.asarray(values)
        next_values = np.asarray(next_values)
        td_errors = rewards + self.gamma * next_values * (1 - dones) - values
        # print("Agent: Calculating advantages (using one-step TD error as placeholder for GAE).")
        return td_errors

    def learn(self):
        if len(self.replay_buffer) < self.min_learn_size:
            return
        self.learn_step_counter += 1
        if self.learn_step_counter % 100 == 0:
            print(f"Agent: Learning step {self.learn_step_counter}")

        batch = self.replay_buffer.sample(self.batch_size)
        if not batch: return

        states, actions, rewards, next_states, dones = zip(*batch)
        states_np = np.array(states)
        actions_np = np.array(actions)
        rewards_np = np.array(rewards).reshape(-1, 1)
        next_states_np = np.array(next_states)
        dones_np = np.array(dones).reshape(-1, 1)

        # --- Placeholder for learning logic ---
        
        # Conceptual: Get values from critic for GAE.
        # For PlaceholderNet, critic_net.forward(state) would return a single random value.
        # We need a value for each state and next_state in the batch.
        values_np = np.array([self.critic_net.forward(states_np[i]) for i in range(self.batch_size)])
        next_values_np = np.array([self.target_critic_net.forward(next_states_np[i]) for i in range(self.batch_size)])
        
        advantages = self._calculate_advantages(
            rewards_np.flatten(), 
            dones_np.flatten(), 
            values_np.flatten(), 
            next_values_np.flatten()
        )
        if self.learn_step_counter % 100 == 0:
            print(f"Agent: Calculated advantages (first 5): {advantages[:5]}")

        # Placeholder for actual network updates using these advantages or TD errors
        # For actor loss (policy gradient style):
        #   log_probs = self.action_net.get_log_probs(states_np, actions_np) # Requires action_net to provide this
        #   actor_loss = -(log_probs * advantages).mean()
        #   self.actor_optimizer.zero_grad()
        #   actor_loss.backward()
        #   self.actor_optimizer.step()

        # For critic loss (MSE of values vs returns/TD-target):
        #   returns = advantages + values_np 
        #   critic_loss = ((self.critic_net.forward_batch(states_np) - returns)**2).mean()
        #   self.critic_optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor_optimizer.step()
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()

        self._update_target_networks()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min: self.epsilon = self.epsilon_min
        if self.learn_step_counter % 100 == 0:
            print(f"Agent: Epsilon decayed to {self.epsilon:.3f}")

    def _update_target_networks(self):
        for net_name in ['action_net', 'critic_net']:
            main_net = getattr(self, net_name)
            target_net = getattr(self, f'target_{net_name}')
            
            new_target_sd = {}
            main_sd = main_net.state_dict()
            target_sd = target_net.state_dict()
            for key in target_sd.keys(): # Iterate over target keys to ensure structure matches
                if key in main_sd:
                    main_w = main_sd[key]
                    target_w = target_sd[key]
                    new_target_sd[key] = self.tau * main_w + (1.0 - self.tau) * target_w
                else: # Should not happen if networks are same structure
                    new_target_sd[key] = target_sd[key] 
            target_net.load_state_dict(new_target_sd)

    def save_checkpoint(self, path=".", filename_prefix="agent_"):
        if not os.path.exists(path): os.makedirs(path)
        actor_path = os.path.join(path, f"{filename_prefix}action_net.npz")
        critic_path = os.path.join(path, f"{filename_prefix}critic_net.npz")
        np.savez(actor_path, **self.action_net.state_dict())
        np.savez(critic_path, **self.critic_net.state_dict())
        print(f"Agent: Checkpoint saved to {path} with prefix {filename_prefix}")

    def load_checkpoint(self, path=".", filename_prefix="agent_"):
        actor_path = os.path.join(path, f"{filename_prefix}action_net.npz")
        critic_path = os.path.join(path, f"{filename_prefix}critic_net.npz")
        try:
            actor_data = np.load(actor_path)
            self.action_net.load_state_dict({k: actor_data[k] for k in actor_data.files})
            self.target_action_net.load_state_dict(self.action_net.state_dict())
            critic_data = np.load(critic_path)
            self.critic_net.load_state_dict({k: critic_data[k] for k in critic_data.files})
            self.target_critic_net.load_state_dict(self.critic_net.state_dict())
            print(f"Agent: Checkpoint loaded from {path} with prefix {filename_prefix}")
        except FileNotFoundError:
            print(f"Agent: No checkpoint found at {path} with prefix {filename_prefix}")
        except Exception as e:
            print(f"Agent: Error loading checkpoint: {e}")

if __name__ == '__main__':
    print("--- Agent Example Usage ---")
    example_state_dim = 10 
    example_action_dims = [3, 3, 5] # e.g. R0-R2, R0-R2, Task0-Task4

    agent = Agent(state_dim=example_state_dim, action_dims=example_action_dims, 
                  buffer_size=1000, batch_size=32, min_learn_size=50)

    for i in range(200): 
        obs = np.random.rand(example_state_dim).astype(np.float32)
        
        # Dummy action mask for testing - dimensions must match example_action_dims [3,3,5]
        action_mask_example = {
            "start_arm_mask": np.random.randint(0, 2, size=example_action_dims[0], dtype=np.int8), 
            "end_arm_mask": np.random.randint(0, 2, size=example_action_dims[1], dtype=np.int8),   
            "task_id_mask": np.random.randint(0, 2, size=example_action_dims[2], dtype=np.int8) 
        }
        # Ensure at least one valid action per component for testing random choice in exploration
        if np.sum(action_mask_example["start_arm_mask"]) == 0: action_mask_example["start_arm_mask"][0] = 1
        if np.sum(action_mask_example["end_arm_mask"]) == 0: action_mask_example["end_arm_mask"][0] = 1
        if np.sum(action_mask_example["task_id_mask"]) == 0: action_mask_example["task_id_mask"][0] = 1
        
        # print(f"Test action mask: {action_mask_example}")
        # To test with no mask: action_mask_example = None
        
        action = agent.choose_action(obs, action_mask=action_mask_example)
        
        next_obs = np.random.rand(example_state_dim).astype(np.float32)
        reward = random.uniform(-1, 1)
        done = random.random() < 0.05 
        agent.store_transition(obs, action, reward, next_obs, done)
        
        if len(agent.replay_buffer) >= agent.min_learn_size:
            agent.learn()

        if (i + 1) % 50 == 0:
            print(f"Simulated step {i+1}. Buffer size: {len(agent.replay_buffer)}. Epsilon: {agent.epsilon:.3f}")
            if done: print("Episode finished.")

    agent.save_checkpoint(filename_prefix="test_agent_")
    new_agent = Agent(state_dim=example_state_dim, action_dims=example_action_dims)
    new_agent.load_checkpoint(filename_prefix="test_agent_")
    print("\nAgent example usage finished.")
