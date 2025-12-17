import gym
import numpy as np
import os
import torch
from typing import Tuple, List

from src.buffer import ReplayBuffer
from src.environment import Environment
from src.networks import Actor, Critic, Value, Distributional_Critic
  
   
class Agent():
    """Abstract Agent class to be inherited by the various SAC agents."""
    
    def __init__(self,
                 lr_Q: float, 
                 lr_pi: float, 
                 input_shape: Tuple, 
                 tau: float, 
                 env: gym.Env, 
                 checkpoint_directory_networks: str,
                 gamma: float = 0.99, 
                 size: int = 1000000,
                 layer_size: int = 256, 
                 batch_size: int = 256,
                 delay: int = 1,
                 grad_clip: float = 1.0,
                 device: str = 'cpu',
                 ) -> None:
        """Constructor method of the Agent class.
        
        Args:
            # ... (Existing args)
            gamma (float): discount factor for future rewards (base value)
        """
        
        self.input_shape = input_shape
        self.action_space_dimension = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.delay = delay
        self.grad_clip = grad_clip
        self.checkpoint_directory_networks = checkpoint_directory_networks
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.learn_step_counter = 0
        
        # GBI specific parameter
        self.max_steps_per_episode = env.max_steps_per_episode

        # Replay Buffer
        self.memory = ReplayBuffer(size=size, 
                                   input_shape=input_shape, 
                                   action_space_dimension=self.action_space_dimension)

        # Networks
        self.actor = Actor(input_shape=input_shape, 
                           n_actions=self.action_space_dimension, 
                           layer_neurons=layer_size, 
                           network_name='actor',
                           checkpoint_directory_networks=checkpoint_directory_networks,
                           device=device)

        # ... (rest of the __init__ remains the same)
        # Assuming Critic and Value networks are also initialized here...


    def learn(self, step: int) -> None:
        """Updates the networks' weights using sampled memory."""
        
        if self.memory.pointer < self.batch_size:
            return
        
        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)
        
        # Convert numpy arrays to torch tensors and move to device
        states_tensor = torch.tensor(states, dtype=torch.float).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).unsqueeze(1).to(self.device)
        new_states_tensor = torch.tensor(new_states, dtype=torch.float).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool).unsqueeze(1).to(self.device)
        
        # --- GBI: DYNAMIC DISCOUNT FACTOR (CRITICAL MODIFICATION) ---
        # The last feature of the new_states is the Time-to-Goal Ratio (t_rem / T), ranging from 1.0 (start) to 0.0 (end).
        # We want the agent to focus more on the goal (higher effective gamma) as time runs out (ratio approaches 0).
        time_to_goal_ratio = new_states_tensor[:, -1].unsqueeze(1) # Shape: (batch_size, 1)
        
        # Dynamic Gamma: Increase gamma slightly as the time-to-goal ratio decreases.
        # This increases the weight of the terminal reward relative to immediate rewards.
        # Factor 0.02 is a hyperparameter to tune the dynamic effect.
        dynamic_gamma = self.gamma + 0.02 * (1.0 - time_to_goal_ratio)
        dynamic_gamma = torch.clamp(dynamic_gamma, self.gamma, 1.0) # Ensure it doesn't drop below base gamma
        # -------------------------------------------------------------
        
        with torch.no_grad():
            
            # --- Q-target calculation using the dynamic gamma ---
            # ... calculate next action, log_prob, value ...
            
            # Use the dynamic_gamma instead of self.gamma in the target calculation
            # Q_target = reward + dynamic_gamma * value_target * (1 - dones_tensor)
            
            # ... (rest of the Q-target calculation logic from original SAC implementation, using dynamic_gamma)

        self.learn_step_counter += 1
        
        # ... (rest of the SAC learning logic remains the same)
        
        # Update target networks (remains the same)
        if self.learn_step_counter % self.delay == 0:
            self._update_target_networks()


class Distributional_Agent(Agent):
    """Distributional Soft Actor Critic Agent."""
    
    def __init__(self,
                 lr_Q: float,
                 lr_pi: float,
                 lr_alpha: float,
                 input_shape: Tuple,
                 tau: float,
                 env: Environment,
                 checkpoint_directory_networks: str,
                 gamma: float = 0.99,
                 size: int = 1000000,
                 layer_size: int = 256,
                 delay: int = 1,
                 grad_clip: float = 1.0,
                 batch_size: int = 256,
                 device: str = 'cpu',
                 ) -> None:
        """Constructor method for the Distributional_Agent class."""
        
        # Initialize base Agent attributes
        super(Distributional_Agent, self).__init__(lr_Q=lr_Q, 
                                                   lr_pi=lr_pi, 
                                                   input_shape=input_shape, 
                                                   tau=tau, 
                                                   env=env, 
                                                   checkpoint_directory_networks=checkpoint_directory_networks,
                                                   gamma=gamma,
                                                   size=size,
                                                   layer_size=layer_size,
                                                   batch_size=batch_size,
                                                   delay=delay,
                                                   grad_clip=grad_clip,
                                                   device=device)

        # Distributional Critic Networks (These are different from the base Agent)
        self.Q_1 = Distributional_Critic(input_shape=input_shape, 
                                         n_actions=self.action_space_dimension, 
                                         layer_neurons=layer_size, 
                                         network_name='Q_1',
                                         checkpoint_directory_networks=checkpoint_directory_networks,
                                         device=device)
        # ... (Initialize Q_2, Target Q_1, Target Q_2, etc.)

        # ... (rest of the __init__ remains the same)

    def learn(self, step: int) -> None:
        """Updates the networks' weights using sampled memory for the distributional agent."""
        
        if self.memory.pointer < self.batch_size:
            return
        
        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)
        
        # Convert numpy arrays to torch tensors and move to device
        states_tensor = torch.tensor(states, dtype=torch.float).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).unsqueeze(1).to(self.device)
        new_states_tensor = torch.tensor(new_states, dtype=torch.float).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool).unsqueeze(1).to(self.device)
        
        # --- GBI: DYNAMIC DISCOUNT FACTOR for Distributional Agent ---
        # The last feature of the new_states is the Time-to-Goal Ratio (t_rem / T).
        time_to_goal_ratio = new_states_tensor[:, -1].unsqueeze(1) # Shape: (batch_size, 1)
        
        # Dynamic Gamma: Increase gamma slightly as the time-to-goal ratio decreases (time runs out).
        # Factor 0.02 is a hyperparameter to tune the dynamic effect.
        dynamic_gamma = self.gamma + 0.02 * (1.0 - time_to_goal_ratio)
        dynamic_gamma = torch.clamp(dynamic_gamma, self.gamma, 1.0) # Ensure it doesn't drop below base gamma
        # -------------------------------------------------------------

        with torch.no_grad():
            
            # --- Q-target calculation using the dynamic gamma ---
            # ... calculate next action, log_prob, value ...
            
            # Use the dynamic_gamma instead of self.gamma in the distributional Q-target calculation
            # V_target = rewards + dynamic_gamma * V_next * (1 - dones_tensor)
            
            # ... (rest of the Distributional SAC Q-target calculation logic, using dynamic_gamma)

        self.learn_step_counter += 1
        
        # ... (rest of the SAC learning logic remains the same)
        
        # Update target networks (remains the same)
        if self.learn_step_counter % self.delay == 0:
            self._update_target_networks()

# --- Instanciation function (remains the same logic, but uses the updated classes) ---
def instanciate_agent(args, env: Environment, device: str, checkpoint_directory: str) -> Agent:
    """Instanciate the correct agent depending on the command line argument."""
    
    checkpoint_directory_networks = os.path.join(checkpoint_directory, 'networks')
    
    if args.agent_type == 'regular' or args.agent_type == 'manual_temperature':
        
        agent = Agent(lr_Q=args.lr_Q, 
                      lr_pi=args.lr_pi, 
                      input_shape=env.observation_space.shape, 
                      tau=args.tau, 
                      env=env, 
                      gamma=args.gamma if hasattr(args, 'gamma') else 0.99, # Use default if not explicitly defined
                      size=args.memory_size, 
                      batch_size=args.batch_size, 
                      layer_size=args.layer_size, 
                      grad_clip=args.grad_clip,
                      delay=args.delay,
                      checkpoint_directory_networks=checkpoint_directory_networks,
                      device=device)
        
    elif args.agent_type == 'distributional':
        
        agent = Distributional_Agent(lr_Q=args.lr_Q,
                                     lr_pi=args.lr_pi, 
                                     lr_alpha=args.lr_alpha,  
                                     input_shape=env.observation_space.shape, 
                                     tau=args.tau,
                                     env=env, 
                                     gamma=args.gamma if hasattr(args, 'gamma') else 0.99, # Use default if not explicitly defined
                                     size=args.memory_size,
                                     batch_size=args.batch_size, 
                                     layer_size=args.layer_size, 
                                     delay=args.delay,
                                     grad_clip=args.grad_clip,
                                     checkpoint_directory_networks=checkpoint_directory_networks,
                                     device=device)
        
    return agent
