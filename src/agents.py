import gymnasium as gym
import numpy as np
import os
import torch
from typing import Tuple, List

from src.buffer import ReplayBuffer
from src.environment import Environment
from src.networks import Actor, Critic, Value, Distributional_Critic
# NOTE: Assumes Actor, Critic, Value, and Distributional_Critic are defined in src/networks.py


class Agent():
    """
    Implementation of the Soft Actor-Critic (SAC) agent.
    
    This agent uses two Q-networks and a single V-network (or target Q-network equivalent)
    to learn a stochastic policy while maximizing both expected return and entropy.
    It supports automatic temperature (alpha) tuning.
    """
    
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
                 auto_alpha: bool = True,
                 lr_alpha: float = 0.0003,
                 sac_temperature: float = 0.1,
                 ) -> None:
        
        self.gamma = gamma
        self.tau = tau
        self.layer_size = layer_size
        self.batch_size = batch_size
        self.delay = delay
        self.grad_clip = grad_clip
        self.n_actions = env.action_space.shape[0]
        self.learn_step_counter = 0
        self.auto_alpha = auto_alpha
        self.sac_temperature = sac_temperature
        self.device = torch.device(device)
        self.checkpoint_dir = checkpoint_directory_networks
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.actor = Actor(
            input_shape=input_shape,
            n_actions=self.n_actions,
            layer_neurons=layer_size,
            network_name='actor',
            checkpoint_directory_networks=checkpoint_directory_networks,
            device=device
        )
        
        self.critic_1 = Critic(
            input_shape=input_shape,
            n_actions=self.n_actions,
            layer_neurons=layer_size,
            network_name='critic_1',
            checkpoint_directory_networks=checkpoint_directory_networks,
            device=device
        )
        self.critic_2 = Critic(
            input_shape=input_shape,
            n_actions=self.n_actions,
            layer_neurons=layer_size,
            network_name='critic_2',
            checkpoint_directory_networks=checkpoint_directory_networks,
            device=device
        )

        self.value = Value(
            input_shape=input_shape,
            layer_neurons=layer_size,
            network_name='value',
            checkpoint_directory_networks=checkpoint_directory_networks,
            device=device
        )
        
        self.target_value = Value(
            input_shape=input_shape,
            layer_neurons=layer_size,
            network_name='target_value',
            checkpoint_directory_networks=checkpoint_directory_networks,
            device=device
        )
        
        # Initialize target_value parameters to be the same as value network
        self.target_value.load_state_dict(self.value.state_dict())
        
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr_Q)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr_Q)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr_Q)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_pi)
        
        if self.auto_alpha:
            self.log_alpha = torch.tensor([np.log(self.sac_temperature)], dtype=torch.float, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
            # Target Entropy: dimension of action space
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape)).item()
        else:
            self.alpha = torch.tensor(sac_temperature, dtype=torch.float, device=self.device)
            self.alpha_optimizer = None
            self.target_entropy = None

        self.memory = ReplayBuffer(size=size, input_shape=input_shape, action_space_dimension=self.n_actions)
        
    def store_transition(self, state, action, reward, new_state, done):
        """
        Stores a transition in the replay buffer.
        """
        self.memory.push(state, action, reward, new_state, done)
        
    def choose_action(self, observation, evaluate=False) -> np.ndarray:
        state = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False, deterministic=evaluate)
        return actions.cpu().detach().numpy()[0]
    
    def reparameterize_and_sample(self, state):
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True, deterministic=False)
        return actions, log_probs

    def learn(self):
        """
        Performs one step of SAC learning.
        """
        if self.memory.pointer < self.batch_size:
            return

        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)

        state_tensor = torch.tensor(states, dtype=torch.float).to(self.device)
        action_tensor = torch.tensor(actions, dtype=torch.float).to(self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float).to(self.device).view(-1, 1)
        new_state_tensor = torch.tensor(new_states, dtype=torch.float).to(self.device)
        done_tensor = torch.tensor(dones, dtype=torch.float).to(self.device).view(-1, 1)

        # ----------------------------------------------------------------------
        # Value Network Update
        # ----------------------------------------------------------------------
        with torch.no_grad():
            new_actions_for_v, log_probs_for_v = self.reparameterize_and_sample(state_tensor)
        
        q1_new = self.critic_1(state_tensor, new_actions_for_v)
        q2_new = self.critic_2(state_tensor, new_actions_for_v)
        q_min = torch.min(q1_new, q2_new)
        
        v_target = q_min - self.alpha * log_probs_for_v
        
        v_current = self.value(state_tensor)

        self.value_optimizer.zero_grad()
        v_loss = 0.5 * torch.mean((v_current - v_target)**2)
        v_loss.backward()
        self.value_optimizer.step()

        # ----------------------------------------------------------------------
        # Critic Networks Update
        # ----------------------------------------------------------------------
        with torch.no_grad():
            v_prime = self.target_value(new_state_tensor)
        
        q_target = reward_tensor + self.gamma * v_prime * (1 - done_tensor)
        
        q1_current = self.critic_1(state_tensor, action_tensor)
        q2_current = self.critic_2(state_tensor, action_tensor)
        
        self.critic_1_optimizer.zero_grad()
        q1_loss = 0.5 * torch.mean((q1_current - q_target)**2)
        q1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        q2_loss = 0.5 * torch.mean((q2_current - q_target)**2)
        q2_loss.backward()
        self.critic_2_optimizer.step()

        # ----------------------------------------------------------------------
        # Actor and Alpha Update (Delayed)
        # ----------------------------------------------------------------------
        if self.learn_step_counter % self.delay == 0:
            new_actions_for_pi, log_probs_for_pi = self.reparameterize_and_sample(state_tensor)
            
            q1_pi = self.critic_1(state_tensor, new_actions_for_pi)
            q2_pi = self.critic_2(state_tensor, new_actions_for_pi)
            q_min_pi = torch.min(q1_pi, q2_pi)
            
            self.actor_optimizer.zero_grad()
            actor_loss = (self.alpha * log_probs_for_pi - q_min_pi).mean()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Alpha/Temperature update
            if self.auto_alpha:
                self.alpha_optimizer.zero_grad()
                alpha_loss = -(self.log_alpha * (log_probs_for_pi + self.target_entropy).detach()).mean()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp()

        self.update_network_parameters()
        self.learn_step_counter += 1

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_networks(self):
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()

    def load_networks(self):
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()

class Distributional_Agent(Agent):
    """Placeholder for Distributional SAC Agent. Not recommended to use per checklist."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Warning: Distributional_Agent initialized. Use '--agent_type sac' for stability.")
        

def instanciate_agent(args, env, checkpoint_directory_networks, device):
    """
    Instantiates the appropriate agent (Agent or Distributional_Agent) 
    based on command-line arguments.
    
    Args:
        args: argparse arguments containing agent configuration.
        env: The Gym environment instance.
        checkpoint_directory_networks: Directory to save network checkpoints.
        device: The PyTorch device ('cpu' or 'cuda').
        
    Returns:
        An instance of Agent or Distributional_Agent.
    """
    
    auto_alpha = args.auto_alpha if hasattr(args, 'auto_alpha') else True

    if args.agent_type == 'sac':
        agent = Agent(
            lr_Q=args.lr_Q,
            lr_pi=args.lr_pi,
            input_shape=env.observation_space.shape,
            tau=args.tau,
            env=env,
            gamma=args.gamma if hasattr(args, 'gamma') else 0.99,
            size=args.memory_size,
            batch_size=args.batch_size,
            layer_size=args.layer_size,
            grad_clip=args.grad_clip,
            delay=args.delay,
            checkpoint_directory_networks=checkpoint_directory_networks,
            device=device,
            auto_alpha=auto_alpha,
            lr_alpha=args.lr_alpha if hasattr(args, 'lr_alpha') else 0.0003,
            sac_temperature=args.sac_temperature,
        )
        
    elif args.agent_type == 'distributional':
        
        agent = Distributional_Agent(
            lr_Q=args.lr_Q,
            lr_pi=args.lr_pi,
            lr_alpha=args.lr_alpha,
            input_shape=env.observation_space.shape,
            tau=args.tau,
            env=env,
            gamma=args.gamma if hasattr(args, 'gamma') else 0.99,
            size=args.memory_size,
            batch_size=args.batch_size,
            layer_size=args.layer_size,
            delay=args.delay,
            grad_clip=args.grad_clip,
            checkpoint_directory_networks=checkpoint_directory_networks,
            device=device,
            auto_alpha=True, # Distributional SAC often uses auto_alpha
            sac_temperature=args.sac_temperature,
        )
        
    return agent
