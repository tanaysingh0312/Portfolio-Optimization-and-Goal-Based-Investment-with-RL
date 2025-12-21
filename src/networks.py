import numpy as np
import os
import torch
from typing import Tuple, List

class Network(torch.nn.Module):
    """Abtract class to be inherited by the various critic and actor classes."""
    
    def __init__(self,
                 input_shape: Tuple, 
                 layer_neurons: int, 
                 network_name: str, 
                 checkpoint_directory_networks: str,
                 device: str = 'cpu',
                 ) -> None:
        """Constructor method for the Network class."""
        
        super(Network, self).__init__()
        self.network_name = network_name
        self.checkpoint_directory_networks = checkpoint_directory_networks
        self.checkpoint_file = os.path.join(self.checkpoint_directory_networks, network_name)
        self.device = torch.device(device)
        self.to(self.device)
        
    def save_checkpoint(self) -> None:
        """Saves the network's state dictionary."""
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self) -> None:
        """Loads the network's state dictionary."""
        # Using map_location to ensure correct device mapping on load
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
        
class Actor(Network):
    """
    Implements the Actor network for SAC, which outputs the mean (mu) and log standard deviation (log_sigma)
    of a Gaussian distribution over actions.
    """
    def __init__(self,
                 input_shape: Tuple,
                 n_actions: int, 
                 layer_neurons: int,
                 network_name: str,
                 checkpoint_directory_networks: str,
                 device: str = 'cpu',
                 log_sigma_min: float = -20.0,
                 log_sigma_max: float = 2.0,
                 ) -> None:
        super(Actor, self).__init__(input_shape, layer_neurons, network_name, checkpoint_directory_networks, device)
        input_dim = input_shape[0] 
        self.linear_1 = torch.nn.Linear(input_dim, layer_neurons)
        self.linear_2 = torch.nn.Linear(layer_neurons, layer_neurons)
        self.linear_mu = torch.nn.Linear(layer_neurons, n_actions)
        self.linear_log_sigma = torch.nn.Linear(layer_neurons, n_actions)
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        # No optimizer defined here; it's managed by the Agent.

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a forward pass, outputting the mean (mu) and log standard deviation (log_sigma)."""
        x = torch.nn.functional.relu(self.linear_1(state))
        x = torch.nn.functional.relu(self.linear_2(x))
        mu = self.linear_mu(x)
        log_sigma = self.linear_log_sigma(x)
        # Clamp log_sigma to ensure stability
        log_sigma = torch.clamp(log_sigma, self.log_sigma_min, self.log_sigma_max)
        return mu, log_sigma

    def sample_normal(self, state, reparameterize=True, deterministic=False):
        """
        Samples an action from the policy distribution, calculates log probabilities, 
        and applies the tanh squash function.
        FIX 2: This method is now correctly named `sample_normal`.
        """
        mu, log_sigma = self.forward(state)
        
        if deterministic:
            # Use the mean for deterministic actions (e.g., in evaluation mode)
            action = torch.tanh(mu)
            return action, None
        
        sigma = log_sigma.exp()
        probabilities = torch.distributions.Normal(mu, sigma)
        
        if reparameterize:
            # Reparameterization trick for policy gradient
            actions = probabilities.rsample()
        else:
            # Simple sampling for exploration or non-gradient steps
            actions = probabilities.sample()
            
        action = torch.tanh(actions)
        
        # Log probability with correction for the tanh squash (SAC standard)
        # log_prob = log_pi(a|s) - sum(log(1 - tanh(a_i)^2))
        log_probs = probabilities.log_prob(actions) - torch.log(1 - action.pow(2) + 1e-6)
        
        # Sum log probabilities over action dimensions (d-dimensional action space)
        return action, log_probs.sum(dim=1, keepdim=True)

class Critic(Network):
    """
    Implements the standard Q-network for SAC, which estimates Q(s, a).
    Outputs a single scalar Q-value.
    """
    
    def __init__(self,
                 input_shape: Tuple, 
                 n_actions: int, 
                 layer_neurons: int,
                 network_name: str,
                 checkpoint_directory_networks: str,
                 device: str = 'cpu',
                 ) -> None:
        """Constructor method for the Critic class."""
        
        super(Critic, self).__init__(input_shape, layer_neurons, network_name, checkpoint_directory_networks, device)
        
        # Input is the concatenated state and action vector
        # Assuming observation space is 1D (S,)
        input_dim = input_shape[0] + n_actions
        
        self.linear_1 = torch.nn.Linear(input_dim, layer_neurons)
        self.linear_2 = torch.nn.Linear(layer_neurons, layer_neurons)
        self.linear_Q = torch.nn.Linear(layer_neurons, 1)

    def forward(self, 
                state: torch.Tensor, 
                action: torch.Tensor,
                ) -> torch.Tensor:
        """Performs a forward pass, outputting the Q-value."""
        
        x = torch.cat([state, action], dim=1) 
        
        x = torch.nn.functional.relu(self.linear_1(x))
        x = torch.nn.functional.relu(self.linear_2(x))
        Q_value = self.linear_Q(x)
        return Q_value

class Value(Network):
    """
    Implements the Value network for SAC, which estimates V(s).
    Outputs a single scalar V-value.
    """
    
    def __init__(self,
                 input_shape: Tuple,
                 layer_neurons: int,
                 network_name: str,
                 checkpoint_directory_networks: str,
                 device: str = 'cpu',
                 ) -> None:
        """Constructor method for the Value class."""
        
        super(Value, self).__init__(input_shape, layer_neurons, network_name, checkpoint_directory_networks, device)
        input_dim = input_shape[0] 
        self.linear_1 = torch.nn.Linear(input_dim, layer_neurons)
        self.linear_2 = torch.nn.Linear(layer_neurons, layer_neurons)
        self.linear_V = torch.nn.Linear(layer_neurons, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass, outputting the V-value."""
        x = torch.nn.functional.relu(self.linear_1(state))
        x = torch.nn.functional.relu(self.linear_2(x))
        V_value = self.linear_V(x)
        return V_value

class Distributional_Critic(Network):
    """
    Placeholder for the Distributional Q-network (Q(s, a) as a Gaussian distribution).
    """
    def __init__(self,
                 input_shape: Tuple, 
                 n_actions: int, 
                 layer_neurons: int,
                 network_name: str,
                 checkpoint_directory_networks: str,
                 device: str = 'cpu',
                 log_sigma_min: float = -20.0,
                 log_sigma_max: float = 2.0,
                 ) -> None:
        """Constructor method for the Distributional Critic class."""
        
        super(Distributional_Critic, self).__init__(input_shape, layer_neurons, network_name, checkpoint_directory_networks, device)
        
        # Cast n_actions to int if it's a tuple (from env.action_space.shape)
        if isinstance(n_actions, tuple):
            n_actions_int = int(np.prod(n_actions))
        else:
            n_actions_int = int(n_actions)

        # Input is the concatenated state and action vector
        # Assuming observation space is 1D (S,)
        input_dim = input_shape[0] + n_actions_int
        
        self.linear_1 = torch.nn.Linear(input_dim, layer_neurons)
        self.linear_2 = torch.nn.Linear(layer_neurons, layer_neurons)

        # Two outputs: mean (mu) and log_std (log_sigma) of the Q-distribution (simplified placeholder)
        self.linear_mu = torch.nn.Linear(layer_neurons, 1)
        self.linear_log_sigma = torch.nn.Linear(layer_neurons, 1)
        
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

    def forward(self, 
                state: torch.Tensor, 
                action: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a forward pass, outputting the mean (mu) and log standard deviation (log_sigma) of the Q-distribution."""
        
        x = torch.cat([state, action], dim=1) 
        
        x = torch.nn.functional.relu(self.linear_1(x))
        x = torch.nn.functional.relu(self.linear_2(x))
        
        mu = self.linear_mu(x)
        log_sigma = self.linear_log_sigma(x)
        log_sigma = torch.clamp(log_sigma, self.log_sigma_min, self.log_sigma_max)
        
        return mu, log_sigma
