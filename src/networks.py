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
        """Loads the network's state dictionary from the saved checkpoint."""
        torch.load(self.checkpoint_file)

class Actor(Network):
    """Actor Network for the SAC agent. Outputs the mean and standard deviation of the action distribution."""
    
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
        """Constructor method for the Actor class."""
        
        super(Actor, self).__init__(input_shape=input_shape,
                                    layer_neurons=layer_neurons,
                                    network_name=network_name,
                                    checkpoint_directory_networks=checkpoint_directory_networks,
                                    device=device)

        # Input is the state vector
        self.linear_1 = torch.nn.Linear(*input_shape, layer_neurons)
        self.linear_2 = torch.nn.Linear(layer_neurons, layer_neurons)

        # Two outputs: mean (mu) and log_std (log_sigma)
        self.linear_mu = torch.nn.Linear(layer_neurons, n_actions)
        self.linear_log_sigma = torch.nn.Linear(layer_neurons, n_actions)
        
        # Hyperparameters for clamping the log_sigma
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)

    def forward(self, 
                state: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a forward pass, outputting the mean and log-standard deviation."""
        
        x = torch.nn.functional.gelu(self.linear_1(state))
        x = torch.nn.functional.gelu(self.linear_2(x))

        mu = self.linear_mu(x)
        log_sigma = self.linear_log_sigma(x)
        
        # Clamping log_sigma to ensure stability
        log_sigma = torch.clamp(log_sigma, min=self.log_sigma_min, max=self.log_sigma_max)

        return mu, log_sigma

    def sample(self, 
               state: torch.Tensor,
               reparameterize: bool = True,
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples an action from the resulting distribution using the reparameterization trick."""
        
        mu, log_sigma = self.forward(state)
        sigma = log_sigma.exp()
        
        # Create the normal distribution
        normal = torch.distributions.Normal(mu, sigma)
        
        # Sample an action
        if reparameterize:
            actions = normal.rsample() # Uses reparameterization trick for training
        else:
            actions = normal.sample()  # Standard sample for evaluation/target calculation
            
        # Log-probability of the sampled action (for SAC entropy term)
        log_probs = normal.log_prob(actions).sum(axis=-1, keepdim=True)
        
        # The actions are in the range (-inf, inf), we squash them to [-1, 1] using tanh
        actions = torch.tanh(actions)
        
        # Correction term for log_probs due to tanh squashing
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6).sum(axis=-1, keepdim=True)
        
        return actions, log_probs, torch.tanh(mu) # Return sampled action, log_prob, and deterministic action

class Critic(Network):
    """Q-Network (Critic) for the SAC agent. Estimates Q(s, a)."""
    
    def __init__(self,
                 input_shape: Tuple,
                 n_actions: int,
                 layer_neurons: int, 
                 network_name: str, 
                 checkpoint_directory_networks: str,
                 device: str = 'cpu',
                 ) -> None:
        """Constructor method for the Critic class."""
        
        super(Critic, self).__init__(input_shape=input_shape,
                                     layer_neurons=layer_neurons,
                                     network_name=network_name,
                                     checkpoint_directory_networks=checkpoint_directory_networks,
                                     device=device)

        # Input is the concatenated state and action vector
        self.linear_1 = torch.nn.Linear(*input_shape + n_actions, layer_neurons)
        self.linear_2 = torch.nn.Linear(layer_neurons, layer_neurons)
        self.linear_3 = torch.nn.Linear(layer_neurons, 1) # Output is a single Q-value

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)

    def forward(self, 
                state: torch.Tensor, 
                action: torch.Tensor,
                ) -> torch.Tensor:
        """Performs a forward pass, outputting the Q-value."""
        
        # Concatenate state and action
        x = torch.cat([state, action], dim=1) 
        
        x = torch.nn.functional.gelu(self.linear_1(x))
        x = torch.nn.functional.gelu(self.linear_2(x))
        q = self.linear_3(x)
        
        return q

class Value(Network):
    """Value Network V(s) for the SAC agent. Estimates the state value."""
    
    def __init__(self,
                 input_shape: Tuple,
                 layer_neurons: int, 
                 network_name: str, 
                 checkpoint_directory_networks: str,
                 device: str = 'cpu',
                 ) -> None:
        """Constructor method for the Value class."""
        
        super(Value, self).__init__(input_shape=input_shape,
                                     layer_neurons=layer_neurons,
                                     network_name=network_name,
                                     checkpoint_directory_networks=checkpoint_directory_networks,
                                     device=device)

        # Input is the state vector
        self.linear_1 = torch.nn.Linear(*input_shape, layer_neurons)
        self.linear_2 = torch.nn.Linear(layer_neurons, layer_neurons)
        self.linear_3 = torch.nn.Linear(layer_neurons, 1) # Output is a single Value

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)

    def forward(self, 
                state: torch.Tensor,
                ) -> torch.Tensor:
        """Performs a forward pass, outputting the Value."""
        
        x = torch.nn.functional.gelu(self.linear_1(state))
        x = torch.nn.functional.gelu(self.linear_2(x))
        v = self.linear_3(x)
        
        return v
        
class Distributional_Critic(Network):
    """Distributional Critic Network. Outputs the mean and standard deviation of the Q-value distribution."""
    
    # NOTE: The implementation of the Distributional_Critic will mirror the structure
    # of the Actor but takes both state and action as input and outputs the distribution
    # parameters for the Q-value. The full implementation would require changes
    # to the SAC loss, but for this project, we'll keep the required network structure
    # consistent with the SAC paper's distributional variant.
    
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
        """Constructor method for the Distributional_Critic class."""
        
        super(Distributional_Critic, self).__init__(input_shape=input_shape,
                                                    layer_neurons=layer_neurons,
                                                    network_name=network_name,
                                                    checkpoint_directory_networks=checkpoint_directory_networks,
                                                    device=device)
                                                    
        # Input is the concatenated state and action vector
        input_dim = input_shape[0] + n_actions
        self.linear_1 = torch.nn.Linear(input_dim, layer_neurons)
        self.linear_2 = torch.nn.Linear(layer_neurons, layer_neurons)

        # Two outputs: mean (mu) and log_std (log_sigma) of the Q-distribution
        self.linear_mu = torch.nn.Linear(layer_neurons, 1)
        self.linear_log_sigma = torch.nn.Linear(layer_neurons, 1)
        
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)

    def forward(self, 
                state: torch.Tensor, 
                action: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a forward pass, outputting the mean (mu) and standard deviation (sigma) of the Q-distribution."""
        
        x = torch.cat([state, action], dim=1) 
        
        x = torch.nn.functional.gelu(self.linear_1(x))
        x = torch.nn.functional.gelu(self.linear_2(x))

        mu = self.linear_mu(x)
        log_sigma = self.linear_log_sigma(x)
        
        log_sigma = torch.clamp(log_sigma, min=self.log_sigma_min, max=self.log_sigma_max)
        sigma = log_sigma.exp()

        return mu, sigma
