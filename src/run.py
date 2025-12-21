import gymnasium as gym
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.agents import Agent
from src.logger import Logger
from src.environment import Environment

class Run():
    """Main class to run the training or the testing."""
    
    def __init__(self, 
                 env: Environment,
                 agent: Agent,
                 n_episodes: int,
                 agent_type: str,
                 scaler: StandardScaler,
                 checkpoint_directory: str,
                 sac_temperature: float = 1.0,
                 mode: str = 'test',
                 plot: bool = False,
                 ) -> None:
        """Constructor method of the class Run."""
        
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.agent_type = agent_type
        self.scaler = scaler
        self.sac_temperature = sac_temperature
        self.mode = mode
        self.plot = plot
        self.best_reward = -np.inf
        self.episode = 1
        self.step = 1
        self.logger = Logger(mode=mode, checkpoint_directory=checkpoint_directory)
        
        # FIX 4: Logger initial portfolio value NEVER SET
        # Call the setter method on the logger instance immediately after environment creation
        self.logger._store_initial_value_portfolio(self.env.initial_portfolio_value)
        
        if self.mode == 'test':
            self.agent.load_networks()

    def run_episode(self):
        """Runs a single episode, including environment interaction and agent learning/evaluation."""
        observation, info = self.env.reset()
        done = False
        reward = 0
        portfolio_value_history = []
        portfolio_content_history = []
        
        self.logger.set_time_stamp(1)
        
        while not done:
            if self.scaler is not None:
                # Scale the observation before feeding to the agent
                observation_scaled = self.scaler.transform([observation])[0]
            else:
                observation_scaled = observation
                
            # Choose action: deterministic for test mode, stochastic for train mode
            action = self.agent.choose_action(observation_scaled, evaluate=(self.mode == 'test'))
            
            # Environment step
            observation_, step_reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Accumulate reward (Note: fixed potential typo in original code logic)
            reward += step_reward
            
            self.step += 1
            
            if self.mode == 'test':
                portfolio_value_history.append(self.env._get_portfolio_value())
                portfolio_content_history.append(self.env.number_of_shares)
            
            # The agent stores the *unscaled* observation in the buffer
            # FIX 3: Agent method mismatch: replace 'remember' with 'store_transition'
            self.agent.store_transition(observation, action, step_reward, observation_, done)
            
            if self.mode == 'train':
                # The agent learns using the *unscaled* data from the buffer
                # FIX 3: Agent method mismatch: replace 'learn(self.step)' with 'learn()'
                self.agent.learn()
                
            observation = observation_
             
        # Logging episode results
        self.logger.logs["reward_history"].append(reward)
        average_reward = np.mean(self.logger.logs["reward_history"][-50:])
        
        if self.mode == 'test':
            self.logger.logs["portfolio_value_history_of_histories"].append(portfolio_value_history)
            self.logger.logs["portfolio_content_history_of_histories"].append(portfolio_content_history)
        
        self.episode += 1
        
        self.logger.set_time_stamp(2)
        self.logger.print_status(episode=self.episode)
        
        # Save networks if current performance is the best so far
        if average_reward > self.best_reward and self.mode == 'train':
            self.best_reward = average_reward
            self.agent.save_networks()
