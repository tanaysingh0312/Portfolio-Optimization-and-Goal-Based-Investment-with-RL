import gym
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
        
        if self.mode == 'test':
            self.agent.load_networks()
        
        if self.scaler is not None and self.mode == 'train':
            self.scaler.fit(self.env.stock_market_history.values)

    def run_episode(self) -> None:
        """Runs one full episode."""
        
        observation = self.env.reset()
        done = False
        reward = 0.0 # Stores cumulative reward for the episode
        
        if self.mode == 'test':
            portfolio_value_history = [self.env.initial_portfolio_value]
            portfolio_content_history = [self.env.number_of_shares]
        
        while not done:
            
            # Scale observation if scaler is provided (scaler handles all features including the new GBI features)
            if self.scaler is not None:
                # We do not scale the last feature (Time-to-Goal Ratio) as it's already normalized [0, 1]
                # Scale all features *except* the last one (the Time-to-Goal ratio)
                scaled_observation_features = self.scaler.transform(observation[:-1].reshape(1, -1))[0]
                
                # Reconstruct the observation: scaled features + unscaled Time-to-Goal ratio
                scaled_observation = np.append(scaled_observation_features, observation[-1])
                action = self.agent.choose_action(scaled_observation)
            else:
                action = self.agent.choose_action(observation)

            observation_, step_reward, done, info = self.env.step(action)

            # Update cumulative reward
            # Original code had 'reward += reward', which is a typo. It should accumulate the step_reward.
            reward += step_reward
            
            self.step += 1
            
            if self.mode == 'test':
                portfolio_value_history.append(self.env._get_portfolio_value())
                portfolio_content_history.append(self.env.number_of_shares)
            
            # The agent stores the *unscaled* observation in the buffer
            self.agent.remember(observation, action, step_reward, observation_, done)
            
            if self.mode == 'train':
                # The agent learns using the *unscaled* data from the buffer
                self.agent.learn(self.step)
                
            observation = observation_
             
        self.logger.logs["reward_history"].append(reward)
        average_reward = np.mean(self.logger.logs["reward_history"][-50:])
        
        if self.mode == 'test':
            self.logger.logs["portfolio_value_history_of_histories"].append(portfolio_value_history)
            self.logger.logs["portfolio_content_history_of_histories"].append(portfolio_content_history)
        
        self.episode += 1
        
        self.logger.set_time_stamp(2)
        self.logger.print_status(episode=self.episode)
        
        if average_reward > self.best_reward:
            self.best_reward = average_reward
            if self.mode == 'train':
                self.agent.save_networks()
