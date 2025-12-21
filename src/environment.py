import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Tuple

from src.utilities import append_corr_matrix, append_corr_matrix_eigenvalues
class Environment(gym.Env):
    """Environment for stock trading, modified for Goal-Based Investment (GBI).
    
    The state space now includes the 'Time-to-Goal' feature, and the terminal reward
    is heavily penalized or rewarded based on reaching the specific investment goal.
    
    Attributes:
        observation_space (gym.spaces.Box): (bank account balance, stocks price, owned shares, Time-to-Goal)
        action_space (gym.spaces.Box): cube [-1,1]^n_stocks. positive value: buy, negative value: sale
    """

    def __init__(self, 
                 stock_market_history: pd.DataFrame,                
                 initial_portfolio: dict,
                 buy_cost: float = 0.001,
                 sell_cost: float = 0.001,
                 bank_rate: float = 0.5,
                 limit_n_stocks: float = 200,
                 buy_rule: str = 'most_first',
                 target_return_rate: float = 0.25,
                 max_steps_per_episode: int = 252,
                 use_corr_matrix: bool = False,
                 use_corr_eigenvalues: bool = False,
                 window: int = 20,
                 number_of_eigenvalues: int = 10,
                 # FIX: Compatibility patch to accept extra args from main.py without error
                 limit_trading: bool = False,
                 **kwargs
                 ) -> None:
        
        self.stock_market_history = stock_market_history
        self.stock_history_size = self.stock_market_history.shape[0]
        self.window = window
        self.number_of_eigenvalues = number_of_eigenvalues
        self.use_corr_matrix = use_corr_matrix
        self.use_corr_eigenvalues = use_corr_eigenvalues
        
        self.cash_in_bank = initial_portfolio['cash_in_bank']
        self.number_of_shares = initial_portfolio['number_of_shares']
        self.n_stocks = len(self.number_of_shares)
        
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.bank_rate = bank_rate
        self.limit_n_stocks = limit_n_stocks
        self.buy_rule = buy_rule
        
        # Apply goal/time_horizon values if they were passed via kwargs (from main.py)
        goal = kwargs.get('goal')
        if goal is not None:
             # Assuming 'goal' is factor (e.g., 1.1) and target_return_rate is percentage (e.g., 0.1)
             self.target_return_rate = goal - 1.0 
        else:
             self.target_return_rate = target_return_rate

        time_horizon = kwargs.get('time_horizon')
        if time_horizon is not None:
             self.max_steps_per_episode = time_horizon
        else:
             self.max_steps_per_episode = max_steps_per_episode

        self.current_step = 0
        
        # Determine the starting point in the data history based on the window size
        # We start at 'window' to allow for correlation matrix computation
        self.step_in_history = self.window 
        self.current_data = self.stock_market_history.iloc[self.step_in_history - self.window : self.step_in_history + self.max_steps_per_episode + 1]

        # FIX: Initialize stock_prices BEFORE calculating initial_portfolio_value
        # Initialize stock prices at the starting step (the day *before* the first trade day)
        self.stock_prices = self.current_data.iloc[self.window-1][:self.n_stocks].values
        
        # This call now works because self.stock_prices is defined
        self.initial_portfolio_value = self._get_portfolio_value()
        self.goal_value = self.initial_portfolio_value * (1.0 + self.target_return_rate)

        # Calculate the dimension of the observation space (cash + prices + shares + time_to_goal)
        self.observation_space_dimension = 1 + 2 * self.n_stocks + 1 
        
        # Adjust dimension if using correlation features
        if self.use_corr_matrix:
            self.observation_space_dimension += self.n_stocks ** 2
        elif self.use_corr_eigenvalues:
            self.observation_space_dimension += self.number_of_eigenvalues
        
        # Define observation space bounds
        high = np.inf * np.ones(self.observation_space_dimension)
        low = -high
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Define action space (fractional weights to buy/sell/hold)
        high = 1.0 * np.ones(self.n_stocks)
        low = -1.0 * np.ones(self.n_stocks)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        
    def reset(self, seed=None, options=None):
        """Resets the state of the environment and returns an initial observation."""

        super().reset(seed=seed)
        # Reset time in episode
        self.current_step = 0

        # Determine a new random starting point in the history for this episode
        # The episode must fit entirely within the remaining history
        end_point = self.stock_history_size - self.max_steps_per_episode
        if end_point < self.window:
            # Not enough data for a full episode with the window size
            self.step_in_history = self.window
        else:
            # Start randomly between window and the point that allows a full episode
            self.step_in_history = np.random.randint(self.window, end_point + 1)
        
        # Slice the data for the current episode
        self.current_data = self.stock_market_history.iloc[self.step_in_history - self.window : self.step_in_history + self.max_steps_per_episode + 1]

        # Reset portfolio to initial values
        self.cash_in_bank = self.initial_portfolio_value  # Use the total initial value
        self.number_of_shares = np.zeros(self.n_stocks)
        
        # Update stock prices for the reset step (must be the one just before the first trading day)
        self.stock_prices = self.current_data.iloc[self.window-1][:self.n_stocks].values
        
        # Return the initial observation (state)
        observation = self._get_observation()
        return observation, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes one time step within the environment.
        """
        
        # Move to the next day's price data
        self.current_step += 1

        max_step = len(self.current_data) - self.window - 1
        if self.current_step >= max_step:
            terminated = True
            truncated = False
            reward = 0.0
            info = {}

            final_portfolio_value = self._get_portfolio_value()
            if final_portfolio_value >= self.goal_value:
                reward += 1000.0
            else:
                reward += -1000.0

            observation = self._get_observation()
            return observation, reward, terminated, truncated, info
        
        # Check for episode termination
        terminated = self.current_step >= self.max_steps_per_episode
        truncated = False
        
        # Get stock prices for the start of the current day (used for trade execution)
        # Note: self.current_step is 1-indexed (1 to max_steps_per_episode)
        current_stock_prices = self.current_data.iloc[self.window + self.current_step - 1][:self.n_stocks].values
        
        # --- Trade Execution ---
        
        # Action is normalized between -1 and 1
        # action > 0 means buy, action < 0 means sell
        
        # Calculate the net amount of shares to buy/sell
        # The total action value should be proportional to the current portfolio value
        current_portfolio_value = self._get_portfolio_value()
        
        # The trade amount for each stock (in currency)
        trade_amount_currency = current_portfolio_value * action 
        
        # Process selling and buying separately to ensure valid cash flow
        
        # 1. Selling (action < 0)
        sell_actions_currency = np.maximum(0, -trade_amount_currency)
        shares_to_sell = np.floor(sell_actions_currency / current_stock_prices)
        
        # Limit selling to available shares
        shares_to_sell = np.minimum(shares_to_sell, self.number_of_shares)
        
        total_sell_value = np.sum(shares_to_sell * current_stock_prices)
        sell_cost = total_sell_value * self.sell_cost
        
        # Update cash and shares after selling
        self.cash_in_bank += (total_sell_value - sell_cost)
        self.number_of_shares -= shares_to_sell
        
        # 2. Buying (action > 0)
        buy_actions_currency = np.maximum(0, trade_amount_currency)
        shares_to_buy_float = buy_actions_currency / current_stock_prices
        
        # Limit shares to buy based on cash available (after selling)
        shares_to_buy_max_cash = np.floor(self.cash_in_bank / current_stock_prices)
        
        # Determine final shares to buy (integer only)
        shares_to_buy = np.floor(np.minimum(shares_to_buy_float, shares_to_buy_max_cash))
        
        total_buy_value = np.sum(shares_to_buy * current_stock_prices)
        buy_cost = total_buy_value * self.buy_cost
        
        # Update cash and shares after buying
        self.cash_in_bank -= (total_buy_value + buy_cost)
        self.number_of_shares += shares_to_buy

        # --- Update Prices and Get New State ---

        # Prices for the end of the day (used for next day's observation/portfolio value)
        self.stock_prices = self.current_data.iloc[self.window + self.current_step][:self.n_stocks].values
        
        # Calculate the portfolio value after all trades and price changes
        new_portfolio_value = self._get_portfolio_value()

        # --- Reward Calculation ---
        
        # The reward is the daily log return of the portfolio
        log_return = np.log(new_portfolio_value / current_portfolio_value)
        reward = log_return
        
        # --- Terminal Reward (GBI) ---
        if terminated:
            final_portfolio_value = new_portfolio_value
            
            if final_portfolio_value >= self.goal_value:
                # Large positive reward for reaching the goal
                terminal_reward = 1000.0
            else:
                # Large negative penalty for failing the goal
                terminal_reward = -1000.0
                
            reward += terminal_reward

        # Get the next observation
        observation = self._get_observation()
        
        info = {'portfolio_value': new_portfolio_value}

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Returns the current state of the environment as an observation vector."""
        
        # Calculate time-to-goal (normalized from 1.0 down to 0.0)
        time_to_goal_ratio = (self.max_steps_per_episode - self.current_step) / self.max_steps_per_episode
        
        # The size is self.observation_space_dimension
        observation = np.empty(self.observation_space_dimension)
        
        # Populate the observation vector with existing features
        offset = 0
        observation[offset] = self.cash_in_bank
        offset += 1

        observation[offset : offset + self.n_stocks] = self.stock_prices
        offset += self.n_stocks
        
        observation[offset : offset + self.n_stocks] = self.number_of_shares
        offset += self.n_stocks
        
        # Append the new time-to-goal feature at the end
        observation[offset] = time_to_goal_ratio 
        
        # Add correlation matrix or eigenvalues if configured
        if self.use_corr_matrix or self.use_corr_eigenvalues:
            # Assumes the utilities correctly format and append these features
            # Indexing: self.window + self.current_step gives the index into self.current_data
            idx = self.window + self.current_step
            if 0 <= idx < len(self.current_data):
                extra_features = self.current_data.iloc[idx][:self.n_stocks].values
            else:
                extra_features = np.zeros(self.n_stocks)
            observation[offset+1 : ] = extra_features
        
        return observation
    
    def _get_portfolio_value(self) -> float:
        """Performs the scalar product of the owned shares and the stock prices and add the bank account."""
        
        portfolio_value = self.cash_in_bank + np.dot(self.number_of_shares, self.stock_prices)
        
        return portfolio_value
