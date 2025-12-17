import gym
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
                 
                 # --- NEW GBI PARAMETERS ---
                 target_return_rate: float = 0.25,
                 max_steps_per_episode: int = 252,
                 # --------------------------
                 
                 use_corr_matrix: bool = False,
                 use_corr_eigenvalues: bool = False,
                 window: int = 20,
                 number_of_eigenvalues: int = 10,
                 ) -> None:
        """Constructor method for the Environment class."""
        
        self.stock_market_history = stock_market_history
        self.stock_history_size = self.stock_market_history.shape[0]
        self.window = window
        self.number_of_eigenvalues = number_of_eigenvalues
        
        # initial portfolio
        self.cash_in_bank = initial_portfolio['cash_in_bank']
        self.number_of_shares = initial_portfolio['number_of_shares']
        self.n_stocks = len(self.number_of_shares)
        
        # costs and rules
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.bank_rate = bank_rate
        self.limit_n_stocks = limit_n_stocks
        self.buy_rule = buy_rule

        # --- GBI GOAL DEFINITION ---
        self.target_return_rate = target_return_rate
        self.max_steps_per_episode = max_steps_per_episode
        self.initial_portfolio_value = self._get_portfolio_value()
        # The explicit goal the agent must achieve by the end of the episode
        self.goal_value = self.initial_portfolio_value * (1.0 + self.target_return_rate)
        self.current_step = 0 # Tracks time-to-goal
        # ---------------------------
        
        # observation space (state vector dimension)
        self.observation_space_dimension = (1 + # cash in bank
                                            self.n_stocks + # stock prices
                                            self.n_stocks + # owned shares
                                            # --- NEW: +1 for Time-to-Goal Feature ---
                                            1)
        
        if use_corr_matrix:
            self.stock_market_history = append_corr_matrix(self.stock_market_history, window=window)
            self.observation_space_dimension += self.n_stocks * self.n_stocks
        
        if use_corr_eigenvalues:
            self.stock_market_history = append_corr_matrix_eigenvalues(self.stock_market_history, window=window, number_of_eigenvalues=number_of_eigenvalues)
            self.observation_space_dimension += number_of_eigenvalues
            
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_space_dimension,))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n_stocks,))
        
        self.reset()
        
    def reset(self) -> np.array:
        """Reset the environment to a new random starting point."""
        
        self.step_in_history = 0
        self.steps_remaining = self.stock_history_size - self.max_steps_per_episode
        self.start_index = np.random.choice(self.steps_remaining)
        
        self.current_data = self.stock_market_history.iloc[self.start_index: self.start_index + self.max_steps_per_episode]
        self.portfolio_value = self._get_portfolio_value()
        
        # Reset GBI trackers
        self.current_step = 0
        self.initial_portfolio_value = self.portfolio_value
        self.goal_value = self.initial_portfolio_value * (1.0 + self.target_return_rate)
        
        return self._get_observation()
    
    # ... (Other methods like _next_day, _buy_stock, _sell_stock remain the same)

    def step(self, action: np.array) -> Tuple[np.array, float, bool, dict]:
        """Performs one step of the environment."""
        
        self.terminal = False
        
        # Clip action to action space limits
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Get dollar value of actions
        dollar_value_action = action * self.portfolio_value
        
        # Determine whether to buy or sell for each stock based on action sign
        for idx in range(self.n_stocks):
            if dollar_value_action[idx] > 0:
                self._buy_stock(idx, dollar_value_action[idx])
            else:
                self._sell_stock(idx, abs(dollar_value_action[idx]))
                
        # Update prices, bank rate, and calculate base reward
        self._next_day()
        
        new_portfolio_value = self._get_portfolio_value()
        
        # BASE REWARD: Log-return from day-to-day (standard SAC)
        reward = np.log(new_portfolio_value / self.portfolio_value)
        self.portfolio_value = new_portfolio_value
        
        # 2. Advance time step
        self.current_step += 1
        
        # 3. --- GBI TERMINAL REWARD LOGIC (CRITICAL FOR PROJECT UNIQUENESS) ---
        if self.current_step >= self.max_steps_per_episode:
            self.terminal = True
            
            # Calculate the final portfolio value's performance relative to the goal
            performance_ratio = new_portfolio_value / self.goal_value
            
            # Explicit, large terminal reward structure for Goal-Based Investing
            if performance_ratio >= 1.0:
                # Large bonus for reaching/exceeding the goal
                terminal_reward = 10.0 + 50.0 * (performance_ratio - 1.0) # Bonus for exceeding
            else:
                # Severe penalty for missing the goal, scaled by how far it was missed
                terminal_reward = -100.0 * (1.0 - performance_ratio) # Penalty for shortfall
            
            reward += terminal_reward # Add the terminal goal outcome to the last step's reward
            
            # Print status at the end of the GBI episode
            print(f"\n--- GBI Episode End ---")
            print(f"Initial Value: ${self.initial_portfolio_value:,.2f}")
            print(f"Goal Value: ${self.goal_value:,.2f} ({self.target_return_rate*100:.0f}%)")
            print(f"Final Value: ${new_portfolio_value:,.2f}")
            print(f"Terminal Reward Added: {terminal_reward:.2f}")
            print("-----------------------")
        # -------------------------------------------------------------------

        # 4. Prepare next observation
        observation_ = self._get_observation()
        
        return observation_, reward, self.terminal, {}
    
    
    def _get_observation(self) -> np.array:
        """Observation in the format given by the state_space, and perceived by the agent.
        
        The final element is the Time-to-Goal ratio (1.0 at start, 0.0 at end).

        Returns:
            np.array for the observation
        """
        
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
            extra_features = self.current_data.iloc[self.step_in_history][self.n_stocks+1:].values
            observation[offset+1 : ] = extra_features
        
        return observation
    
    def _get_portfolio_value(self) -> float:
        """Performs the scalar product of the owned shares and the stock prices and add the bank account."""
        
        portfolio_value = self.cash_in_bank + np.dot(self.number_of_shares, self.stock_prices)
        
        return portfolio_value
