import numpy as np
import os
import pandas as pd
import time

from src.utilities import plot_reward, plot_portfolio_value
class Logger():
    """A helper class to better handle the saving of outputs."""
    
    def __init__(self,
                 mode: str,
                 checkpoint_directory: str,
                 ) -> None:
        """Constructor method for the Logger class.
        
        Args:
            mode (bool): 'train' of 'test'
            checkpoint_directory: path to the main checkpoint directory, in which the logs
                                  and plots subdirectories are located
                                  
        Returns:
            no value
        """
        
        self.mode = mode
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_directory_logs = os.path.join(self.checkpoint_directory, "logs")
        self.checkpoint_directory_plots = os.path.join(self.checkpoint_directory, "plots")
        
        self.logs = {'reward_history': []}
        if self.mode =='test':
            self.logs['portfolio_value_history_of_histories'] = []
            self.logs['portfolio_content_history_of_histories'] = []
            
        self.time_stamp = [0, 0]
        # Initialize to None, will be set by Run.__init__ (FIX 4)
        self.initial_value_portfolio = None 
            
    def set_time_stamp(self, i: int) -> None:
        """Sets a time stamp for measuring episode duration."""
        self.time_stamp[i-1] = time.time()

    def _store_initial_value_portfolio(self,
                                       initial_value_portfolio: float,
                                       ) -> None:
        """Setter method for the initial_portfolio_value attribute (FIX 4)."""
        
        self.initial_value_portfolio = initial_value_portfolio

    def print_status(self, episode: int) -> None:
        """Prints the current episode status."""
        
        reward = self.logs['reward_history'][-1]
        
        # FIX 4: Check for initialization before division
        norm_reward = reward / self.initial_value_portfolio if self.initial_value_portfolio else reward
        
        episode_duration = self.time_stamp[1] - self.time_stamp[0]
        
        print(f' episode: {episode:<13d} | reward: {reward:<10.2f} | norm reward: {norm_reward:<10.2f} | duration: {episode_duration:<5.2f}s')

    def save_logs_and_plots(self, n_episodes: int = 50000) -> None:
        """Saves the logs to disk and generates plots."""
        
        print('>>>>> Saving logs and plots <<<<<')
        
        os.makedirs(self.checkpoint_directory_logs, exist_ok=True)
        os.makedirs(self.checkpoint_directory_plots, exist_ok=True)
        
        reward_history_array = np.array(self.logs['reward_history'])
        np.save(os.path.join(self.checkpoint_directory_logs, self.mode+"_reward_history"), reward_history_array)

        x = [i + 1 for i in range(len(reward_history_array))]
        plot_reward(x=x,
                    rewards=reward_history_array,
                    figure_file=os.path.join(self.checkpoint_directory_plots, self.mode+"_reward"),
                    mode=self.mode,
                    bins=np.sqrt(n_episodes).astype(int))
        
        if self.mode =='test':
            portfolio_value_history_of_histories_array = np.array(self.logs['portfolio_value_history_of_histories'])
            np.save(os.path.join(self.checkpoint_directory_logs, self.mode+"_portfolio_value_history"), portfolio_value_history_of_histories_array)
            
            n_days = portfolio_value_history_of_histories_array.shape[1]
            days = [i+1 for i in range(n_days)]
            
            # Select a subset of episodes to plot for clarity (min of n_episodes or 5)
            idx = np.random.choice(len(portfolio_value_history_of_histories_array), min(len(portfolio_value_history_of_histories_array), 5), replace=False)
            
            plot_portfolio_value(x=days, 
                                 values=portfolio_value_history_of_histories_array[idx], 
                                 figure_file=os.path.join(self.checkpoint_directory_plots, self.mode+"_portfolioValue"))
                                 
            portfolio_content_history_array = np.array(self.logs['portfolio_content_history_of_histories'])
            np.save(os.path.join(self.checkpoint_directory_logs, self.mode+"_portfolio_content_history"), portfolio_content_history_array)
      
    def portfolio_content_to_dataframe(self,
                                       tickers: str,
                                       i: int,
                                       ) -> pd.DataFrame:
        """Converts the portfolio content history of a specific episode to a DataFrame."""
        
        portfolio_content_history_array = np.array(self.logs['portfolio_content_history_of_histories'])[i]
        df = pd.DataFrame(data=portfolio_content_history_array, columns=tickers)
        
        return df
