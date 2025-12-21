from datetime import datetime
import gymnasium as gym
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import seaborn as sns
sns.set_theme()
from sklearn.preprocessing import StandardScaler
from typing import List, Union
 
def create_directory_tree(checkpoint_directory: str = None,
                          simple: bool = False,
                          mode: str = 'train'):
    """Creates the necessary directory structure for saving logs, networks, and plots.

    It creates a timestamped folder inside 'saved_outputs/' unless 'simple' is True
    or 'checkpoint_directory' is explicitly provided.

    Args:
        checkpoint_directory (str, optional): The base directory for saving outputs. 
                                              If None, a new timestamped directory is created. Defaults to None.
        simple (bool, optional): If True, forces the creation of a new timestamped 
                                 directory regardless of 'checkpoint_directory' being None. 
                                 Defaults to False.
        mode (str, optional): Not currently used in directory creation logic but kept for compatibility. 
                              Defaults to 'train'.
                                  
    Returns:
        str: The path to the newly created base checkpoint directory.
    """
    
    date = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    # If simple is True or no checkpoint_directory is provided, create a new timestamped one
    if simple or checkpoint_directory is None:
        checkpoint_directory = os.path.join("saved_outputs", date)

    # Define paths for subdirectories
    checkpoint_directory_networks = os.path.join(checkpoint_directory, "networks")
    checkpoint_directory_logs = os.path.join(checkpoint_directory, "logs")
    checkpoint_directory_plots = os.path.join(checkpoint_directory, "plots")

    # Create the directories
    Path(checkpoint_directory_networks).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_directory_logs).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_directory_plots).mkdir(parents=True, exist_ok=True)

    return checkpoint_directory

def plot_reward(x: List[int], 
                rewards: np.ndarray, 
                figure_file: str, 
                mode: str,
                bins: int = 20,
                ) -> None:
    """Helper function to plot the reward history in train mode and the reward probability distribution in test mode.
    
    Args:
        x (np.array): a linspace, horizontal axis of the plot
        rewards (np.array): reward history
        figure_file (str): filepath of the figure file
        mode (str): train or test to decide what type of plot to generate
        bins (int): number of bins for the histogram, by default set to the square root of the number of samples
        
    Returns:
        no value
    """
    
    running_average = np.zeros(len(rewards))
    for i in range(len(running_average)):
        running_average[i] = np.mean(rewards[max(0, i-50): i+1])
        
    if mode == 'train':
        plt.plot(x, rewards, linestyle='-', color='blue', label='reward')
        plt.plot(x, running_average, linestyle='--', color='green', label='running average 50')
        plt.legend()
        plt.title('Reward as a function of the epoch/episode')
        
    elif mode == 'test':
        plt.hist(rewards, bins=bins)
        plt.title('Reward distribution')
    
    plt.savefig(figure_file) 
    
def plot_portfolio_value(x: List[int], 
                         values: np.ndarray, 
                         figure_file: str, 
                         ) -> None:
    
    plt.plot(x, values.T, linestyle='-', linewidth=0.5)
    plt.xlim((0, len(x)))
    plt.title('Portfolio value')
    plt.savefig(figure_file) 
        
def instanciate_scaler(use_scaler: bool,
                       env: gym.Env = None,
                       mode: str = 'train',
                       checkpoint_directory: str = None) -> Union[StandardScaler, None]:
    """Instanciate and either fit or load the StandardScaler parameters depending on the mode and configuration.

    In train mode, if scaling is enabled, the agent behaves randomly in order to store typical observations 
    in the environment, which are then used to fit the scaler.
    
    Args:
        use_scaler (bool): Flag indicating whether to use scaling or not.
        env (gym.Env, optional): Trading environment. Required if use_scaler is True and mode is 'train'.
        mode (str): train or test.
        checkpoint_directory (str): Directory where the scaler file is saved/loaded. Required if use_scaler is True.
        
    Returns:
        StandardScaler or None: A trained sklearn standard scaler object, or None if use_scaler is False.
    """
    
    if not use_scaler:
        # If scaling is not used, return None immediately.
        return None

    scaler = StandardScaler()
    
    if mode == 'train':
        
        # Check if environment and checkpoint_directory are provided for fitting
        if env is None or checkpoint_directory is None:
            # If the necessary context for fitting is missing, return the instantiated scaler 
            # and warn the user. This addresses the incorrect call location in main.py.
            print("WARNING: 'env' and 'checkpoint_directory' were not provided in 'train' mode when 'use_scaler' is True. Skipping environment interaction for scaler fitting.")
            return scaler

        observations = []
        for _ in range(10):
            observation, info = env.reset()
            observations.append(observation)
            done = False
            while not done:
                action = env.action_space.sample()
                observation_, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                observations.append(observation_)

        scaler.fit(observations)
        Path(os.path.join(checkpoint_directory, 'networks')).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(checkpoint_directory, 'networks', 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    
    if mode == 'test':
        if checkpoint_directory is None:
            print("ERROR: 'checkpoint_directory' must be provided in 'test' mode when scaling is enabled. Returning None.")
            return None
            
        try:
            with open(os.path.join(checkpoint_directory, 'networks', 'scaler.pkl'), 'rb') as f:
                scaler = pickle.load(f)
        except FileNotFoundError:
            print("ERROR: Scaler file not found. Cannot run in 'test' mode with scaling. Returning None.")
            return None
    
    return scaler

def prepare_initial_portfolio(initial_portfolio: Union[int, float, str],
                              tickers: List[str]) -> dict:
    """Prepare the initial portfolio to give to the environment constructor.
    
    Args:
        initial_portfolio (int or float or string): if numerical then initial cash in bank assuming no shares are owned initially, otherwise
                                                    path to a json file prescribing the initial cah in bank as  well as the number of owned shares of each asset.
        tickers (List[str]): list of asset names
        
    Returns:
        dictionary giving the structure of the initial portfolio
    """
    
    print('>>>>> Reading the provided initial portfolio <<<<<')
    
    if isinstance(initial_portfolio, int) or isinstance(initial_portfolio, float):
        initial_portfolio_returned = {key: 0 for key in tickers}
        initial_portfolio_returned["Bank_account"] = initial_portfolio
    
    else:
        with open(initial_portfolio, "r") as file:
            initial_portfolio = json.load(file)
            
        initial_portfolio_returned = {key: 0 for key in tickers}
        initial_portfolio_returned["Bank_account"] = initial_portfolio["Bank_account"]
        
        for key in initial_portfolio_returned.keys():
            if key in initial_portfolio.keys():
                initial_portfolio_returned[key] = initial_portfolio[key]
            
    return initial_portfolio_returned

def append_corr_matrix(df: pd.DataFrame,
                       window: int,
                       ) -> pd.DataFrame:
    """Append the sliding correlation matrix of a multidimensional time series.
        
    timewise flattens it and extracts just the upper triangular part (since it is symmetric), 
    then appends it to the initial time series.
    
    Args:
        df (pd.DataFrame): the multidimensional time series whose sliding correlation matrix is computed
        window (int): size of the sliding window used to compute the correlation matrix
        
    Returns:
        the input time series with the sliding correlation matrix appended
    """

    print('>>>>> Appending the correlation matrix <<<<<')

    columns = ['{}/{}'.format(m, n) for (m, n) in itertools.combinations_with_replacement(df.columns, r=2)]
    corr = df.rolling(window).cov()
    corr_flattened = pd.DataFrame(index=columns).transpose()

    for i in range(df.shape[0]):

        ind = np.triu_indices(df.shape[1])
        data = corr[df.shape[1]*i : df.shape[1]*(i+1)].to_numpy()[ind]
        index = [corr.index[df.shape[1]*i][0]]

        temp = pd.DataFrame(data=data, columns=index, index=columns).transpose()
        corr_flattened = pd.concat([corr_flattened, temp])

    return pd.concat([df, corr_flattened], axis=1).iloc[window-1 : ]

def append_corr_matrix_eigenvalues(df: pd.DataFrame,
                                   window: int,
                                   number_of_eigenvalues: int = 10
                                   ) -> pd.DataFrame:
    """Append the number_of_eigenvalues greatest eigenvalues of the sliding correlation matrix of a multidimensional time series.
        
    Args:
        df (pd.DataFrame): the multidimensional time series whose sliding correlation matrix is computed
        window (int): size of the sliding window used to compute the correlation matrix
        
    Returns:
        the input time series with the number_of_eigenvalues greatest eigenvalues of the sliding correlation matrix appended
    """
    
    print('>>>>> Appending the eigenvalues <<<<<')
    
    if number_of_eigenvalues > df.shape[1]:
        number_of_eigenvalues = df.shape[1]
    
    columns = ['Eigenvalue_{}'.format(m+1) for m in range(number_of_eigenvalues)]
    corr = df.rolling(window).cov()
    corr_eigenvalues = pd.DataFrame(index=columns).transpose()

    for i in range(window-1, df.shape[0]):
        data = corr[df.shape[1]*i : df.shape[1]*(i+1)].to_numpy()
        data = np.linalg.eig(data)
        data = data[0].real
        data[::-1].sort()
        data = data[:number_of_eigenvalues]

        index = [corr.index[df.shape[1]*i][0]]
        temp = pd.DataFrame(data=data, columns=index, index=columns).transpose()
        corr_eigenvalues = pd.concat([corr_eigenvalues, temp])

    print('>>>>> Eigenvalues have been appended <<<<<')

    return pd.concat([df.iloc[window-1 : ], corr_eigenvalues], axis=1)
