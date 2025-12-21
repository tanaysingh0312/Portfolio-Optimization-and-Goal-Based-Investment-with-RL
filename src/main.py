import os
import pandas as pd
from pathlib import Path
from typing import List
import argparse
import json
import numpy as np

from src.agents import instanciate_agent
from src.environment import Environment
from src.run import Run
from src.utilities import append_corr_matrix, append_corr_matrix_eigenvalues, create_directory_tree, instanciate_scaler # Import utilities for data augmentation

class DataFetcher():
    """
    A class to fetch stock data from Yahoo Finance (yfinance) and save it locally.
    """
    
    def __init__(self,
                 stock_symbols: List[str],
                 start_date: str = "2010-01-01",
                 end_date: str = "2020-12-31",
                 directory_path: str = "data",
                 ) -> None:
        """
        Constructor for DataFetcher.
        """
        
        # Ensure the data directory exists
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        
        self.stock_symbols = stock_symbols
        self.start_date = start_date
        self.end_date = end_date
        self.directory_path = directory_path
        
    def fetch_and_merge_data(self) -> None:
        """
        Fetches data from Yahoo Finance for all symbols and merges them into a single CSV.
        """
        import yfinance as yf
        print('>>>>> Fetching data from Yahoo Finance <<<<<')
        
        all_data = []
        for symbol in self.stock_symbols:
            try:
                data = yf.download(symbol, start=self.start_date, end=self.end_date)
                data['symbol'] = symbol
                data = data[['Close', 'symbol']]
                data.columns = ['Close', 'symbol']
                all_data.append(data)
            except Exception as e:
                print(f"Could not download data for {symbol}: {e}")

        if all_data:
            df = pd.concat(all_data)
            df = df.pivot_table(index=df.index, columns='symbol', values='Close')
            df.to_csv(os.path.join(self.directory_path, 'stocks.csv'))
            print(f'>>>>> Data saved to {os.path.join(self.directory_path, "stocks.csv")} <<<<<')
        else:
            print("No data was successfully fetched.")


class Preprocessor():
    """
    A class to preprocess the raw stock data.
    """

    def __init__(self, 
                 df_directory: str = 'data', 
                 file_name: str = 'stocks.csv',
                 ) -> None:
        """
        Constructor for Preprocessor.
        """

        self.df_directory = df_directory
        self.file_name = file_name
        self.df_path = os.path.join(self.df_directory, self.file_name)
        self.df = pd.read_csv(self.df_path, index_col=0, parse_dates=True)
        
    def collect_close_prices(self) -> pd.DataFrame:
        """
        Selects only the Close prices.
        """
        
        self.df.index.name = 'Date'
        self.df.columns.name = 'Ticker'
        return self.df
        
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Fills missing values using forward fill (fillna(method='ffill')) and then drops
        any remaining NaNs (usually from the beginning of the series).
        """
        
        self.df.fillna(method='ffill', inplace=True)
        self.df.dropna(inplace=True)
        
        self.df.to_csv(os.path.join(self.df_directory, 'close.csv'))
        
        return self.df

def load_data(tickers_subset: str, 
              mode: str, 
              time_horizon: int,
              use_corr_matrix: bool = False,
              use_corr_eigenvalues: bool = False,
              window: int = 20,
              n_eigenvalues: int = 10,
              ) -> pd.DataFrame:
    """
    Loads stock data, applies subsetting, and performs training/testing splits and
    data augmentation (correlation matrix/eigenvalues).
    
    Args:
        tickers_subset (str): Path to the file containing the list of tickers to use.
        mode (str): 'train' or 'test'.
        time_horizon (int): Total number of days in the dataset (needed for splitting).
        use_corr_matrix (bool): Whether to append correlation matrix features.
        use_corr_eigenvalues (bool): Whether to append correlation eigenvalues features.
        window (int): Sliding window size for correlation calculation.
        n_eigenvalues (int): Number of eigenvalues to append.
        
    Returns:
        pd.DataFrame: The processed stock data.
    """

    if os.path.exists('data/stocks.csv') and not os.path.exists('data/close.csv'):
        print('>>>>> Extracting close prices and handling missing values <<<<<')
        preprocessor = Preprocessor(df_directory='data', file_name='stocks.csv')
        df = preprocessor.collect_close_prices()
        df = preprocessor.handle_missing_values()
    
    elif os.path.exists('data/close.csv'):
        print('\n>>>>> Reading the preprocessed data <<<<<')
        df = pd.read_csv('data/close.csv', index_col=0)
        
        # FIX: Check if the ticker file exists before trying to open it
        if not os.path.exists(tickers_subset):
            # Provide helpful debug information if the file is missing
            available_files = os.listdir('portfolios_and_tickers') if os.path.exists('portfolios_and_tickers') else "Folder not found"
            raise FileNotFoundError(
                f"Ticker subset file not found: {tickers_subset}\n"
                f"Available files in portfolios_and_tickers/: {available_files}"
            )
            
        with open(tickers_subset) as f:
            stocks_subset = f.read().splitlines()
            stocks_subset = [ticker for ticker in stocks_subset if ticker in df.columns]
            
        df = df[stocks_subset]
    
    else:
        raise FileNotFoundError("Data files not found.")
        
    # --- Data Augmentation ---
    # Append correlation features if requested
    if use_corr_matrix:
        df = append_corr_matrix(df=df, window=window)
    
    if use_corr_eigenvalues:
        df = append_corr_matrix_eigenvalues(df=df, window=window, number_of_eigenvalues=n_eigenvalues)

    # --- Train/Test Split ---
    # Use the entire loaded time series length for splitting
    data_size = df.shape[0] 
    
    if mode == 'train':
        # Train data is the first 75%
        df = df.iloc[: 3*data_size//4]
    elif mode == 'test':
        # Test data is the last 25%
        df = df.iloc[3*data_size//4:]
    
    # Ensure the dataframe still has enough data points for the given time horizon
    if df.shape[0] < time_horizon:
        raise ValueError(f"Not enough data points ({df.shape[0]}) for the specified time horizon ({time_horizon}) after train/test split.")
        
    print(f'>>>>> Data loaded for {mode} mode, shape: {df.shape} <<<<<')
    
    return df


def _read_tickers_file(path: str) -> List[str]:
    with open(path) as f:
        tickers = f.read().splitlines()
    return [t for t in tickers if t]


def _load_initial_portfolio_for_env(initial_portfolio_path: str, tickers: List[str]) -> dict:
    with open(initial_portfolio_path, 'r') as f:
        portfolio = json.load(f)

    cash_in_bank = float(portfolio.get('Bank_account', 0.0))
    number_of_shares = np.array([float(portfolio.get(ticker, 0.0)) for ticker in tickers], dtype=np.float32)
    return {
        'cash_in_bank': cash_in_bank,
        'number_of_shares': number_of_shares,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--n_episodes', type=int, default=2)
    parser.add_argument('--time_horizon', type=int, default=252)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--assets_to_trade', type=str, default='portfolios_and_tickers/tickers_S&P500_dummy.txt')
    parser.add_argument('--initial_portfolio', type=str, default='portfolios_and_tickers/initial_portfolio_subset.json')

    parser.add_argument('--start_date', type=str, default='2010-01-01')
    parser.add_argument('--end_date', type=str, default='2020-12-31')

    parser.add_argument('--checkpoint_directory', type=str, default=None)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--use_scaler', action='store_true')

    parser.add_argument('--use_corr_matrix', action='store_true')
    parser.add_argument('--use_corr_eigenvalues', action='store_true')
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--number_of_eigenvalues', type=int, default=10)

    parser.add_argument('--buy_cost', type=float, default=0.001)
    parser.add_argument('--sell_cost', type=float, default=0.001)
    parser.add_argument('--bank_rate', type=float, default=0.5)
    parser.add_argument('--limit_n_stocks', type=float, default=200)
    parser.add_argument('--buy_rule', type=str, default='most_first')
    parser.add_argument('--target_return_rate', type=float, default=0.25)
    parser.add_argument('--sac_temperature', type=float, default=0.1)

    parser.add_argument('--lr_Q', type=float, default=0.0003)
    parser.add_argument('--lr_pi', type=float, default=0.0003)
    parser.add_argument('--lr_alpha', type=float, default=0.0003)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--layer_size', type=int, default=256)
    parser.add_argument('--memory_size', type=int, default=1000000)
    parser.add_argument('--delay', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--auto_alpha', action='store_true', default=True)
    parser.add_argument('--no_auto_alpha', action='store_false', dest='auto_alpha')
    parser.add_argument('--agent_type', type=str, default='sac', choices=['sac', 'distributional'])
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    tickers = _read_tickers_file(args.assets_to_trade)

    if not os.path.exists('data/close.csv'):
        fetcher = DataFetcher(
            stock_symbols=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            directory_path='data',
        )
        fetcher.fetch_and_merge_data()

        preprocessor = Preprocessor(df_directory='data', file_name='stocks.csv')
        preprocessor.collect_close_prices()
        preprocessor.handle_missing_values()

    df = load_data(
        tickers_subset=args.assets_to_trade,
        mode=args.mode,
        time_horizon=args.time_horizon,
        use_corr_matrix=args.use_corr_matrix,
        use_corr_eigenvalues=args.use_corr_eigenvalues,
        window=args.window,
        n_eigenvalues=args.number_of_eigenvalues,
    )

    initial_portfolio = _load_initial_portfolio_for_env(args.initial_portfolio, tickers=tickers)

    env = Environment(
        stock_market_history=df,
        initial_portfolio=initial_portfolio,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
        bank_rate=args.bank_rate,
        limit_n_stocks=args.limit_n_stocks,
        buy_rule=args.buy_rule,
        target_return_rate=args.target_return_rate,
        max_steps_per_episode=args.time_horizon,
        use_corr_matrix=args.use_corr_matrix,
        use_corr_eigenvalues=args.use_corr_eigenvalues,
        window=args.window,
        number_of_eigenvalues=args.number_of_eigenvalues,
    )

    checkpoint_directory = create_directory_tree(checkpoint_directory=args.checkpoint_directory, mode=args.mode)
    checkpoint_directory_networks = os.path.join(checkpoint_directory, 'networks')

    scaler = instanciate_scaler(
        use_scaler=args.use_scaler,
        env=env,
        mode=args.mode,
        checkpoint_directory=checkpoint_directory,
    )

    agent = instanciate_agent(args=args, env=env, checkpoint_directory_networks=checkpoint_directory_networks, device=args.device)

    runner = Run(
        env=env,
        agent=agent,
        n_episodes=args.n_episodes,
        agent_type=args.agent_type,
        scaler=scaler,
        checkpoint_directory=checkpoint_directory,
        sac_temperature=args.sac_temperature,
        mode=args.mode,
        plot=args.plot,
    )

    # ================================
    # RUN TRAINING / TESTING EPISODES
    # ================================

    for episode in range(args.n_episodes):
        runner.run_episode()
        print(f"[INFO] Episode {episode + 1}/{args.n_episodes} completed")

    # Save networks at the end of training (ensure they're always saved)
    if args.mode == 'train':
        runner.agent.save_networks()
        print("[INFO] Networks saved to disk")

    # Save logs and plots after training
    runner.logger.save_logs_and_plots(args.n_episodes)


if __name__ == '__main__':
    main()
