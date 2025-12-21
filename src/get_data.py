import os
import pandas as pd
from pathlib import Path
from typing import List
import yfinance as yf

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
        Fetches data for each stock symbol and merges them into a single file 
        (data/stocks.csv) after saving individual files.
        """
        
        final_df = None
        
        for stock in self.stock_symbols:
            
            file_path = os.path.join(self.directory_path, "{}.csv".format(stock))
            
            # Check if the individual file already exists
            if not os.path.exists(file_path):
                
                print('Fetching data for {}'.format(stock))
                try:
                    # Download data from yfinance
                    df = yf.download(stock, start=self.start_date, end=self.end_date)
                    df.to_csv(file_path)
                except Exception as e:
                    print('Could not fetch data for {}. Error: {}'.format(stock, e))
                    continue
                    
            # Read the (newly fetched or existing) individual file
            df = pd.read_csv(file_path)
            
            # Merge the individual stock data into the final DataFrame
            if final_df is None:
                final_df = df
            else:
                # Merge based on common columns (Date, Open, High, Low, Close, Adj Close, Volume)
                final_df = pd.merge(final_df, df, on=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], how='outer')
        
        # Save the combined DataFrame
        if final_df is not None:
            final_df.to_csv(os.path.join(self.directory_path, 'stocks.csv'))


class Preprocessor():
    """
    A class to preprocess the combined stock data, focusing on close prices
    and handling missing values.
    """
    
    def __init__(self,
                 df_directory: str = 'data',
                 file_name: str = 'stocks.csv'
                 ) -> None:
        """
        Constructor for Preprocessor. Loads the combined stock data.
        """
        
        self.df = pd.read_csv(os.path.join(df_directory, file_name), index_col=0)

    def collect_close_prices(self) -> pd.DataFrame:
        """
        Extracts only the 'Close' prices for all stocks from the DataFrame.
        """
        
        # Filter columns to only include those related to 'Close' prices
        close_prices_cols = [col for col in self.df.columns if 'Close' in col]
        close_prices = self.df[close_prices_cols]
        
        # Attempt to access data via multi-index 'Close' level, if present
        try:
            close_prices = self.df.Close
            close_prices.columns = self.df.loc[:, 'Open'].columns
            self.df = close_prices
        except AttributeError:
            # Fallback for flattened columns
            print("Warning: Assuming flattened column names for close prices.")
            self.df = close_prices 
            
        return self.df
    
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Drops columns (stocks) and rows (days) with any missing data 
        and saves the resulting cleaned close prices to 'data/close.csv'.
        """
        
        # Drop columns (stocks) with any NaN values
        self.df = self.df.dropna(axis=1)
        # Drop rows (days) with any NaN values
        self.df = self.df.dropna(axis=0)
        
        self.df.to_csv('data/close.csv')
        
        return self.df

def load_data(initial_date: str,
              final_date: str,
              tickers_subset: str,
              mode: str = 'test',
              scaler=None 
              ) -> pd.DataFrame:
    """Helper function to load and preprocess the stock market data."""
    
    try:
        Path(tickers_subset).parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directory: {e}")

    full_tickers_file = 'portfolios_and_tickers/tickers_S&P500.txt'
    
    stocks_symbols = []
    if os.path.exists(full_tickers_file):
        with open(full_tickers_file) as f:
            stocks_symbols = f.read().splitlines()
      
    if not os.path.exists('data/'):  
        print('\n>>>>> Fetching the data <<<<<')
        fetcher = DataFetcher(stock_symbols=stocks_symbols,
                              start_date=initial_date,
                              end_date=final_date,
                              directory_path="data")
        fetcher.fetch_and_merge_data()
    
    if not os.path.exists('data/close.csv') and os.path.exists('data/stocks.csv'):
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

    time_horizon = df.shape[0]
    if mode == 'train':
        df = df.iloc[: 3*time_horizon//4]
    elif mode == 'test':
        df = df.iloc[3*time_horizon//4 :]
        
    return df
