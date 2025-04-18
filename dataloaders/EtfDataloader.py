import pandas as pd
import os
from pathlib import Path
from .IDataloader import IDataloader

class EtfDataloader(IDataloader):
    """
    A class to load ETF data.
    """

    def __init__(self, data_dir: str):
        self.data_dir: Path = Path(data_dir)
        self.data: dict[str, pd.DataFrame] = {}
        self.tickers: list[str] = []

        self._load_data()

    def _load_data(self):
        ticker_files = list(self.data_dir.glob("*.csv"))
        if not ticker_files:
            raise FileNotFoundError(f"No ticker files found in {self.data_dir}")
        
        for file_path in ticker_files:
            ticker = file_path.stem
            
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            # df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
            df = df.sort_index()
            
            if not df.empty:
                self.data[ticker] = df
                self.tickers.append(ticker)
        
        print(f"Loaded data for {len(self.tickers)} tickers")

    def get_tickers(self) -> list[str]:
        return self.tickers
    
    def get_data_by_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Returns the data for a specific ticker.
        """
        if ticker not in self.data:
            raise ValueError(f"Ticker {ticker} not found in the loaded data.")
        return self.data[ticker]
    
    def get_data_by_tickers(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """
        Returns the data for a list of tickers.
        """
        result = {}
        for ticker in tickers:
            if ticker in self.data:
                result[ticker] = self.data[ticker]
            else:
                raise ValueError(f"Ticker {ticker} not found in the loaded data.")
        return result
    
    def get_data_by_date(self, date: pd.Timestamp) -> dict[str, pd.Series]:
        """
        Returns the data for all tickers on a specific date.
        """
        result = {}
        for ticker, df in self.data.items():
            if date in df.index:
                result[ticker] = df.loc[date]
            else:
                raise ValueError(f"Date {date} not found in the data for ticker {ticker}.")
        return result

    def get_close_df_for_all_tickers(self, 
                                     start_date: pd.Timestamp = None, 
                                     end_Date: pd.Timestamp = None) -> pd.DataFrame:
        """
        Returns a DataFrame with the close prices for all tickers.
        """
        close_prices = {}
        for ticker, df in self.data.items():
            close_prices[ticker] = df['Close']
        
        close_df = pd.DataFrame(close_prices)
        close_df.index.name = 'Date'
        close_df = close_df.sort_index()
        if start_date and end_Date:
            close_df = close_df[(close_df.index >= start_date) & (close_df.index <= end_Date)]
        close_df = close_df.dropna()
        return close_df
    
if __name__ == "__main__":
    # Example usage
    data_dir = Path("data/tickers")
    start_date = pd.Timestamp("2020-01-01")
    end_date = pd.Timestamp("2021-01-01")

    dataloader = EtfDataloader(data_dir, start_date, end_date)
    # print(dataloader.get_tickers())
    common_df = dataloader.get_close_df_for_all_tickers()
    common_df.info()
    # print(dataloader.get_data_by_ticker("AAPL"))
    # print(dataloader.get_data_by_tickers(["AAPL", "MSFT"]))
    # print(dataloader.get_data_by_date(pd.Timestamp("2020-06-01")))