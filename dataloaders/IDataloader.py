from abc import ABC, abstractmethod
import pandas as pd

class IDataloader(ABC):
    @abstractmethod
    def __init__(self, data_dir: str):
        pass
        
    @abstractmethod
    def get_tickers(self) -> list[str]:
        pass

    @abstractmethod
    def get_data_by_ticker(self, ticker: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_data_by_tickers(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        pass

    @abstractmethod
    def get_data_by_date(self, date: pd.Timestamp) -> dict[str, pd.Series]:
        pass

    @abstractmethod
    def get_close_df_for_all_tickers(self) -> pd.DataFrame:
        pass

    def split_train_test_by_size(self, data: dict[str, pd.DataFrame], test_size: float) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """
        Splits the data into train and test sets based on a specified size.
        """
        train_data = {}
        test_data = {}
        
        for ticker, df in data.items():
            split_index = int(len(df) * (1 - test_size))
            train_data[ticker] = df.iloc[:split_index]
            test_data[ticker] = df.iloc[split_index:]
        
        return train_data, test_data
    
    def split_train_test_by_date(self, data: dict[str, pd.DataFrame], split_date: pd.Timestamp) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """
        Splits the data into train and test sets based on a specified date.
        """
        train_data = {}
        test_data = {}
        
        for ticker, df in data.items():
            if split_date in df.index:
                train_data[ticker] = df[df.index < split_date]
                test_data[ticker] = df[df.index >= split_date]
            else:
                raise ValueError(f"Split date {split_date} not found in the data for ticker {ticker}.")
        
        return train_data, test_data
    
    def split_train_test_by_date_single_df(self, data: pd.DataFrame, split_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into train and test sets based on a specified date.
        """
        if split_date in data.index:
            train_data = data[data.index < split_date]
            test_data = data[data.index >= split_date]
        else:
            raise ValueError(f"Split date {split_date} not found in the data.")
        
        return train_data, test_data

    def take(self, data: dict[str, pd.DataFrame], idx: list[int]) -> dict[str, pd.DataFrame]:
        """
        Takes a subset of the data based on the provided indices.
        """
        subset_data = {}
        
        for ticker, df in data.items():
            subset_data[ticker] = df.iloc[idx]
        
        return subset_data
    
    def crop_data(self, data: dict[str, pd.DataFrame], start_date: pd.Timestamp, end_date: pd.Timestamp) -> dict[str, pd.DataFrame]:
        """
        Crops the data to the specified date range.
        """
        cropped_data = {}
        
        for ticker, df in data.items():
            cropped_data[ticker] = df[(df.index >= start_date) & (df.index <= end_date)]
        
        return cropped_data