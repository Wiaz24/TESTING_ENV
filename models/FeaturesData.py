from dataclasses import dataclass, field
from typing import List, Dict, Callable
from pathlib import Path
import pandas as pd
from .MarketData import MarketData

@dataclass
class FeaturesData:
    _market_data: MarketData = field(default=None, repr=True, init=True)
    _features: List[str] = field(default_factory=list, repr=True, init=False)
    _targets: List[str] = field(default_factory=list, repr=True, init=False)
    _dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self):
        if self._market_data is None:
            raise ValueError("MarketData object cannot be None.")
        
        for ticker in self._market_data.tickers:
            if ticker not in self._dataframes:
                self._dataframes[ticker] = pd.DataFrame(index=self._market_data.df_index)

    @property
    def tickers(self) -> List[str]:
        """Zwraca listę tickerów."""
        return self._market_data.tickers
    
    @property
    def features(self) -> List[str]:
        """Zwraca listę cech."""
        return self._features.copy()
    
    @property
    def targets(self) -> List[str]:
        """Zwraca listę celów."""
        return self._targets.copy()
    
    @property
    def market_data(self) -> MarketData:
        """Zwraca obiekt MarketData."""
        return self._market_data.copy()
    
    @property
    def minimum_date(self) -> pd.Timestamp:
        """Zwraca minimalną datę."""
        min_date = pd.Timestamp.min

        for df in self._dataframes.values():
            if not df.empty:
                min_date = max(min_date, df.index.min())
        return min_date
    
    @property
    def maximum_date(self) -> pd.Timestamp:
        """Zwraca maksymalną datę."""
        max_date = pd.Timestamp.max

        for df in self._dataframes.values():
            if not df.empty:
                max_date = min(max_date, df.index.max())
        return max_date
    
    @property
    def df_index(self) -> pd.Index:
        """Zwraca indeks dataframe'ów."""
        if self._dataframes:
            return next(iter(self._dataframes.values())).index
        return pd.Index([])

    def add_feature(self, name: str, func: Callable[[pd.DataFrame], pd.Series]):
        """
        Add a feature to the features list and apply it to all dataframes.
        """
        self._features.append(name)
        
        for ticker in self.tickers:
            if ticker not in self._dataframes:
                raise ValueError(f"Dataframe for ticker {ticker} is empty.")
            self._dataframes[ticker][name] = func(self._market_data.dataframes[ticker])
            self._dataframes[ticker].dropna(inplace=True)

    def add_target(self, name: str, func: Callable[[pd.DataFrame], pd.Series]):
        """
        Add a target to the targets list and apply it to all dataframes.
        """
        self._targets.append(name)
        for ticker in self.tickers:
            if ticker not in self._dataframes:
                raise ValueError(f"Dataframe for ticker {ticker} is empty.")
            self._dataframes[ticker][name] = func(self._market_data.dataframes[ticker])
            self._dataframes[ticker].dropna(inplace=True)

    def crop_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """
        Crop the dataframes for all tickers to the specified date range.
        """
        if not isinstance(start_date, pd.Timestamp):
            start_date = pd.to_datetime(start_date)
        if not isinstance(end_date, pd.Timestamp):
            end_date = pd.to_datetime(end_date)

        if start_date > end_date:
            raise ValueError("Start date must be less than or equal to end date.")
        if start_date < self.minimum_date or end_date > self.maximum_date:
            raise ValueError("Date range is outside the available data range.")
        

        for ticker in self.tickers:
            df = self._dataframes[ticker]
            if not df.empty:
                self._dataframes[ticker] = df[(df.index >= start_date) & (df.index <= end_date)]
                self._dataframes[ticker].sort_index(inplace=True)
            else:
                raise ValueError(f"Dataframe for ticker {ticker} is empty.")

    def copy(self) -> 'FeaturesData':
        """
        Create a deep copy of the FeaturesData object.
        """
        features_data = FeaturesData(self._market_data)
        features_data._features = self._features.copy()
        features_data._targets = self._targets.copy()
        features_data._dataframes = {ticker: df.copy() for ticker, df in self._dataframes.items()}
        return features_data

    def split_by_date(self, date: pd.Timestamp) -> tuple['FeaturesData', 'FeaturesData']:
        """
        Split the dataframes into two sets: before and after the specified date.
        """
        before = self.copy()
        after = self.copy()

        before.crop_data(self.minimum_date, date)
        after.crop_data(date, self.maximum_date)
        
        return before, after

    def get_features_for_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Get features for a specific ticker.
        """
        if ticker not in self.tickers:
            raise ValueError(f"Ticker {ticker} not found in the dataset.")
        return self._dataframes[ticker].loc[:, self.features].copy()
    
    def get_targets_for_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Get targets for a specific ticker.
        """
        if ticker not in self.tickers:
            raise ValueError(f"Ticker {ticker} not found in the dataset.")
        return self._dataframes[ticker].loc[:, self.targets].copy()

    def get_features_for_day(self, day: pd.Timestamp) -> 'FeaturesData':
        """
        Get features for a specific day for all tickers.
        """
        features = FeaturesData(self.tickers)
        features.features = self.features
        features.targets = self.targets

        for ticker in self.tickers:
            features.dataframes[ticker] = self.dataframes[ticker].loc[day]
            features.dataframes[ticker].sort_index(inplace=True)
        return features
    
    def get_targets_for_day(self, day: pd.Timestamp) -> dict[str, pd.DataFrame]:
        """
        Get targets for a specific day for all tickers.
        """
        targets = {}
        for ticker in self.tickers:
            targets[ticker] = self.dataframes[ticker].loc[day]
        return targets

    def check_for_inf(self) -> bool:
        """
        Check if any of the dataframes contain infinite values.
        """
        for ticker, df in self._dataframes.items():
            if df.isin([float('inf'), float('-inf')]).any().any():
                print(f"Inf values found in {ticker}")
                return True
        return False
    
    def check_for_nan(self) -> bool:
        """
        Check if any of the dataframes contain NaN values.
        """
        for ticker, df in self._dataframes.items():
            if df.isna().any().any():
                print(f"NaN values found in {ticker}")
                return True
        return False
