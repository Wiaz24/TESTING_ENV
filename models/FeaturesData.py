from dataclasses import dataclass, field
from typing import List, Dict, Callable
from pathlib import Path
import pandas as pd
from .MarketData import MarketData
from sklearn.preprocessing import MinMaxScaler

@dataclass
class FeaturesData:
    _market_data: MarketData = field(default=None, repr=True, init=True)
    _features: List[str] = field(default_factory=list, repr=True, init=False)
    _target: str = field(default_factory=list, repr=True, init=False)
    _target_lag: int = field(default=1, repr=True, init=False)
    _dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict, repr=False, init=False)

    _is_normalized: bool = field(default=False, repr=True, init=False)
    _scalers: Dict[str, Dict[str, MinMaxScaler]] = field(default_factory=dict, repr=False, init=False)

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
    def target(self) -> str:
        """Zwraca cel."""
        return self._target
    
    @property
    def target_lag(self) -> int:
        """Zwraca opóźnienie celu."""
        return self._target_lag
    
    @property
    def market_data(self) -> MarketData:
        """Zwraca obiekt MarketData."""
        return self._market_data
    
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
    
    @property
    def dataframes(self) -> Dict[str, pd.DataFrame]:
        """Zwraca słownik dataframe'ów."""
        return {ticker: df.copy() for ticker, df in self._dataframes.items()}
    
    @property
    def is_normalized(self) -> bool:
        """Zwraca informację, czy dane są znormalizowane."""
        return self._is_normalized
    
    @property
    def scalers(self) -> Dict[str, Dict[str, MinMaxScaler]]:
        """Zwraca słownik scalerów."""
        return {ticker: scaler.copy() for ticker, scaler in self._scalers.items()}

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
        Add a target and apply it to all dataframes.
        """
        self._target = name
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
        features_data._target = self._target
        features_data._target_lag = self._target_lag
        features_data._is_normalized = self._is_normalized
        features_data._scalers = {ticker: scaler.copy() for ticker, scaler in self._scalers.items()}
        features_data._market_data = self._market_data
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
    
    def get_target_for_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Get targets for a specific ticker.
        """
        if ticker not in self.tickers:
            raise ValueError(f"Ticker {ticker} not found in the dataset.")
        return self._dataframes[ticker].loc[:, self.target].copy()

    def get_features_for_day(self, day: pd.Timestamp) -> 'FeaturesData':
        """
        Get features for a specific day for all tickers.
        """
        features = FeaturesData(self.tickers)
        features._features = self.features
        features._target = self.target
        features._target_lag = self.target_lag

        for ticker in self.tickers:
            features._dataframes[ticker] = self._dataframes[ticker].loc[day]
            features._dataframes[ticker].sort_index(inplace=True)
        return features
    
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
    
    def normalize(self, lower: float = -1, upper: float = 1):
        """
        Normalize the dataframes using MinMaxScaler.
        """
        if self._is_normalized:
            raise ValueError("Data is already normalized.")
        
        for ticker in self.tickers:
            df = self._dataframes[ticker].copy()
            self._scalers[ticker] = {}

            for feature in self.features:
                if feature not in df.columns:
                    raise ValueError(f"Feature {feature} not found in dataframe for ticker {ticker}.")
                scaler = MinMaxScaler(feature_range=(lower, upper))
                df[feature] = scaler.fit_transform(df[[feature]])
                self._scalers[ticker][feature] = scaler

            self._dataframes[ticker] = df
        self._is_normalized = True

    # def denormalize_data(self, data: pd.DataFrame | pd.Series, ticker: str) -> pd.DataFrame:
    #     """
    #     Denormalize all columns of provided dataframe or Series using the scaler for the specified ticker and target column.
    #     """
    #     if ticker not in self.tickers:
    #         raise ValueError(f"Ticker {ticker} not found in the dataset.")
    #     if ticker not in self._scalers:
    #         raise ValueError(f"Scaler for ticker {ticker} not found.")
        
    #     if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series):
    #         raise ValueError("Data must be a DataFrame or Series.")
    #     scaler = self._scalers[ticker][self.target]
    #     if isinstance(data, pd.DataFrame):
    #         for col in data.columns:
    #             data[col] = scaler.inverse_transform(data[[col]])
    #         return data
    #     elif isinstance(data, pd.Series):
    #         unscaled_data = scaler.inverse_transform(data.values.reshape(-1, 1)).flatten()
    #         return pd.Series(unscaled_data, index=data.index, name=self.target)
