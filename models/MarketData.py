from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd

@dataclass
class MarketData:
    data_dir: Path = field(default_factory=Path, repr=True, init=True)
    # Pola prywatne z wartościami domyślnymi, niedostępne bezpośrednio z zewnątrz
    _tickers: List[str] = field(default_factory=list, repr=True, init=False)
    _features: List[str] = field(default_factory=list, repr=True, init=False)
    _minimum_date: Optional[pd.Timestamp] = field(default=None, repr=True, init=False)
    _maximum_date: Optional[pd.Timestamp] = field(default=None, repr=True, init=False)
    _dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict, repr=False, init=False)

    @property
    def tickers(self) -> List[str]:
        """Zwraca listę tickerów."""
        return self._tickers.copy()
    
    @property
    def features(self) -> List[str]:
        """Zwraca listę cech."""
        return self._features.copy()
    
    @property
    def minimum_date(self) -> pd.Timestamp:
        """Zwraca minimalną datę."""
        return self._minimum_date
    
    @property
    def maximum_date(self) -> pd.Timestamp:
        """Zwraca maksymalną datę."""
        return self._maximum_date
    
    @property
    def dataframes(self) -> Dict[str, pd.DataFrame]:
        """Zwraca kopie dataframe'ów dla każdego tickera."""
        # Zwracamy kopie aby zapobiec modyfikacji z zewnątrz
        return {ticker: df.copy() for ticker, df in self._dataframes.items()}
    
    @property
    def close_df(self) -> pd.DataFrame:
        """Zwraca dataframe z zamknięciami dla wszystkich tickerów."""
        df = pd.DataFrame(index=self.df_index)
        if self._dataframes:
            for ticker, df_ticker in self._dataframes.items():
                if 'Close' in df_ticker.columns:
                    df[ticker] = df_ticker['Close']
                else:
                    print(f"Warning: 'Close' column not found for ticker {ticker}")
            return df
        return pd.DataFrame()
    
    @property
    def df_index(self) -> pd.Index:
        """Zwraca indeks dataframe'ów."""
        if self._dataframes:
            return next(iter(self._dataframes.values())).index
        return pd.Index([])
    
    def __post_init__(self):
        if not isinstance(self.data_dir, Path):
            self.data_dir = Path(self.data_dir)
        self._load_data()
    
    def _load_data(self):
        """
        Prywatna metoda wczytująca dane i inicjalizująca pola klasy.
        """
        # Sprawdzenie czy katalog istnieje
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory {self.data_dir} does not exist")
        
        ticker_files = list(self.data_dir.glob("*.csv"))
        if not ticker_files:
            raise FileNotFoundError(f"No ticker files found in {self.data_dir}")
        
        for file_path in ticker_files:
            ticker = file_path.stem
            
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            df = df.sort_index()
            
            if not df.empty:
                self._dataframes[ticker] = df
                self._tickers.append(ticker)
        
        # Ustalanie cech na podstawie pierwszego dataframe
        if self._dataframes:
            first_ticker = next(iter(self._dataframes))
            self._features = list(self._dataframes[first_ticker].columns)

        print(f"Loaded data for {len(self._tickers)} tickers")

        self._set_dates(False)
        self.crop_data(self._minimum_date, self._maximum_date) # Przytnij dane do zakresu dat
    
    def _set_dates(self, verbose: bool = False):
        """
        Prywatna metoda ustalająca minimalną i maksymalną datę na podstawie dataframe'ów.
        """
        min_date = pd.Timestamp.min
        max_date = pd.Timestamp.max

        for df in self._dataframes.values():
            if not df.empty:
                min_date = max(min_date, df.index.min())
                max_date = min(max_date, df.index.max())

        self._minimum_date = min_date
        self._maximum_date = max_date

        if verbose:
            print(f"All tickers have data from {self._minimum_date} to {self._maximum_date}")
    
    def crop_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """
        Crop the dataframes for all tickers to the specified date range.
        """
        for ticker, df in self._dataframes.items():
            self._dataframes[ticker] = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Ustalanie nowych dat
        self._set_dates(True)

