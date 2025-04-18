import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from pathlib import Path
import sys
import os
import tempfile
import shutil

# # Zakładamy, że moduł models jest w ścieżce projektu
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import klasy MarketData
from models.MarketData import MarketData

class TestMarketData(unittest.TestCase):
    def setUp(self):
        """Przygotowanie danych testowych przed każdym testem."""
        # Tworzenie tymczasowego katalogu dla danych testowych
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        
        # Konfiguracja tickerów
        self.tickers = ['AAPL', 'MSFT', 'GOOG']
        
        # Konfiguracja danych
        self.features = ['open', 'high', 'low', 'close', 'volume']
        
        # Tworzenie przykładowych plików CSV dla każdego tickera
        dates = pd.date_range(start='2023-01-01', end='2023-01-10')
        
        for ticker in self.tickers:
            # Tworzenie przykładowego DataFrame dla każdego tickera
            df = pd.DataFrame({
                'open': np.random.rand(len(dates)) * 100,
                'high': np.random.rand(len(dates)) * 110,
                'low': np.random.rand(len(dates)) * 90,
                'close': np.random.rand(len(dates)) * 100,
                'volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
            
            # Zapisanie DataFrame do pliku CSV
            df.index.name = 'Date'
            file_path = self.data_dir / f"{ticker}.csv"
            df.to_csv(file_path)
    
    def tearDown(self):
        """Czyszczenie po testach."""
        # Usunięcie tymczasowego katalogu
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test inicjalizacji obiektu MarketData."""
        with patch('builtins.print'):  # Pomijamy wyświetlanie komunikatów
            market_data = MarketData(self.data_dir)
        
        # Sprawdzenie czy dane są poprawnie wczytane
        self.assertEqual(set(market_data.tickers), set(self.tickers))
        self.assertEqual(set(market_data.features), set(self.features))
        
        # Sprawdzenie czy dataframe'y zostały utworzone
        self.assertEqual(len(market_data.dataframes), len(self.tickers))
        for ticker in self.tickers:
            self.assertIn(ticker, market_data.dataframes)
    
    def test_tickers_property(self):
        """Test property tickers."""
        with patch('builtins.print'):
            market_data = MarketData(self.data_dir)
        
        # Sprawdzenie czy zwracana jest kopia
        tickers = market_data.tickers
        self.assertEqual(set(tickers), set(self.tickers))
        
        # Modyfikacja kopii nie powinna wpływać na oryginał
        tickers.append('NEW')
        self.assertEqual(set(market_data.tickers), set(self.tickers))
    
    def test_features_property(self):
        """Test property features."""
        with patch('builtins.print'):
            market_data = MarketData(self.data_dir)
        
        # Sprawdzenie czy zwracana jest kopia
        features = market_data.features
        self.assertEqual(set(features), set(self.features))
        
        # Modyfikacja kopii nie powinna wpływać na oryginał
        features.append('NEW')
        self.assertEqual(set(market_data.features), set(self.features))
    
    def test_dataframes_property(self):
        """Test property dataframes."""
        with patch('builtins.print'):
            market_data = MarketData(self.data_dir)
        
        # Pobranie dataframe'ów
        dataframes = market_data.dataframes
        
        # Sprawdzenie czy wszystkie tickery są w dataframes
        self.assertEqual(set(dataframes.keys()), set(self.tickers))
        
        # Sprawdzenie czy zwracane są kopie dataframe'ów
        for ticker in self.tickers:
            # Modyfikacja kopii
            original_value = dataframes[ticker].iloc[0, 0]
            dataframes[ticker].iloc[0, 0] = 999
            
            # Pobranie ponownie dataframe'ów
            new_dataframes = market_data.dataframes
            
            # Sprawdzenie czy oryginał nie został zmodyfikowany
            self.assertEqual(new_dataframes[ticker].iloc[0, 0], original_value)
    
    def test_minimum_date_property(self):
        """Test property minimum_date."""
        with patch('builtins.print'):
            market_data = MarketData(self.data_dir)
        
        # Sprawdzenie czy minimum_date jest ustawione
        self.assertIsNotNone(market_data.minimum_date)
        self.assertEqual(market_data.minimum_date, pd.Timestamp('2023-01-01'))
    
    def test_maximum_date_property(self):
        """Test property maximum_date."""
        with patch('builtins.print'):
            market_data = MarketData(self.data_dir)
        
        # Sprawdzenie czy maximum_date jest ustawione
        self.assertIsNotNone(market_data.maximum_date)
        self.assertEqual(market_data.maximum_date, pd.Timestamp('2023-01-10'))
    
    def test_df_index_property(self):
        """Test property df_index."""
        with patch('builtins.print'):
            market_data = MarketData(self.data_dir)
        
        # Sprawdzenie czy df_index zwraca indeks
        expected_index = pd.date_range(start='2023-01-01', end='2023-01-10')
        expected_index.name = 'Date'  # Dodajemy nazwę do indeksu
        
        # Sprawdzamy wartości indeksu, ale bez porównywania atrybutów (metadanych)
        self.assertTrue(market_data.df_index.equals(expected_index))
        # Sprawdzamy również nazwę indeksu
        self.assertEqual(market_data.df_index.name, 'Date')
    
    def test_crop_data(self):
        """Test metody crop_data."""
        with patch('builtins.print'):
            market_data = MarketData(self.data_dir)
        
        # Daty do przycinania
        start_date = pd.Timestamp('2023-01-03')
        end_date = pd.Timestamp('2023-01-07')
        
        # Przycinanie danych
        market_data.crop_data(start_date, end_date)
        
        # Sprawdzenie czy dane zostały przycięte
        for ticker, df in market_data.dataframes.items():
            self.assertEqual(df.index.min(), start_date)
            self.assertEqual(df.index.max(), end_date)
    
    def test_missing_directory(self):
        """Test czy rzucany jest wyjątek dla nieistniejącego katalogu."""
        non_existent_dir = Path(self.temp_dir) / "non_existent"
        
        with self.assertRaises(FileNotFoundError):
            with patch('builtins.print'):
                MarketData(non_existent_dir)
    
    def test_empty_directory(self):
        """Test czy rzucany jest wyjątek dla pustego katalogu."""
        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir()
        
        with self.assertRaises(FileNotFoundError):
            with patch('builtins.print'):
                MarketData(empty_dir)

if __name__ == '__main__':
    unittest.main()