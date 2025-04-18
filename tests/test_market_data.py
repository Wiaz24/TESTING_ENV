import unittest
from pathlib import Path
import pandas as pd
import os
import shutil
import tempfile
from datetime import datetime
from models.MarketData import MarketData

class TestMarketData(unittest.TestCase):
    
    def setUp(self):
        """Przygotowanie środowiska testowego przed każdym testem."""
        # Utwórz tymczasowy katalog dla danych testowych
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Utwórz przykładowe dane dla tickerów
        self.create_sample_data()
        
        # Inicjalizuj obiekt MarketData dla testów
        self.market_data = MarketData(self.test_dir)
    
    def tearDown(self):
        """Sprzątanie po każdym teście."""
        # Usuń tymczasowy katalog po zakończeniu testu
        shutil.rmtree(self.test_dir)
    
    def create_sample_data(self):
        """Tworzenie próbnych danych CSV dla testów."""
        # Dane dla pierwszego tickera
        data1 = {
            'Date': pd.date_range(start='2020-01-01', end='2020-01-10'),
            'Open': [100, 102, 104, 103, 105, 107, 108, 109, 110, 111],
            'High': [105, 107, 108, 106, 110, 112, 112, 114, 115, 116],
            'Low': [98, 100, 102, 101, 103, 105, 106, 107, 108, 109],
            'Close': [102, 104, 103, 105, 107, 108, 109, 110, 111, 112],
            'Volume': [1000, 1200, 900, 1100, 1300, 1400, 1500, 1600, 1700, 1800]
        }
        df1 = pd.DataFrame(data1)
        df1.set_index('Date', inplace=True)
        
        # Dane dla drugiego tickera (krótszy okres)
        data2 = {
            'Date': pd.date_range(start='2020-01-02', end='2020-01-09'),
            'Open': [50, 51, 52, 53, 54, 55, 56, 57],
            'High': [55, 56, 57, 58, 59, 60, 61, 62],
            'Low': [48, 49, 50, 51, 52, 53, 54, 55],
            'Close': [51, 52, 53, 54, 55, 56, 57, 58],
            'Volume': [500, 550, 600, 650, 700, 750, 800, 850]
        }
        df2 = pd.DataFrame(data2)
        df2.set_index('Date', inplace=True)
        
        # Zapisanie danych do plików CSV
        df1.to_csv(self.test_dir / 'AAPL.csv')
        df2.to_csv(self.test_dir / 'MSFT.csv')

    def test_load_data(self):
        market_data = MarketData(Path("data/tickers"))
        self.assertEqual(len(market_data.dataframes), 50)

    
    def test_initialization(self):
        """Test inicjalizacji obiektu MarketData."""
        self.assertEqual(len(self.market_data.tickers), 2)
        self.assertIn('AAPL', self.market_data.tickers)
        self.assertIn('MSFT', self.market_data.tickers)
        
        expected_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.assertEqual(self.market_data.features, expected_features)
        
        # Sprawdź daty
        self.assertEqual(self.market_data.minimum_date, pd.Timestamp('2020-01-02'))
        self.assertEqual(self.market_data.maximum_date, pd.Timestamp('2020-01-09'))
    
    def test_read_only_properties(self):
        """Test, czy właściwości są tylko do odczytu."""
        with self.assertRaises(AttributeError):
            self.market_data.tickers = ['GOOG']
        
        with self.assertRaises(AttributeError):
            self.market_data.features = ['Price']
        
        with self.assertRaises(AttributeError):
            self.market_data.minimum_date = pd.Timestamp('2021-01-01')
        
        with self.assertRaises(AttributeError):
            self.market_data.maximum_date = pd.Timestamp('2021-01-10')
    
    def test_dataframes_property(self):
        """Test, czy property dataframes zwraca kopie."""
        # Pobierz dataframe'y
        dfs = self.market_data.dataframes
        
        # Zmodyfikuj je
        dfs['AAPL']['Close'] = 0
        
        # Sprawdź, czy oryginalne dataframe'y nie zostały zmienione
        new_dfs = self.market_data.dataframes
        self.assertFalse((new_dfs['AAPL']['Close'] == 0).all())
    
    def test_crop_data(self):
        """Test metody crop_data."""
        # Przytnij dane do węższego zakresu
        start_date = pd.Timestamp('2020-01-03')
        end_date = pd.Timestamp('2020-01-07')
        
        self.market_data.crop_data(start_date, end_date)
        
        # Sprawdź, czy daty zostały zaktualizowane
        self.assertEqual(self.market_data.minimum_date, start_date)
        self.assertEqual(self.market_data.maximum_date, end_date)
        
        # Sprawdź, czy dane zostały rzeczywiście przycięte
        for ticker, df in self.market_data.dataframes.items():
            self.assertTrue(df.index.min() >= start_date)
            self.assertTrue(df.index.max() <= end_date)
    
    def test_nonexistent_directory(self):
        """Test zachowania dla nieistniejącego katalogu."""
        non_existent_dir = Path('/non/existent/directory')
        with self.assertRaises(FileNotFoundError):
            MarketData(non_existent_dir)
    
    def test_empty_directory(self):
        """Test zachowania dla pustego katalogu."""
        empty_dir = Path(tempfile.mkdtemp())
        try:
            with self.assertRaises(FileNotFoundError):
                MarketData(empty_dir)
        finally:
            shutil.rmtree(empty_dir)
    
    def test_data_consistency(self):
        """Sprawdź spójność wczytanych danych."""
        df_aapl = self.market_data.dataframes['AAPL']
        df_msft = self.market_data.dataframes['MSFT']
        
        # Sprawdź, czy dane zostały wczytane poprawnie
        common_dates = df_aapl.index.intersection(df_msft.index)
        
        # Powinny być 8 wspólnych dat (od 2 do 9 stycznia)
        self.assertEqual(len(common_dates), 8)
        
        # Sprawdź wartości dla konkretnej daty
        test_date = pd.Timestamp('2020-01-05')
        self.assertIn(test_date, df_aapl.index)
        self.assertEqual(df_aapl.loc[test_date, 'Close'], 107)

    def test_additional_methods(self):
        """
        Test dla dodatkowych metod, które możesz chcieć dodać do klasy MarketData.
        
        Możesz dostosować te testy w zależności od tego, jakie dodatkowe metody 
        planujesz dodać do swojej klasy.
        """
        # Przykład - jeśli dodasz metodę get_returns
        # Zakładając, że masz metodę zwracającą zwroty:
        # returns = self.market_data.get_returns('Close')
        # self.assertIsInstance(returns, pd.DataFrame)
        # self.assertEqual(returns.shape[1], 2)  # 2 tickery
        pass


if __name__ == '__main__':
    unittest.main()