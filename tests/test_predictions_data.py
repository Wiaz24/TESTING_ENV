import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from pathlib import Path
import sys
import os
import tempfile
import shutil

# Zakładamy, że moduł models jest w ścieżce projektu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import klasy PredictionsData
from models.PredictionsData import PredictionsData

class TestPredictionsData(unittest.TestCase):
    def setUp(self):
        """Przygotowanie danych testowych przed każdym testem."""
        # Konfiguracja tickerów
        self.tickers = ['AAPL', 'MSFT', 'GOOG']
        
        # Tworzenie indeksu czasowego dla testów
        self.dates = pd.date_range(start='2023-01-01', end='2023-01-10')
        
        # Tworzenie obiektu PredictionsData do testów
        self.predictions_data = PredictionsData(self.dates, self.tickers)
    
    def test_init(self):
        """Test inicjalizacji obiektu PredictionsData."""
        # Sprawdzenie czy dane są poprawnie zainicjowane
        self.assertEqual(self.predictions_data._tickers, self.tickers)
        self.assertTrue(self.dates.equals(self.predictions_data._index))
        
        # Sprawdzenie czy dataframe'y zostały utworzone
        self.assertEqual(len(self.predictions_data._dataframes), len(self.tickers))
        
        for ticker in self.tickers:
            # Sprawdzenie czy ticker jest w dataframes
            self.assertIn(ticker, self.predictions_data._dataframes)
            
            # Sprawdzenie czy dataframe ma odpowiednie kolumny
            df = self.predictions_data._dataframes[ticker]
            self.assertIn(self.predictions_data._predictions_col, df.columns)
            self.assertIn(self.predictions_data._correct_data_col, df.columns)
            
            # Sprawdzenie czy dataframe ma odpowiedni indeks
            self.assertTrue(df.index.equals(self.dates))
            
            # Sprawdzenie czy kolumny mają odpowiedni typ danych
            self.assertEqual(df[self.predictions_data._predictions_col].dtype, np.float64)
            self.assertEqual(df[self.predictions_data._correct_data_col].dtype, np.float64)
    
    def test_tickers_property(self):
        """Test property tickers."""
        # Sprawdzenie czy zwracana jest kopia
        tickers = self.predictions_data.tickers
        self.assertEqual(tickers, self.tickers)
        
        # Modyfikacja kopii nie powinna wpływać na oryginał
        tickers.append('NEW')
        self.assertEqual(self.predictions_data._tickers, self.tickers)
    
    def test_minimum_date_property(self):
        """Test property minimum_date."""
        # Przypadek, gdy wszystkie DataFrame'y mają jednakowe daty
        min_date = self.predictions_data.minimum_date
        expected_min_date = pd.Timestamp('2023-01-01')
        self.assertEqual(min_date, expected_min_date)
        
        # Przypadek, gdy jeden DataFrame ma późniejszą datę początkową
        # Tworzymy nowy indeks
        later_dates = pd.date_range(start='2023-01-05', end='2023-01-15')
        
        # Tworzymy nowy obiekt PredictionsData z różnymi indeksami dla różnych tickerów
        predictions_data_mixed = PredictionsData(self.dates, self.tickers)
        
        # Ręcznie modyfikujemy dataframe dla jednego tickera
        predictions_data_mixed._dataframes['AAPL'] = pd.DataFrame(index=later_dates)
        predictions_data_mixed._dataframes['AAPL'][predictions_data_mixed._predictions_col] = pd.Series(dtype=np.float64, index=later_dates)
        predictions_data_mixed._dataframes['AAPL'][predictions_data_mixed._correct_data_col] = pd.Series(dtype=np.float64, index=later_dates)
        
        # Sprawdzamy minimalną datę
        min_date = predictions_data_mixed.minimum_date
        expected_min_date = pd.Timestamp('2023-01-05')
        self.assertEqual(min_date, expected_min_date)
    
    def test_maximum_date_property(self):
        """Test property maximum_date."""
        # Przypadek, gdy wszystkie DataFrame'y mają jednakowe daty
        max_date = self.predictions_data.maximum_date
        expected_max_date = pd.Timestamp('2023-01-10')
        self.assertEqual(max_date, expected_max_date)
        
        # Przypadek, gdy jeden DataFrame ma wcześniejszą datę końcową
        # Tworzymy nowy indeks
        earlier_dates = pd.date_range(start='2023-01-01', end='2023-01-07')
        
        # Tworzymy nowy obiekt PredictionsData z różnymi indeksami dla różnych tickerów
        predictions_data_mixed = PredictionsData(self.dates, self.tickers)
        
        # Ręcznie modyfikujemy dataframe dla jednego tickera
        predictions_data_mixed._dataframes['AAPL'] = pd.DataFrame(index=earlier_dates)
        predictions_data_mixed._dataframes['AAPL'][predictions_data_mixed._predictions_col] = pd.Series(dtype=np.float64, index=earlier_dates)
        predictions_data_mixed._dataframes['AAPL'][predictions_data_mixed._correct_data_col] = pd.Series(dtype=np.float64, index=earlier_dates)
        
        # Sprawdzamy maksymalną datę
        max_date = predictions_data_mixed.maximum_date
        expected_max_date = pd.Timestamp('2023-01-07')
        self.assertEqual(max_date, expected_max_date)
    
    def test_add_prediction(self):
        """Test metody add_prediction."""
        # Tworzenie przykładowych danych predykcji
        prediction = pd.Series(np.random.rand(len(self.dates)), index=self.dates)
        
        # Dodanie predykcji
        self.predictions_data.add_prediction('AAPL', prediction)
        
        # Pobieranie serii z DataFrame (po dodaniu)
        result_series = self.predictions_data._dataframes['AAPL'][self.predictions_data._predictions_col]
        
        # Sprawdzenie czy dane predykcji zostały dodane poprawnie
        # Używamy np.testing.assert_array_equal zamiast pd.testing.assert_series_equal
        # aby uniknąć problemów z atrybutem name
        np.testing.assert_array_equal(result_series.values, prediction.values)
        pd.testing.assert_index_equal(result_series.index, prediction.index)
        
        # Test dla nieistniejącego tickera
        with self.assertRaises(ValueError):
            self.predictions_data.add_prediction('NONEXISTENT', prediction)
        
        # Test dla pustej predykcji
        with self.assertRaises(ValueError):
            self.predictions_data.add_prediction('AAPL', pd.Series())
        
        # Test dla predykcji z innym indeksem
        wrong_index = pd.date_range(start='2023-02-01', periods=len(self.dates))
        wrong_prediction = pd.Series(np.random.rand(len(wrong_index)), index=wrong_index)
        
        with self.assertRaises(ValueError):
            self.predictions_data.add_prediction('AAPL', wrong_prediction)
    
    def test_add_correct_data(self):
        """Test metody add_correct_data."""
        # Tworzenie przykładowych poprawnych danych
        correct_data = pd.Series(np.random.rand(len(self.dates)), index=self.dates)
        
        # Dodanie poprawnych danych
        self.predictions_data.add_correct_data('AAPL', correct_data)
        
        # Pobieranie serii z DataFrame (po dodaniu)
        result_series = self.predictions_data._dataframes['AAPL'][self.predictions_data._correct_data_col]
        
        # Sprawdzenie czy dane zostały dodane poprawnie
        # Używamy np.testing.assert_array_equal zamiast pd.testing.assert_series_equal
        # aby uniknąć problemów z atrybutem name
        np.testing.assert_array_equal(result_series.values, correct_data.values)
        pd.testing.assert_index_equal(result_series.index, correct_data.index)
        
        # Test dla nieistniejącego tickera
        with self.assertRaises(ValueError):
            self.predictions_data.add_correct_data('NONEXISTENT', correct_data)
        
        # Test dla pustych poprawnych danych
        with self.assertRaises(ValueError):
            self.predictions_data.add_correct_data('AAPL', pd.Series())
        
        # Test dla poprawnych danych z innym indeksem
        wrong_index = pd.date_range(start='2023-02-01', periods=len(self.dates))
        wrong_correct_data = pd.Series(np.random.rand(len(wrong_index)), index=wrong_index)
        
        with self.assertRaises(ValueError):
            self.predictions_data.add_correct_data('AAPL', wrong_correct_data)
    
    def test_metrics_df(self):
        """Test property metrics_df."""
        # Dodanie przykładowych danych predykcji i poprawnych danych
        for ticker in self.tickers:
            # Rzeczywiste wartości
            correct_data = pd.Series(np.random.rand(len(self.dates)) * 100, index=self.dates)
            
            # Predykcje (nieco różniące się od rzeczywistych)
            prediction = correct_data + np.random.randn(len(self.dates)) * 10
            
            # Dodanie danych
            self.predictions_data.add_prediction(ticker, prediction)
            self.predictions_data.add_correct_data(ticker, correct_data)
        
        # Pobranie metrics_df
        metrics_df = self.predictions_data.metrics_df
        
        # Sprawdzenie czy metrics_df ma odpowiedni kształt
        self.assertEqual(metrics_df.shape, (len(self.tickers), 4))  # 4 metryki
        
        # Sprawdzenie czy wszystkie tickery są w indeksie
        self.assertEqual(set(metrics_df.index), set(self.tickers))
        
        # Sprawdzenie czy wszystkie metryki są nieujemne
        self.assertTrue((metrics_df >= 0).all().all())
        
        # Sprawdzenie czy nazwa indeksu jest "Ticker"
        self.assertEqual(metrics_df.index.name, 'Ticker')
        
        # Sprawdzenie czy wszystkie oczekiwane kolumny są obecne
        expected_columns = ['MSE', 'RMSE', 'MAE', 'MAPE']
        self.assertEqual(list(metrics_df.columns), expected_columns)
        
        # Sprawdzenie czy RMSE jest pierwiastkiem z MSE (z małym marginesem błędu)
        for ticker in self.tickers:
            mse = metrics_df.loc[ticker, 'MSE']
            rmse = metrics_df.loc[ticker, 'RMSE']
            self.assertAlmostEqual(np.sqrt(mse), rmse, places=10)
    
    def test_metrics_df_empty(self):
        """Test property metrics_df dla pustych danych."""
        # Pobranie metrics_df bez dodawania danych
        metrics_df = self.predictions_data.metrics_df
        
        # Sprawdzenie czy metrics_df ma odpowiedni kształt
        self.assertEqual(metrics_df.shape, (len(self.tickers), 4))  # 4 metryki
        
        # Sprawdzenie czy wszystkie wartości są zerami
        self.assertTrue((metrics_df == 0).all().all())

if __name__ == '__main__':
    unittest.main()