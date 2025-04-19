import numpy as np
import pandas as pd
import cvxpy as cp
from skfolio.optimization import BaseOptimization, ObjectiveFunction
from skfolio import RiskMeasure

class WorstCaseOmega(BaseOptimization):
    def __init__(
        self, 
        delta=0.8,  # Parametr ryzyka-zwrotu (risk-return preference)
        l1_coef=0,
        l2_coef=0,
        min_weight=0,
        max_weight=1,
        portfolio_params=None
    ):
        super().__init__(portfolio_params=portfolio_params)
        self.delta = delta
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.min_weight = min_weight
        self.max_weight = max_weight
        
    def fit(self, X, y=None):
        """Dopasuj model Worst-Case Omega do danych.
        
        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Zwroty aktywów.
        
        y : Ignored
            Nie używane, obecne dla spójności API.
            
        Returns
        -------
        self : object
            Zwraca samego siebie.
        """
        # Konwersja danych wejściowych do numpy array
        X = np.asarray(X)
        n_assets = X.shape[1]
        
        # Definiowanie zmiennych dla modelu optymalizacji
        weights = cp.Variable(n_assets)
        psi = cp.Variable(1)  # Zmienna psi z równania (1) artykułu
        eta = {}  # Zmienne pomocnicze eta
        
        # Podstawowe ograniczenia
        constraints = [
            cp.sum(weights) == 1,
            weights >= self.min_weight,
            weights <= self.max_weight
        ]
        
        # Przygotuj dane dla modelu worst-case omega
        # Dla uproszczenia używamy dwóch rozkładów normalnych (j=1,2)
        # Pierwszy używa pierwszych 10 obserwacji, drugi używa ostatnich 10
        for j in range(2):
            if j == 0 and X.shape[0] >= 10:
                sample_returns = X[:10]
            elif j == 1 and X.shape[0] >= 20:
                sample_returns = X[-10:]
            else:
                # Jeśli mamy mniej niż 20 obserwacji, użyj wszystkich
                sample_returns = X
                
            T_j = sample_returns.shape[0]
            mean_returns_j = np.mean(sample_returns, axis=0)
            
            # Utwórz zmienne eta dla tego rozkładu
            eta[j] = cp.Variable(T_j)
            
            # Dodaj ograniczenie z równania (2) z artykułu
            constraints.append(
                self.delta * (weights @ mean_returns_j) - (1 - self.delta) / T_j * cp.sum(eta[j]) >= psi
            )
            
            # Dodaj ograniczenia z równań (3) i (4)
            for t in range(T_j):
                constraints.append(eta[j][t] >= -weights @ sample_returns[t])
                constraints.append(eta[j][t] >= 0)
        
        # Cel: maksymalizacja psi (równanie (1))
        objective = cp.Maximize(psi)
        
        # Rozwiąż problem optymalizacji
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                # Jeśli nie znaleziono optymalnego rozwiązania, użyj równych wag
                self.weights_ = np.ones(n_assets) / n_assets
            else:
                self.weights_ = weights.value
                
        except cp.error.SolverError:
            # W przypadku błędu solvera, użyj równych wag
            self.weights_ = np.ones(n_assets) / n_assets
        
        return self