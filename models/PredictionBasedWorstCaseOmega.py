from dataclasses import dataclass, field
import numpy as np
import cvxpy as cp
from skfolio.optimization import BaseOptimization

@dataclass
class PredictionBasedWorstCaseOmega(BaseOptimization):
    """Prediction based Worst-Case Omega Model.
    
    Model optymalizacji portfela oparty na przewidywaniach, który wykorzystuje
    model worst-case omega i dodatkowe funkcje celu oparte na prognozach
    i krótkoterminowej wydajności.
    """
    
    delta: float = 0.8
    k1: float = 2/8
    k2: float = 2/8
    k3: float = 2/8
    k4: float = 1/8
    k5: float = 1/8
    l1_coef: float = 0
    l2_coef: float = 0
    min_weight: float = 0
    max_weight: float = 1
    portfolio_params: dict = field(default=None)
    
    def __post_init__(self):
        super().__init__(portfolio_params=self.portfolio_params)
        
    def fit(self, X, predictions=None, y=None):
        """Dopasuj model Prediction based worst-case omega."""
        # Konwersja danych wejściowych do numpy array
        X = np.asarray(X)
        n_observations, n_assets = X.shape
        
        if predictions is None:
            predictions = X.copy()
        else:
            predictions = np.asarray(predictions)

        # Zamień logarytmiczne stopy zwrotu na stopy zwrotu
        predictions = np.exp(predictions) - 1
        # Obliczenie błędów predykcyjnych (errors)
        errors = X - predictions
        
        # Obliczenie R20 i R5 (średnie zwroty z ostatnich 20 i 5 dni)
        R20 = np.zeros(n_assets)
        R5 = np.zeros(n_assets)
        
        if n_observations >= 20:
            R20 = np.mean(X[-20:], axis=0)
        else:
            R20 = np.mean(X, axis=0)
            
        if n_observations >= 5:
            R5 = np.mean(X[-5:], axis=0)
        else:
            R5 = np.mean(X, axis=0)
        
        # Przygotuj dane dla modelu worst-case omega
        # Używamy dwóch rozkładów normalnych (j=1,2) zgodnie z artykułem
        mean_errors = []
        sample_errors = []
        
        for j in range(2):
            if j == 0 and n_observations >= 10:
                sample_err = errors[:10]
            elif j == 1 and n_observations >= 20:
                sample_err = errors[-10:]
            else:
                # Jeśli mamy mniej niż 20 obserwacji, użyj wszystkich
                sample_err = errors
                
            mean_errors.append(np.mean(sample_err, axis=0))
            sample_errors.append(sample_err)
        
        # Definiowanie zmiennych dla modelu optymalizacji
        weights = cp.Variable(n_assets)
        psi = cp.Variable(1)  # Zmienna psi z równania (8) artykułu
        eta = {}  # Zmienne pomocnicze eta
        
        # Podstawowe ograniczenia
        constraints = [
            cp.sum(weights) == 1,
            weights >= self.min_weight,
            weights <= self.max_weight
        ]
        
        # Dodajemy ograniczenia związane z modelem worst-case omega
        for j in range(2):
            T_j = sample_errors[j].shape[0]
            eta[j] = cp.Variable(T_j)
            
            # Dodaj ograniczenie z równania (13) z artykułu
            constraints.append(
                self.delta * (weights @ mean_errors[j]) - (1 - self.delta) / T_j * cp.sum(eta[j]) >= psi
            )
            
            # Dodaj ograniczenia z równań (14) i (15)
            for t in range(T_j):
                constraints.append(eta[j][t] >= -weights @ sample_errors[j][t])
                constraints.append(eta[j][t] >= 0)
        
        # Cel: kombinacja liniowa pięciu funkcji celu zgodnie z równaniem (16)
        mean_pred = np.mean(predictions, axis=0)  # Średnie przewidywane zwroty
        mean_err = np.mean(errors, axis=0)  # Średnie błędy predykcyjne
        
        objective = cp.Maximize(
            self.k1 * psi + 
            self.k2 * (weights @ mean_pred) + 
            self.k3 * (weights @ mean_err) + 
            self.k4 * (weights @ R20) + 
            self.k5 * (weights @ R5)
        )
        
        # Dodajemy regularyzację L1 i L2 jeśli są zdefiniowane
        if self.l1_coef > 0:
            objective = cp.Maximize(
                self.k1 * psi + 
                self.k2 * (weights @ mean_pred) + 
                self.k3 * (weights @ mean_err) + 
                self.k4 * (weights @ R20) + 
                self.k5 * (weights @ R5) - 
                self.l1_coef * cp.norm(weights, 1)
            )
        
        if self.l2_coef > 0:
            objective = cp.Maximize(
                self.k1 * psi + 
                self.k2 * (weights @ mean_pred) + 
                self.k3 * (weights @ mean_err) + 
                self.k4 * (weights @ R20) + 
                self.k5 * (weights @ R5) - 
                self.l2_coef * cp.sum_squares(weights)
            )
        
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