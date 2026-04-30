import numpy as np

class KalmanSmoother:
    """Filtro de Kalman escalar para suavizado de probabilidades de anomalía."""
    def __init__(self, sigma_w2=1e-4, sigma_v2=1e-2):
        self.sigma_w2 = sigma_w2
        self.sigma_v2 = sigma_v2
        self.s_hat = 0.0
        self.P = 1.0
        
    def update(self, p_t):
        """Ejecuta un paso de predicción + actualización."""
        # Predicción
        s_pred = self.s_hat
        P_pred = self.P + self.sigma_w2
        
        # Ganancia
        K_t = P_pred / (P_pred + self.sigma_v2)
        
        # Actualización
        self.s_hat = s_pred + K_t * (p_t - s_pred)
        self.P = (1 - K_t) * P_pred
        
        return self.s_hat
    
    def smooth_sequence(self, probs):
        """Aplica el filtro a una secuencia completa de probabilidades."""
        smoothed = []
        self.s_hat = 0.0
        self.P = 1.0
        for p in probs:
            s = self.update(p)
            smoothed.append(s)
        return np.array(smoothed)