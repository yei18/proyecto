import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    """Módulo de Estado Espacio Selectivo con parámetros dependientes de la entrada."""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.log_A = nn.Parameter(torch.log(torch.ones(d_state) * 0.5))
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.D_param = nn.Parameter(torch.ones(d_model))
        self.dt_proj = nn.Linear(d_model, 1)
        
    def forward(self, u):
        B, L, D = u.shape
        N = self.d_state
        
        A = -torch.exp(self.log_A)
        dt = F.softplus(self.dt_proj(u))
        dA = 1 + A * dt
        
        B_t = self.B_proj(u)
        C_t = self.C_proj(u)
        
        x = torch.zeros(B, N, device=u.device)
        ys = []
        
        for t in range(L):
            x = dA[:, t, :] * x + B_t[:, t, :] * dt[:, t, 0].unsqueeze(1)
            y = torch.einsum('bn,bn->b', x, C_t[:, t, :])
            ys.append(y)
            
        y = torch.stack(ys, dim=1)
        return y.unsqueeze(-1) + self.D_param * u

class TemporalBlock(nn.Module):
    """Bloque temporal con convolución local, SSM selectivo, gating y conexión residual."""
    def __init__(self, d_model, d_state=16, kernel_size=3):
        super().__init__()
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size-1, groups=d_model)
        self.ssm = SelectiveSSM(d_model, d_state)
        self.norm = nn.LayerNorm(d_model)
        self.silu = nn.SiLU()
        self.proj_in = nn.Linear(d_model, d_model * 2)
        
    def forward(self, x):
        residual = x
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)[:, :x.size(1), :]
        x = self.silu(x)
        x_ssm = self.ssm(x)
        x_gate = self.proj_in(x_ssm)
        x1, x2 = x_gate.chunk(2, dim=-1)
        x = x1 * torch.sigmoid(x2)
        return self.norm(x + residual)

class SpectralBlock(nn.Module):
    """Extractor de features frecuenciales mediante DFT, filtro complejo y DFT inversa."""
    def __init__(self, d_model, top_k=8):
        super().__init__()
        self.top_k = top_k
        self.complex_filter = nn.Parameter(torch.ones(d_model, dtype=torch.cfloat))
        
    def forward(self, x):
        B, L, D = x.shape
        X_f = torch.fft.rfft(x, dim=1)
        magnitudes = torch.abs(X_f)
        _, topk_idx = torch.topk(magnitudes, self.top_k, dim=1)
        mask = torch.zeros_like(magnitudes, dtype=torch.bool)
        mask.scatter_(1, topk_idx, True)
        X_f_filtered = X_f * mask * self.complex_filter.unsqueeze(0).unsqueeze(0)
        x_out = torch.fft.irfft(X_f_filtered, n=L, dim=1)
        return x_out

class Fusion(nn.Module):
    """Fusión ponderada elemento a elemento con coeficientes aprendibles alfa y beta."""
    def __init__(self, d_model):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x_temp, x_freq):
        return self.alpha * x_temp + self.beta * x_freq

class AnomalyDetector(nn.Module):
    """Arquitectura completa ensamblada para detección de anomalías."""
    def __init__(self, input_dim, d_model=64, d_state=16, top_k=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.temporal_block = TemporalBlock(d_model, d_state)
        self.spectral_block = SpectralBlock(d_model, top_k)
        self.fusion = Fusion(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 2)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x_temp = self.temporal_block(x)
        x_freq = self.spectral_block(x)
        z = self.fusion(x_temp, x_freq)
        logits = self.classifier(z)
        return logits
    
    def predict_proba(self, x):
        """Retorna la probabilidad de anomalía para una ventana dada."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            prob = torch.softmax(logits, dim=-1)
            return prob[..., 1]
    
    @classmethod
    def load(cls, path, input_dim, d_model=64, d_state=16, top_k=8):
        """Carga un modelo desde un checkpoint."""
        model = cls(input_dim, d_model, d_state, top_k)
        model.load_state_dict(torch.load(path))
        return model