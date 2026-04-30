import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from model import AnomalyDetector
from kalman import KalmanSmoother

def evaluate_model(model, test_loader, input_dim, d_model=32, d_state=8, top_k=5):
    """Evalúa el modelo baseline y compara con 3 configuraciones del Filtro de Kalman."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('checkpoint.pt'))
    model.eval()

    raw_probs = []
    true_labels = []

    print("Extrayendo probabilidades del conjunto de Test...")
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            probs = model.predict_proba(X_batch)
            raw_probs.extend(probs.cpu().numpy())
            true_labels.extend(y_batch.numpy())

    raw_probs = np.array(raw_probs)
    true_labels = np.array(true_labels)

    # Baseline
    baseline_preds = (raw_probs >= 0.5).astype(int)
    b_acc = accuracy_score(true_labels, baseline_preds)
    b_rec = recall_score(true_labels, baseline_preds, average='macro')
    b_f1 = f1_score(true_labels, baseline_preds, average='macro')
    b_mae = mean_absolute_error(true_labels, raw_probs)
    b_mse = mean_squared_error(true_labels, raw_probs)

    print(f"Baseline  | Acc: {b_acc:.4f} | Rec: {b_rec:.4f} | F1: {b_f1:.4f} | MAE: {b_mae:.4f} | MSE: {b_mse:.4f}")

    # Configuraciones de Kalman
    configs = [
        (1e-5, 1e-1), # Baja varianza de proceso (Suavizado agresivo)
        (1e-3, 1e-2), # Equilibrado
        (1e-1, 1e-3)  # Alta varianza de proceso (Seguimiento rápido)
    ]

    for (sw, sv) in configs:
        ks = KalmanSmoother(sigma_w2=sw, sigma_v2=sv)
        smoothed_probs = ks.smooth_sequence(raw_probs)
        smoothed_probs = np.clip(smoothed_probs, 0, 1) # Acotar a [0,1]
        smoothed_preds = (smoothed_probs >= 0.5).astype(int)

        acc = accuracy_score(true_labels, smoothed_preds)
        rec = recall_score(true_labels, smoothed_preds, average='macro')
        f1 = f1_score(true_labels, smoothed_preds, average='macro')
        mae = mean_absolute_error(true_labels, smoothed_probs)
        mse = mean_squared_error(true_labels, smoothed_probs)
        
        print(f"Kalman sw={sw}, sv={sv} | Acc: {acc:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | MAE: {mae:.4f} | MSE: {mse:.4f}")