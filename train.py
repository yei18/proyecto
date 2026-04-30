import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from model import AnomalyDetector

def train_model(train_loader, test_loader, input_dim, epochs=5, lr=1e-3, d_model=32, d_state=8, top_k=5):
    """Loop de entrenamiento con validación y guardado del mejor checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    model = AnomalyDetector(input_dim=input_dim, d_model=d_model, d_state=d_state, top_k=top_k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        all_preds, all_labels = [], []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits[:, -1, :], y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits[:, -1, :], dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())
            
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                preds = torch.argmax(logits[:, -1, :], dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(y_batch.numpy())
                
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'checkpoint.pt')
            print("  --> Checkpoint guardado")

    print("Entrenamiento completado.")
    return model