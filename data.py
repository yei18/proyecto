import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import torch
from torch.utils.data import DataLoader, TensorDataset

def create_windows(X, y, window_size=10):
    """Aplica ventaneo deslizante a los datos."""
    X_w, y_w = [], []
    for i in range(len(X) - window_size):
        X_w.append(X[i:i+window_size])
        y_w.append(y[i+window_size-1])
    return np.array(X_w), np.array(y_w)

def get_dataloaders(window_size=10, batch_size=32):
    """Carga UNSW-NB15, preprocesa, balancea y retorna DataLoaders."""
    print("Descargando/Cargando dataset...")
    file_path_train = "UNSW_NB15_training-set.csv"
    file_path_test = "UNSW_NB15_testing-set.csv"

    df_train = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "mrwellsdavid/unsw-nb15", file_path_train)
    df_test = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "mrwellsdavid/unsw-nb15", file_path_test)

    cat_cols = ['proto', 'service', 'state']
    target_col = 'label'
    id_cols = ['id']

    drop_cols = [c for c in df_train.columns if c not in cat_cols + [target_col] and df_train[c].dtype == 'object']
    df_train = df_train.drop(columns=drop_cols + id_cols, errors='ignore')
    df_test = df_test.drop(columns=drop_cols + id_cols, errors='ignore')

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])
        df_test[col] = df_test[col].apply(lambda x: x if x in le.classes_ else -1)
        df_test[col] = le.transform(df_test[col].replace({-1: le.classes_[0]}))
        encoders[col] = le

    X_train = df_train.drop(columns=[target_col]).values.astype(np.float32)
    y_train = df_train[target_col].values.astype(np.int64)
    X_test = df_test.drop(columns=[target_col]).values.astype(np.float32)
    y_test = df_test[target_col].values.astype(np.int64)

    # Normalización ajustada solo en Train
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_w, y_train_w = create_windows(X_train, y_train, window_size)
    X_test_w, y_test_w = create_windows(X_test, y_test, window_size)

    n_samples, n_timesteps, n_features = X_train_w.shape
    X_train_flat = X_train_w.reshape(n_samples, n_timesteps * n_features)

    print("Aplicando balanceo de clases (SMOTE + RUS)...")
    resampler = ImbPipeline([
        ('oversample', SMOTE(sampling_strategy=0.5, random_state=42)),
        ('undersample', RandomUnderSampler(sampling_strategy=1.0, random_state=42))
    ])

    X_res_flat, y_res = resampler.fit_resample(X_train_flat, y_train_w)
    X_res = X_res_flat.reshape(-1, n_timesteps, n_features)

    train_dataset = TensorDataset(torch.tensor(X_res), torch.tensor(y_res))
    test_dataset = TensorDataset(torch.tensor(X_test_w), torch.tensor(y_test_w))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, n_features