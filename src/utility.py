# My Utility: auxiliary functions
import numpy as np
import pandas as pd

def load_data(file_path, gain_indices=None):
    # Cargar datos
    data = pd.read_csv(file_path, header=None)
    print(f"Datos cargados: {data.shape[0]} filas, {data.shape[1]} columnas")  # Validar cantidad de datos cargados
    
    # Separar características y etiquetas
    X = data.iloc[:, :-1].values  # Todas las columnas excepto la última
    y = data.iloc[:, -1].values  # Última columna como etiquetas

    X_data = X[:, gain_indices] if gain_indices is not None else X
    
    y_binary = np.zeros((len(y), 2))
    y_binary[y == 1, 0] = 1  # Class 1: [1, 0]
    y_binary[y == 2, 1] = 1  # Class 2: [0, 1]

    return X_data, y_binary

def load_config_files():
    """Load configuration files and parameters"""
    # Leer configuración SAE
    sae_config = pd.read_csv('./dataset/config_sae.csv', header=None)
    sae_params = sae_config[0].values  # Leer como una lista de valores en una columna
    
    if len(sae_params) < 4:
        raise ValueError("El archivo config_sae.csv debe contener al menos 4 filas para los parámetros.")
    
    # Leer configuración Softmax
    softmax_config = pd.read_csv('./dataset/config_softmax.csv', header=None)
    softmax_params = softmax_config[0].values  # Leer como una lista de valores en una columna
    
    if len(softmax_params) < 3:
        raise ValueError("El archivo config_softmax.csv debe contener al menos 3 filas para los parámetros.")
    
    # Leer índices de ganancia
    idx_gain = pd.read_csv('./dataset/idx_igain.csv', header=None)
    gain_indices = idx_gain[0].values  # Leer como una lista de valores en la primera columna
    
    if len(gain_indices) == 0:
        raise ValueError("El archivo idx_igain.csv no puede estar vacío.")

    return {
        'sae_params': {
            'hidden_nodes1': int(sae_params[0]),
            'hidden_nodes2': int(sae_params[1]),
            'penalty': int(sae_params[2]),
            'num_runs': int(sae_params[3])
        },
        'softmax_params': {
            'max_iterations': int(softmax_params[0]),
            'batch_size': int(softmax_params[1]),
            'learning_rate': softmax_params[2]
        },
        'gain_indices': gain_indices - 1  # Convertir a índices base 0
    }

def activation_function(x, activation='sigmoid', alpha=1.0, scale=1.67326324):
    """Apply the selected activation function"""
    if activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif activation == 'tanh':
        return np.tanh(x)
    elif activation == 'relu':
        return np.maximum(0, x)
    elif activation == 'elu':
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    elif activation == 'selu':
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
    else:
        raise ValueError("Activation function not supported")

def pseudo_inverse(H, C):
    HHT = H @ H.T
    I = np.eye(HHT.shape[0])
    A = HHT + I / C
    A_inv = np.linalg.inv(A)  
    return H.T @ A_inv

def mtx_confusion(y_true, y_pred):
    # Get predicted classes
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true_class = np.argmax(y_true, axis=1)

    # Initialize confusion matrix
    cm = np.zeros((2, 2))
    for t, p in zip(y_true_class, y_pred_class):
        cm[t, p] += 1

    # Calculate metrics for each class
    f_scores = np.zeros(2)
    for i in range(2):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP

        # Calculate precision and recall
        precision = TP/(TP + FP) if (TP + FP) != 0 else 0
        recall = TP/(TP + FN) if (TP + FN) != 0 else 0

        # Calculate F-score
        f_scores[i] = 2 * (precision * recall)/(precision +
                                                recall) if (precision + recall) != 0 else 0

    return cm, f_scores