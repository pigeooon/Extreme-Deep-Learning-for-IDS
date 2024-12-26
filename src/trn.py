# Extreme Deep Learning
import os
import numpy as np
import pandas as pd
import utility as ut

def train_sae_elm(config, X_train, y_train, n_hidden):
    _, n_features = X_train.shape

    best_error = np.inf
    best_weights = None

    r = np.sqrt(6 / (n_features + n_hidden))

    for i in range(config['sae_params']['num_runs']):
        w = np.random.uniform(-r, r, (n_features, n_hidden))
        h = ut.activation_function(X_train @ w, activation='sigmoid')
        h_pinv = ut.pseudo_inverse(h, config['sae_params']['penalty'])
        w_2 = h_pinv @ y_train
        w_r = h @ w_2
        error = np.mean((y_train - w_r) ** 2)

        if error < best_error:
            best_error = error
            best_weights = w
    
        print(f"Run: {i + 1}, Error: {error:.6f}")
    
    return best_weights

def train_softmax(config, X_train, y_train):
    n_features = X_train.shape[1]
    n_classes = y_train.shape[1]
    w = np.random.randn(n_features, n_classes) * config['softmax_params']['learning_rate']
    m, v = np.zeros_like(w), np.zeros_like(w)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    costs = []
    patience, min_delta, min_epochs = 25, 1e-8, 500
    best_cost, patience_counter = float('inf'), 0
    best_weights = None
    n_batches = X_train.shape[0] // config['softmax_params']['batch_size']

    for epoch in range(config['softmax_params']['max_iterations']):
        idx = np.random.permutation(X_train.shape[0])
        X_shuffled, Y_shuffled = X_train[idx], y_train[idx]
        epoch_cost = 0

        for i in range(n_batches):
            start_idx, end_idx = i * config['softmax_params']['batch_size'], (i + 1) * config['softmax_params']['batch_size']
            X_batch, y_batch = X_shuffled[start_idx:end_idx], Y_shuffled[start_idx:end_idx]
            
            logits = X_batch @ w
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            batch_cost = -np.mean(np.sum(y_batch * np.log(probs + epsilon), axis=1))
            epoch_cost += batch_cost
            
            grad = (1 / config['softmax_params']['batch_size']) * X_batch.T @ (probs - y_batch)
            
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            m_hat = m / (1 - beta1 ** (epoch + 1))
            v_hat = v / (1 - beta2 ** (epoch + 1))

            w -= config['softmax_params']['learning_rate'] * m_hat / (np.sqrt(v_hat) + epsilon)

        epoch_cost /= n_batches
        costs.append(epoch_cost)

        if epoch_cost <= best_cost:
            best_cost, best_weights = epoch_cost, w.copy()

        if epoch >= min_epochs:
            if epoch_cost < best_cost - min_delta:
                best_cost, patience_counter = epoch_cost, 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    return best_weights, np.array(costs)

def train_edl():
    # Cargar configuraciones
    config = ut.load_config_files()
    
    # Cargar datos de entrenamiento
    X_train, y_train = ut.load_data('./dataset/dtrain.csv', config['gain_indices'])
    
    # Normalizar datos
    mean, std = np.mean(X_train, axis=0), np.std(X_train, axis=0) + 1e-10
    
    X_train = (X_train - mean) / std
    
    # Guardar parámetros
    np.save('mean.npy', mean)
    np.save('std.npy', std)

    # Entrenamiento primera capa oculta SAE-ELM
    print("\nEntrenando primera capa oculta SAE-ELM...")
    w1 = train_sae_elm(config, X_train, y_train, config['sae_params']['hidden_nodes1'])
    h1 = ut.activation_function(X_train @ w1, activation='sigmoid')

    # Entrenamiento segunda capa oculta SAE-ELM
    print("\nEntrenando segunda capa oculta SAE-ELM...")
    w2 = train_sae_elm(config, h1, y_train, config['sae_params']['hidden_nodes2'])
    h2 = ut.activation_function(h1 @ w2, activation='sigmoid')    

    # Entrenamiento de la capa softmax
    print("\nEntrenando capa softmax...")
    w3, costs = train_softmax(config, h2, y_train)

    # Guardar pesos y costos
    costs_file = os.path.join('', 'costo.csv')
    np.savetxt(costs_file, costs, delimiter=',', fmt='%.6f')

    weights = [w1, w2, w3]
    for i, w in enumerate(weights, 1):
        weight_path = os.path.join('', f'w{i}.npy')
        np.save(weight_path, w)

    return weights, costs

def main():
    train_edl()
    print("\nTraining completed.")

if __name__ == '__main__':
    main()
