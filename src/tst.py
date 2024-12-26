import numpy as np
import utility as ut

def forward_pass(X_test, weights, mean, std):
    # Normalize test data
    X_test_norm = (X_test - mean) / std

    # Forward pass through each layer
    w1, w2, w3 = weights
    H1 = ut.activation_function(X_test_norm @ w1, activation='sigmoid')
    H2 = ut.activation_function(H1 @ w2, activation='sigmoid')
    logits = H2 @ w3

    # Numerical stability adjustment for Softmax
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    predictions = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return predictions

def calculate_metrics(y_test, predictions):
    conf_matrix, f_scores = ut.mtx_confusion(y_test, predictions)

    # Overall accuracy
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

    # Class-specific metrics
    class_metrics = []
    for i in range(2):
        precision = conf_matrix[i, i] / np.sum(conf_matrix[:, i]) 
        recall = conf_matrix[i, i] / np.sum(conf_matrix[i, :]) 
        f1 = f_scores[i]
        class_metrics.append({'precision': precision, 'recall': recall, 'f1': f1})

    return conf_matrix, class_metrics, accuracy

def forward_edl():
    # Load test data
    print("Loading test data...")

    # Cargar configuraciones
    config = ut.load_config_files()
    
    # Cargar datos de entrenamiento
    X_test, y_test = ut.load_data('./dataset/dtrain.csv', config['gain_indices'])

    # Load trained weights and normalization parameters
    print("Loading weights and normalization parameters...")
    weights = []
    for i in range(1, 4):
        w = np.load(f'w{i}.npy')
        weights.append(w)

    w1, w2, w3 = weights

    # Load normalization parameters
    mean = np.load('mean.npy')
    std = np.load('std.npy')

    # Normalize test data using training statistics
    print("Normalizing test data...")
    X_test_norm = (X_test - mean) / std

    print("Performing forward pass...")
    # First SAE layer forward pass
    H1 = ut.activation_function(X_test_norm @ w1, activation='sigmoid')

    # Second SAE layer forward pass
    H2 = ut.activation_function(H1 @ w2, activation='sigmoid')

    # Softmax layer forward pass with numerical stability
    logits = H2 @ w3
    # Subtract max for numerical stability
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    predictions = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Calculate confusion matrix and F-scores
    print("Calculating metrics...")
    conf_matrix, f_scores = ut.mtx_confusion(y_test, predictions)

    # Calculate and print additional metrics
    print("\nPerformance Metrics:")
    # Calculate accuracy
    accuracy = (conf_matrix[0, 0] +
                conf_matrix[1, 1]) / np.sum(conf_matrix)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Calculate metrics for each class
    for i, class_name in enumerate(['Normal', 'Attack']):
        precision = conf_matrix[i, i] / np.sum(conf_matrix[:, i])
        recall = conf_matrix[i, i] / np.sum(conf_matrix[i, :])
        f1 = f_scores[i]
        print(f"\n{class_name} Class Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

    # Save confusion matrix and F-scores
    np.savetxt('confusion.csv', conf_matrix, fmt='%d') 
    np.savetxt('fscores.csv', f_scores, fmt='%.4f')

    return predictions, conf_matrix, f_scores

def main():
    print("Starting EDL testing...")
    predictions, conf_matrix, f_scores = forward_edl()
    if predictions is not None:
        print("\nTesting completed successfully")

        # Print confusion matrix in a more readable format
        print("\nConfusion Matrix:")
        print("                 Predicted Normal  Predicted Attack")
        print(f"Actual Normal    {conf_matrix[0, 0]:^15.0f} {conf_matrix[0, 1]:^16.0f}")
        print(f"Actual Attack    {conf_matrix[1, 0]:^15.0f} {conf_matrix[1, 1]:^16.0f}")
    else:
        print("Testing failed")


if __name__ == '__main__':
    main()