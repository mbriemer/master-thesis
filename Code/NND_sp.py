import numpy as np
from sklearn.neural_network import MLPClassifier

def average_discriminators(models):
    # Initialize arrays to accumulate the weights and biases
    avg_coefs = [np.zeros_like(coef) for coef in models[0].coefs_]
    avg_intercepts = [np.zeros_like(intercept) for intercept in models[0].intercepts_]

    # Sum up the parameters from each model
    for model in models:
        for i, coef in enumerate(model.coefs_):
            avg_coefs[i] += coef
        for i, intercept in enumerate(model.intercepts_):
            avg_intercepts[i] += intercept

    # Divide by the number of models to get the average
    avg_coefs = [coef / len(models) for coef in avg_coefs]
    avg_intercepts = [intercept / len(models) for intercept in avg_intercepts]

    # Create a new model with the same architecture
    new_model = MLPClassifier(hidden_layer_sizes=models[0].hidden_layer_sizes)

    # Assign the averaged parameters to the new model
    new_model.coefs_ = avg_coefs
    new_model.intercepts_ = avg_intercepts

    return new_model

def NDD_train(true_samples, fake_samples, num_hidden=10):
    input = np.concatenate((true_samples, fake_samples))
    labels = np.concatenate((np.ones_like(true_samples), np.zeros_like(fake_samples)))
    model = MLPClassifier(hidden_layer_sizes=(num_hidden,))
    model.fit(input, labels)
    return model

def NDD_loss(true_samples, fake_samples, num_hidden=10, num_models=1):
    models = [NDD_train(true_samples, fake_samples, num_hidden) for _ in range(num_models)]
    avg_model = average_discriminators(models)
    return avg_model.score(np.concatenate((true_samples, fake_samples)), np.concatenate((np.ones_like(true_samples), np.zeros_like(fake_samples))))  