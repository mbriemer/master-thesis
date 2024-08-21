from sklearn.neural_network import MLPClassifier
import numpy as np

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
    
    # Set additional required attributes
    new_model.n_layers_ = models[0].n_layers_
    new_model.n_outputs_ = models[0].n_outputs_
    new_model.n_iter_ = max(model.n_iter_ for model in models)
    new_model.out_activation_ = models[0].out_activation_
    new_model._label_binarizer = models[0]._label_binarizer
    new_model.classes_ = models[0].classes_
    new_model.n_features_in_ = models[0].n_features_in_
    new_model.n_iter_ = max(model.n_iter_ for model in models)
    new_model.loss_ = models[0].loss_
    new_model._no_improvement_count = 0
    new_model.best_loss_ = min(model.best_loss_ for model in models)
    new_model.loss_curve_ = models[0].loss_curve_  # You might want to average this across models
    new_model.t_ = max(model.t_ for model in models)
    new_model.n_outputs_ = models[0].n_outputs_
    new_model._random_state = models[0]._random_state
    
    # Mark the model as fitted
    new_model._fitted = True
    
    return new_model

def NND_train(true_samples, fake_samples, num_hidden=10):
    input_data = np.concatenate((true_samples, fake_samples))
    labels = np.concatenate((np.ones(len(true_samples)), np.zeros(len(fake_samples))))
    model = MLPClassifier(hidden_layer_sizes=(num_hidden,), max_iter=1000)
    model.fit(input_data, labels)
    #print("NN trained!")
    return model

def generator_loss(true_samples, fake_samples, num_hidden=10, num_models=1):
    models = [NND_train(true_samples, fake_samples, num_hidden) for _ in range(num_models)]
    avg_model = average_discriminators(models)
    #print(f"Average model predictions: {avg_model.predict_proba(fake_samples)}")
    generator_loss = np.mean(np.log(avg_model.predict_proba(fake_samples)))
    discriminator_loss = np.mean(np.log(avg_model.predict_proba(true_samples))) + np.mean(np.log(1 - avg_model.predict_proba(fake_samples)))
    print(f"Discriminator loss: {discriminator_loss}")
    print(f"Generator loss: {generator_loss}")
    return generator_loss