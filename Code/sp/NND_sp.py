from sklearn.neural_network import MLPClassifier
import numpy as np
#import logging
from roy import royinv

#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)

def average_discriminators(models, fake_samples):


    '''
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
    '''

def NND_train(true_samples, fake_samples, num_hidden=10):
    #print("Training NND")
    #Set random seed of NN training
    input_data = np.column_stack((true_samples, fake_samples)).T
    nplusm = len(input_data)
    labels = np.column_stack((np.ones_like(true_samples), np.zeros_like(fake_samples)))[0,:].T
    model = MLPClassifier(hidden_layer_sizes=(num_hidden,),
                          activation='tanh', #paper
                          solver='adam',
                          alpha=0.01, #net.performParam.regularization = 0.01; cannot be easily translated
                          #batch_size=nplusm, #not needed with lbfgs
                          max_iter=2000)#, learning_rate_init=0.001, early_stopping=True, validation_fraction=0.1)
    model.fit(input_data, labels)
    if not (model.classes_[0] == 0 and model.classes_[1] == 1):
        print("Warning: classes are not [0, 1], switching them")
        model.classes_[:, [0, 1]] = model.classes_[:, [1, 0]]
    #print("NND trained")
    '''
    # Check convergence based on loss improvement
    if hasattr(model, 'loss_curve_'):
        converged = len(model.loss_curve_) < model.max_iter
    else:
        converged = model.n_iter_ < model.max_iter
    
    logger.info(f"Model converged: {converged}")
    logger.info(f"Number of iterations: {model.n_iter_}")
    '''
    return model

def generator_loss(true_samples, fake_samples, num_hidden=10, num_models=1):
    models = [NND_train(true_samples, fake_samples, num_hidden) for _ in range(num_models)]
    fakearray = np.asarray(fake_samples).T
    truearray = np.asarray(true_samples).T
    average_prediction_fake = np.mean([model.predict_proba(fakearray)[:,1] for model in models], axis=0)
    average_prediction_true = np.mean([model.predict_proba(truearray)[:,1] for model in models], axis=0)
    generator_loss = np.mean(np.log(average_prediction_true)) + np.mean(np.log(1 - average_prediction_fake))
    #discriminator_loss = np.mean(np.log(avg_model.predict_proba(true_samples))) + np.mean(np.log(1 - avg_model.predict_proba(fake_samples)))
    #print(f"Discriminator loss: {discriminator_loss}")
    #print(f"Generator loss: {generator_loss}")
    return generator_loss

"""
# Test

u_1 = np.random.rand(100, 4)
u_2 = np.random.rand(100, 4)
theta_1 = np.array([1.8, 2, 0.5, 0, 1, 1, 0.5])
theta_2 = np.array([1.9, 2.1, 0.6, 0.1, 1.1, 1.1, 0.6])
X_1 = royinv(u_1, theta_1)
X_2 = royinv(u_2, theta_2)
print(generator_loss(X_1, X_2, num_models=1))
"""