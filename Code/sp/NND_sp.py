from sklearn.neural_network import MLPClassifier
import numpy as np
from roy import royinv

def NND_train(true_samples, fake_samples, num_hidden=10):
    input_data = np.column_stack((true_samples, fake_samples)).T
    nplusm = len(input_data)
    labels = np.column_stack((np.ones_like(true_samples), np.zeros_like(fake_samples)))[0,:].T
    model = MLPClassifier(hidden_layer_sizes=(num_hidden,),
                          activation='tanh', #paper; output activation is very likely set to logistic internally
                          solver='adam',
                          tol=1e-4, #default: 1e-4
                          verbose=False)#,
                          #alpha=0.01, #net.performParam.regularization = 0.01; cannot be easily translated
                          #batch_size=nplusm, #not needed with lbfgs
                          #max_iter=2000)#, learning_rate_init=0.001, early_stopping=True, validation_fraction=0.1)
    model.fit(input_data, labels)
    if not (model.classes_[0] == 0 and model.classes_[1] == 1):
        print("Warning: classes are not [0, 1], switching them")
        model.classes_[:, [0, 1]] = model.classes_[:, [1, 0]]
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