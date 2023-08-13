# WineMLproject
ML Project for final assignment on AI/ML course

Hyperparameter Tuning and Neural Network Model for Classification

This project involves hyperparameter tuning and training a feedforward neural network model for classification tasks. The code uses Hyperopt for tuning hyperparameters and Keras to build and train the model.

1. Define the Hyperparameter Space

First, we define the hyperparameter space, specifying the possible values for each hyperparameter.

space = {
    'n_layers': hp.choice('n_layers', [1, 2]),
    'n_units_layer_0': hp.choice('n_units_layer_0', [32, 64, 128]),
    'epochs': hp.choice('epochs', [50, 100]),
    'batch_size': hp.choice('batch_size', [16, 32, 64])
}

In the space definition, n_layers represents the number of hidden layers, and n_units_layer_0 indicates the number of units in the first hidden layer. The epochs and batch_size control the training process.

2. Build the Neural Network Model

We create a function build_model to construct the neural network model based on the hyperparameters.

def build_model(params, input_shape):
    model = Sequential()
    model.add(Dense(params['n_units_layer_0'], activation='relu', input_shape=input_shape))
    if 'n_units_layer_1' in params:
        model.add(Dense(params['n_units_layer_1'], activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

This function creates a sequential model with one or two hidden layers and a final output layer with a softmax activation function for classification.

3. Define the Objective Function

We then define the objective function for Hyperopt to minimize, including building and training the model.

def objective(params):
    if params['n_layers'] == 2:
        params['n_units_layer_1'] = hp.choice('n_units_layer_1', [32, 64, 128])
    model = build_model(params, input_shape=(X_train_scaled.shape[1],))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=params['epochs'], batch_size=params['batch_size'], callbacks=[early_stop], verbose=0)
    loss = history.history['val_loss'][-1]
    return loss

The objective function first checks the number of layers, builds the model using the build_model function, applies early stopping, and fits the model to the training data. The function returns the validation loss.

4. Hyperparameter Optimization

We utilize the fmin function from Hyperopt to find the optimal hyperparameters.

best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=30, verbose=2)

The fmin function takes the objective function and hyperparameter space and returns the best hyperparameters that minimize the objective function.

By using Hyperopt to optimize hyperparameters and Keras to build and train a neural network model, this code provides a flexible solution for classification tasks.


Analysis of Biases, Methods, and Results
Biases in Data

It is important to recognize that the quality of a machine learning model's predictions largely depends on the quality of the data. If there are biases present in the data collection process, the model will likely learn these biases. For example, if certain classes are underrepresented in the training data, the model might be biased towards predicting the overrepresented classes. In this specific project, without further details about the data collection and preprocessing, it's challenging to identify specific biases. However, understanding the source and nature of the data would be a crucial step in identifying and mitigating any potential biases.

Methods

The project utilized a feedforward neural network model with hyperparameter tuning. The hyperparameters, including the number of hidden layers, units in each layer, batch size, and number of epochs, were optimized using Hyperopt. The neural network was trained using a combination of the ReLU activation function for hidden layers and the softmax activation function for the output layer. Early stopping was applied to prevent overfitting.

Results

The best hyperparameters found were:

Batch size: 2
Epochs: 2
Number of hidden layers: 1
Units in the first hidden layer: 2
Units in the second hidden layer (if applicable): 2
The model was then trained and evaluated, leading to the following results:

Training loss: 0.8708
Training accuracy: 56.64%
Validation loss: 0.8246
Validation accuracy: 65.52%
Test accuracy: 50.00%
Interpretation and Potential Concerns

The results show a significant discrepancy between the validation accuracy (65.52%) and the test accuracy (50.00%). This might indicate an issue with the model's generalization to unseen data or might reflect the biases in the data distribution between training, validation, and test sets.

The relatively low accuracy on the test set also suggests that the model might not be capturing the underlying patterns in the data effectively. The small number of epochs (2) and hidden units (2) might be too restrictive, leading to an underfitting problem.

Further investigation would be needed to understand the underlying cause of these results. Potential areas to explore might include the data distribution, model architecture, and the range of hyperparameters considered in the optimization process. The addition of more informative features, data preprocessing, or trying different machine learning models could also lead to improved results.
