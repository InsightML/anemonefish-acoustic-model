"""
Hypermodels for Keras Tuner
"""

import tensorflow as tf
import keras_tuner
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam


class TargetToNoiseHyperModel(keras_tuner.HyperModel):
    """
    HyperModel for target-to-noise classification.
    
    This class encapsulates the model architecture and hyperparameter search space
    for multi-class classification of spectrograms (noise, anemonefish, biological).
    
    Parameters
    ----------
    input_shape : tuple
        Shape of input spectrograms (height, width, channels), e.g., (256, 256, 3)
    num_classes : int
        Number of output classes (e.g., 3 for noise/anemonefish/biological)
    
    Examples
    --------
    >>> hypermodel = TargetToNoiseHyperModel(
    ...     input_shape=(256, 256, 3),
    ...     num_classes=3
    ... )
    >>> tuner = keras_tuner.RandomSearch(
    ...     hypermodel=hypermodel,
    ...     objective='val_loss',
    ...     max_trials=10
    ... )
    """

    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_cnn_blocks = 4
        self.filters = [32, 64, 128, 256]
    
    def build(self, hp):
        """
        Build the model with hyperparameters.
        
        This method is called by Keras Tuner during hyperparameter search.
        
        Parameters
        ----------
        hp : keras_tuner.HyperParameters
            Hyperparameter object from Keras Tuner
            
        Returns
        -------
        keras.Model
            Compiled model ready for training
        """
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        
        # Tunable activation function
        activation = hp.Choice('activation', ['relu', 'sigmoid'])
        
        # Tunable CNN depth per block
        cnn_depth = hp.Int('cnn_depth', min_value=1, max_value=6)
        
        # Build CNN blocks
        for cnn_block in range(self.num_cnn_blocks):
            for i in range(cnn_depth):
                model.add(Conv2D(
                    self.filters[cnn_block], 
                    (3, 3), 
                    padding='same', 
                    activation=activation
                ))
            
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Dense layers
        model.add(Flatten())
        model.add(Dense(128, activation=activation))
        model.add(BatchNormalization())
        
        # Optional dropout
        if hp.Boolean('use_dropout'):
            dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compile with tunable learning rate
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_fixed(self, cnn_depth=2, activation='relu', use_dropout=True, 
                    dropout_rate=0.3, learning_rate=0.001):
        """
        Build model with fixed hyperparameters (no tuning).
        
        Useful for training with known optimal hyperparameters or quick testing.
        
        Parameters
        ----------
        cnn_depth : int, default=2
            Number of convolutional layers per block
        activation : str, default='relu'
            Activation function ('relu', 'sigmoid', etc.)
        use_dropout : bool, default=True
            Whether to use dropout regularization
        dropout_rate : float, default=0.3
            Dropout rate if use_dropout is True
        learning_rate : float, default=0.001
            Learning rate for Adam optimizer
            
        Returns
        -------
        keras.Model
            Compiled model ready for training
        """
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        
        # Build CNN blocks
        for cnn_block in range(self.num_cnn_blocks):
            for i in range(cnn_depth):
                model.add(Conv2D(
                    self.filters[cnn_block], 
                    (3, 3), 
                    padding='same', 
                    activation=activation
                ))
            
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Dense layers
        model.add(Flatten())
        model.add(Dense(128, activation=activation))
        model.add(BatchNormalization())
        
        if use_dropout:
            model.add(Dropout(dropout_rate))
        
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compile
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model