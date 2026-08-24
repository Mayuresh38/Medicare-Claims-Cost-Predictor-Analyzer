import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

def build_regression_model():
    """
    Returns a Ridge Regression model.
    """
    return Ridge(alpha=1.0)

def build_ann_model(input_dim, learning_rate=0.001, dropout_rate=0.2, units_1=128, units_2=64, units_3=32):
    """
    Creates a Feedforward Artificial Neural Network (ANN) in Keras.
    Accepts hyperparameters for tuning.
    """
    inputs = layers.Input(shape=(input_dim,), name="tabular_inputs")
    x = layers.Dense(units_1, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(units_2, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(units_3, activation="relu")(x)
    
    outputs = layers.Dense(1, activation="linear", name="cost_prediction")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="ANN_Cost_Predictor")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_absolute_error",
        metrics=["mean_squared_error"]
    )
    return model

if __name__ == "__main__":
    # Test architectures
    ann = build_ann_model(20)
    ann.summary()
