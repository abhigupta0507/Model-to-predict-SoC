##Code for Submission

import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, Input, Activation, Add, Concatenate, Dropout, Dense, GRU
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Custom attention layer for capturing rapid fluctuations
class SimpleAttention(Layer):
    def __init__(self):
        super(SimpleAttention, self).__init__()
        
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                shape=(input_shape[-1], 1),
                                initializer='random_normal',
                                trainable=True)
        
    def call(self, inputs):
        attention_weights = tf.nn.softmax(tf.matmul(inputs, self.W), axis=1)
        return tf.multiply(inputs, attention_weights)

# Enhanced TCN block with skip connections
def enhanced_tcn_block(input_layer, nb_filters, kernel_size, dilation_rate):
    # Shortcut connection
    shortcut = input_layer
    
    conv1 = Conv1D(filters=nb_filters, 
                   kernel_size=kernel_size, 
                   padding='causal', 
                   dilation_rate=dilation_rate,
                   kernel_regularizer=l2(0.0005))(input_layer)
    conv1 = LayerNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    
    if shortcut.shape[-1] != nb_filters:
        shortcut = Conv1D(nb_filters, 1, padding='same')(shortcut)
    
    output = Add()([conv1, shortcut])
    return output

# Model definition
def create_model(input_shape, l2_strength=0.0005):
    input_layer = Input(shape=input_shape)
    
    # TCN blocks with reduced filters
    tcn_blocks = []
    nb_filters = 24 
    
    for dilation_rate in [1, 2, 4]:
        tcn_block = enhanced_tcn_block(input_layer, 
                                     nb_filters=nb_filters, 
                                     kernel_size=3, 
                                     dilation_rate=dilation_rate)
        tcn_blocks.append(tcn_block)
    
    # Merge TCN blocks with attention
    merged = Add()(tcn_blocks)
    attention = SimpleAttention()(merged)
    
    # GRU with reduced units
    gru_layer = GRU(24,  
                    activation='tanh',
                    return_sequences=False,
                    kernel_regularizer=l2(l2_strength),
                    recurrent_regularizer=l2(l2_strength))(attention)
    
    # Efficient dense layers
    dense1 = Dense(12,  
                  activation='relu',
                  kernel_regularizer=l2(l2_strength))(gru_layer)
    
    # Output layer
    output_layer = Dense(1, 
                        activation='linear',
                        kernel_regularizer=l2(l2_strength))(dense1)
    
    return Model(inputs=input_layer, outputs=output_layer)

# Create and compile model
model = create_model((X_train_norm.shape[1], 1))
model.summary()
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
             loss='huber', 
             metrics=['mae', 'mape'])

# Enhanced callbacks
callbacks = [
    EarlyStopping(monitor='val_loss',
                  patience=20,
                  restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss',
                      factor=0.2,
                      patience=5,
                      min_lr=0.00001)
]

# Train with smaller batch size
history = model.fit(X_train_norm, y_train,
                   epochs=50,
                   batch_size=32, 
                   validation_data=(X_val_norm, y_val),
                   callbacks=callbacks,
                   verbose=1)

# Evaluate and predict
y_pred = model.predict(X_test_norm)
regression_metrics(y_test, y_pred)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual', alpha=0.7)
plt.plot(y_pred, label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
