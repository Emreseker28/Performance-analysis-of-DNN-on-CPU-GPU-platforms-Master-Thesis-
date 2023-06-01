#%%
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch

train_data = pd.read_csv('C:\\Users\\Emre\\Desktop\\Projects\\cavity\\cavity1.0\\0.1\\p')
train_data = train_data.iloc[19:419, :]
print(train_data)
test_data = pd.read_csv('C:\\Users\\Emre\\Desktop\\Projects\\cavity\\cavity4.0\\0.1\\p')
test_data = test_data.iloc[19:419, :]

#%%
# Define the CNN model
def CNN():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(400, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

#%%
def preprocess_data(data):
    # Normalize the data
    normalized_data = (data - np.mean(data)) / np.std(data)
    
    # Reshape the data to match the input shape of the CNN
    train_data_np = normalized_data.values.astype(np.float32)  # Convert DataFrame to NumPy array with float32 data type
    
    return train_data_np

preprocessed_train_data = preprocess_data(train_data)

# Create the CNN model
model = CNN()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(preprocessed_train_data, test_data, epochs=10, batch_size=32)

#history = model.fit(preprocessed_train_data, train_labels, epochs=10, batch_size=32)
epochs = 30
steps = 150
print("Train")
for epoche in range(epochs):
    err = 0.
    for step in range(steps):
        inputs, labels = next(iter(train_data))
        outputs = model(inputs)


#%%
# Plot the training loss
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot the training accuracy
plt.plot(history.history['accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

#%%

print(test_data)
# Preprocess the test data
preprocessed_test_data = preprocess_data(test_data)

# Make predictions
predictions = model.predict(preprocessed_test_data)

# Print the predictions
print(predictions)


# %%
