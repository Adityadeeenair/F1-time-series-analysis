import tensorflow as tf
tf.config.list_physical_devices('GPU')

import fastf1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import os
os.makedirs('/content/f1_cache', exist_ok=True)

fastf1.Cache.enable_cache('/content/f1_cache')
print("FastF1 cache enabled ✅")

session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load()

laps = session.laps
laps.head()

ver_laps = laps[laps['Driver'] == 'VER']
ver_laps.head()

ham_laps = laps[laps['Driver'] == 'HAM'].copy()
ham_laps.head()

ver_laps = ver_laps[['LapNumber', 'LapTime']]
ver_laps.head()

ver_laps['LapTimeSeconds'] = ver_laps['LapTime'].dt.total_seconds()
ver_laps.head()

ham_laps['LapTimeSeconds'] = ham_laps['LapTime'].dt.total_seconds()
ham_laps.head()

#Graph Demonstration :

plt.figure(figsize=(10, 5))
plt.plot(ver_laps['LapNumber'], ver_laps['LapTimeSeconds'], marker='o')
plt.xlabel('Lap Number')
plt.ylabel('Lap Time (seconds)')
plt.title('Lap Times (Bahrain GP 2023)')
plt.grid(True)
plt.show()

# Remove laps where lap time is unusually high (pit laps) bcoz or else prediction will be false prediction
clean_laps = ver_laps[ver_laps['LapTimeSeconds'] < 105].copy()

clean_laps.head()

#Replotting after removing laptime :

plt.figure(figsize=(10, 5))
plt.plot(clean_laps['LapNumber'], clean_laps['LapTimeSeconds'], marker='o')
plt.xlabel('Lap Number')
plt.ylabel('Lap Time (seconds)')
plt.title('Verstappen Clean Lap Times (Pit Laps Removed)')
plt.grid(True)
plt.show()

# Scale the data

scaler = MinMaxScaler()
lap_times = clean_laps['LapTimeSeconds'].values

lap_times_scaled = scaler.fit_transform(lap_times.reshape(-1, 1))
lap_times_scaled[:10]

def create_sequences(data, window_size=5):
    X = []
    y = []

    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])

    return np.array(X), np.array(y)

window_size = 5

X, y = create_sequences(
    lap_times_scaled.flatten(),
    window_size
)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Reshape X for LSTM

X = X.reshape((X.shape[0], X.shape[1], 1))
X.shape

# TRAINING LSTM MODEL

model = Sequential([LSTM(64, input_shape=(X.shape[1], 1)), Dense(1)])

model.compile(optimizer='adam',loss='mse')
model.summary()

history = model.fit( X, y, epochs=30, batch_size=8, verbose=1 )

# PREDICTIONS :

y_pred_scaled = model.predict(X)
y_pred_scaled[:5]

y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y.reshape(-1, 1))

y_pred[:5], y_true[:5]

lap_numbers = clean_laps['LapNumber'].values[window_size:]

#PLOTTING ACTUAL VS PREDICTED LAP

plt.figure(figsize=(12, 6))

plt.plot(lap_numbers, y_true.flatten(), label='Actual Lap Time', marker='o')
plt.plot(lap_numbers, y_pred.flatten(), label='Predicted Lap Time', marker='x')

plt.xlabel('Lap Number')
plt.ylabel('Lap Time (seconds)')
plt.title('Actual vs Predicted Lap Times - (Bahrain GP 2023)')
plt.legend()
plt.grid(True)
plt.show()

## ERROR DETECTION

errors = y_true.flatten() - y_pred.flatten()
print("Mean Absolute Error (seconds):", abs(errors).mean())

## TRACK VISUALIZATION

ver_laps_ff1 = session.laps.pick_driver('VER')


lap = ver_laps_ff1.pick_fastest()


telemetry = lap.get_telemetry()

telemetry.head()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.figure(figsize=(10, 5))

x_pov = -telemetry['Y']
y_pov = telemetry['X']

sc = plt.scatter( x_pov, y_pov, c=telemetry['Speed'], cmap='Reds', s=6 )

plt.colorbar(sc, label='Speed (km/h)')
plt.title('Bahrain Grand Prix - Speed Map (Single Lap)')
plt.axis('equal')
plt.axis('on')

logo = mpimg.imread('/content/logo.png')
ax = plt.gca()
logo_ax = ax.inset_axes([0.85, 0.90, 0.08, 0.12])

logo_ax.imshow(logo)
logo_ax.axis('off')

plt.show()

