import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# ---------------- Title ----------------
st.title("🌤️ KNN Weather Prediction App")

st.write(
    """
This app predicts whether the weather is Sunny or Rainy based on Temperature and Humidity using K-Nearest Neighbors.
"""
)

# ---------------- Input ----------------
st.sidebar.header("Enter Weather Conditions")
temperature = st.sidebar.slider("Temperature (°C)", 15, 40, 26)
humidity = st.sidebar.slider("Humidity (%)", 40, 100, 78)

# ---------------- Data ----------------
X = np.array([[30, 70], [25, 80], [27, 60], [31, 65], [23, 85], [28, 75]])
y = np.array([0, 1, 0, 0, 1, 1])  # 0 = Sunny, 1 = Rainy

# ---------------- KNN Model ----------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# ---------------- Prediction ----------------
new_point = np.array([[temperature, humidity]])
prediction = knn.predict(new_point)[0]

weather = "Sunny ☀️" if prediction == 0 else "Rainy 🌧️"
st.subheader(f"Predicted Weather: {weather}")

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(7, 5))

# Sunny points
ax.scatter(X[y == 0, 0], X[y == 0, 1], label="Sunny", s=100)

# Rainy points
ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Rainy", s=100)

# New predicted point
ax.scatter(new_point[0, 0], new_point[0, 1], marker="*", s=300, color="red", label="New Prediction")

ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Humidity (%)")
ax.set_title("KNN Weather Classification")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
