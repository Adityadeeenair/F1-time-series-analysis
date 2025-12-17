# 🏁 Formula 1 Race Pace Prediction using Machine Learning

---

## 📌 Overview
This project explores how **Formula 1 race data** can be analyzed beyond traditional lap charts and final standings. By focusing on **lap-by-lap race pace**, it studies whether short-term lap time patterns within a race can be learned and predicted using **time-series modeling**, instead of relying on simple averages or linear trends.

Using real telemetry and timing data, the project combines **machine learning**, **race analytics**, and **visualization** to extract deeper insights into race dynamics.

---

## 🎯 Project Goals
- Analyze lap-by-lap race pace using real Formula 1 data  
- Study stint-wise performance evolution across a race  
- Predict future lap times using deep learning models  
- Visualize telemetry data directly on the circuit layout  

---

## 📊 Data Source
The project uses the **FastF1** Python library to access official Formula 1 race data, including:
- Lap times and stint information  
- Session-level timing data  
- Telemetry data such as speed and track position  

FastF1 provides a structured and reproducible way to work with real-world Formula 1 race sessions.

---

## 🧠 Methodology

### 🔹 Data Preparation
- Extract raw lap and telemetry data from race sessions  
- Clean and structure lap times into a time-series format  
- Segment laps by stints to observe pace changes over time  

### 🔹 Exploratory Analysis
- Analyze lap time trends across the race  
- Identify patterns related to tire wear, pit windows, and race phases  
- Understand sources of noise introduced by real race conditions  

### 🔹 Modeling Approach
- Use an **LSTM-based neural network** for time-series forecasting  
- Train the model using recent lap history as contextual input  
- Predict the next lap time in a race sequence  

### 🔹 Evaluation
- Compare predicted lap times against actual values  
- Analyze prediction errors to understand model strengths and limitations  
- Examine where race dynamics affect predictability  

---

## 🗺️ Visualization
Telemetry data is mapped directly onto the **circuit layout**, creating a **speed-based track visualization**.  
This highlights:
- High-speed and low-speed sections of the track  
- Performance variation across corners and straights  

---

## 🚀 Key Takeaways
- Race pace exhibits short-term patterns that can be partially learned  
- Time-series models are effective but sensitive to real-world race events  
- Domain-specific visualizations improve interpretability  

---

## 🛠️ Tools & Technologies
- **Python**
- **FastF1**
- **NumPy & Pandas**
- **Matplotlib**
- **TensorFlow / Keras**
- **LSTM Neural Networks**

---

## 📈 Future Improvements
- Extend the model to multi-lap forecasting  
- Incorporate additional features such as tire compounds and weather  
- Compare deep learning models with classical forecasting methods  
- Generalize the approach across multiple circuits and seasons  

---

## 📎 Notes
This project is intended as a **learning-focused exploration** of machine learning applied to motorsport analytics, emphasizing clarity, interpretability, and real-world data challenges.

---

## 🏎️ Final Thought
Formula 1 is a data-rich sport where performance is shaped by countless interacting factors. This project demonstrates how **machine learning and telemetry analysis** can be combined to better understand race pace beyond surface-level metrics.
