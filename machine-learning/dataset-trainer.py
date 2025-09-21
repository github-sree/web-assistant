from sklearn import datasets
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Create a dataset
data = {
    "engine_size": [1.0, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0],
    "horsepower": [70, 85, 110, 150, 180, 220, 280, 350],
    "weight": [900, 1000, 1100, 1300, 1500, 1700, 1800, 2000],
    "aerodynamics": [0.35, 0.34, 0.32, 0.30, 0.29, 0.28, 0.27, 0.25],
    "top_speed": [150, 165, 180, 200, 220, 240, 260, 300]  # target variable
}

df = pd.DataFrame(data)

x = df.drop("top_speed", axis=1)
y = df["top_speed"]
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_tr, y_tr)
y_prediction = model.predict(x_te)

print("Predicted Speeds:", y_prediction)
print("Actual Speeds:", y_te.values)

new_car = [[2.2, 160, 1400, 0.29]]
predicted_speed = model.predict(new_car)
print("Predicted Top Speed:", predicted_speed[0])
