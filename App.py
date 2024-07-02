import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon=":syringe:", layout="wide")

st.title("Diabetes Risk Predictor :syringe:")

st.write("""
Welcome to the enhanced diabetes risk prediction app powered by advanced machine learning techniques.

**Scroll down to discover your results** :point_down:
""")

# Load data
data = pd.read_csv('data/diabetes.csv')

# Display dataset overview
st.subheader("Dataset Snapshot :open_file_folder:")
st.dataframe(data.style.highlight_max(axis=0))

# Display statistical information
st.subheader("Statistical Overview :bar_chart:")
st.write(data.describe().style.format("{:.2f}").background_gradient(cmap='viridis'))

# Data visualization
st.subheader("Visual Data Insights :chart_with_upwards_trend:")
st.line_chart(data)

# User input parameters
st.sidebar.header("Input Your Health Metrics :clipboard:")

X = data.iloc[:, 0:8].values
Y = data.iloc[:, -1].values

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Function to capture user input
def user_input():
    pregnancies = st.sidebar.slider("Number of Pregnancies :baby:", 0, 17, 3, format="%d")
    glucose = st.sidebar.slider("Glucose Level :droplet:", 0, 199, 117, format="%d")
    blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg) :heart:", 0, 122, 72, format="%d")
    skin_thickness = st.sidebar.slider("Skin Thickness (mm) :skin-tone-3:", 0, 99, 23, format="%d")
    insulin = st.sidebar.slider("Insulin Level (μU/mL) :syringe:", 0.0, 846.0, 30.0, format="%.1f")
    BMI = st.sidebar.slider("Body Mass Index (BMI) :straight_ruler:", 0.0, 67.1, 32.0, format="%.1f")
    dpf = st.sidebar.slider("Diabetes Pedigree Function :dna:", 0.078, 2.42, 0.3725, format="%.3f")
    age = st.sidebar.slider("Age (years) :older_man:", 21, 81, 29, format="%d")

    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'Blood Pressure': blood_pressure,
        'Skin Thickness': skin_thickness,
        'Insulin': insulin,
        'BMI': BMI,
        'Diabetes Pedigree Function': dpf,
        'Age': age
    }
    features = pd.DataFrame(user_data, index=['User Input'])
    return features

df = user_input()

st.subheader("Your Health Metrics :clipboard:")
st.write(df.style.format("{:.2f}").background_gradient(cmap='viridis'))

# Train the model
clf = RandomForestClassifier()
clf.fit(X_train, Y_train)

# Prediction
prediction = clf.predict(df)
pred_prob = clf.predict_proba(df)

# Model evaluation
predictions = clf.predict(X_test)
accuracy = metrics.accuracy_score(Y_test, predictions)

# Display results in a more UX-friendly way
st.subheader("Prediction Results :mag:")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("### Diagnosis :memo:")
    diagnosis = 'Diabetes Detected :warning:' if prediction[0] == 1 else 'No Diabetes :smile:'
    st.write(diagnosis)

with col2:
    st.write("### Prediction Confidence :chart_with_upwards_trend:")
    st.write(f"Positive: {pred_prob[0][1]:.2f}, Negative: {pred_prob[0][0]:.2f}")

with col3:
    st.write("### Model Accuracy :thumbsup:")
    st.write(f"{accuracy:.2f}")
