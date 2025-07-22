import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import os

# ---------------------------------------------
# 📌 Page Config
# ---------------------------------------------
st.set_page_config(
    page_title="🎓 Student Performance Classifier",
    layout="wide",
    page_icon="📊"
)

st.title("🎓 Student Performance Classification App")
st.write(
    "Upload student performance data, train a deep learning model, and make predictions interactively."
)

# ---------------------------------------------
# 📌 File Uploader
# ---------------------------------------------
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("Student_Performance.csv")

st.write("## Dataset Preview")
st.dataframe(df.head())

# ---------------------------------------------
# 📌 Data Preprocessing
# ---------------------------------------------
def categorize_performance(score):
    if score < 50:
        return 0  # Fail
    elif score <= 75:
        return 1  # Pass
    else:
        return 2  # Excellent

df['Performance Class'] = df['Performance Index'].apply(categorize_performance)
y = to_categorical(df['Performance Class'])
X = df.drop(['Performance Index', 'Performance Class'], axis=1)

# One-hot encode categorical features if any
X = pd.get_dummies(X)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------
# 📌 Train-Test Split
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------
# 📌 Model Building Function
# ---------------------------------------------
def build_model():
    model = Sequential([
        Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dense(32, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.3),
        Dense(16, activation='relu', kernel_initializer='he_normal'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------------------------------------
# 📌 Train Button
# ---------------------------------------------
if st.button("🚀 Train Model"):
    with st.spinner("Training in progress... ⏳"):
        model = build_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, checkpoint, reduce_lr],
            verbose=0
        )

    st.success("✅ Training completed!")

    # Metrics
    best_model = load_model('best_model.h5')
    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = np.mean(y_pred_classes == y_true)
    st.write(f"**Accuracy:** {acc:.4f}")

    st.write("## 📌 Classification Report")
    report = classification_report(y_true, y_pred_classes, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.write("## 📌 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred_classes)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail', 'Pass', 'Excellent'], yticklabels=['Fail', 'Pass', 'Excellent'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # Loss Plot
    st.write("## 📌 Training History")
    fig2, ax2 = plt.subplots(1,2, figsize=(14,5))

    ax2[0].plot(history.history['loss'], label='Train Loss')
    ax2[0].plot(history.history['val_loss'], label='Val Loss')
    ax2[0].set_title("Loss Curve")
    ax2[0].legend()

    ax2[1].plot(history.history['accuracy'], label='Train Acc')
    ax2[1].plot(history.history['val_accuracy'], label='Val Acc')
    ax2[1].set_title("Accuracy Curve")
    ax2[1].legend()

    st.pyplot(fig2)

    # Download
    with open("best_model.h5", "rb") as f:
        st.download_button("⬇️ Download Best Model", f, file_name="best_model.h5")

# ---------------------------------------------
# 📌 PREDICTION SECTION
# ---------------------------------------------
st.header("🔍 Make Predictions")

predict_option = st.radio("Choose Input Type:", ["Manual Input", "Upload CSV for Prediction"])

best_model_exists = os.path.exists("best_model.h5")

if best_model_exists:
    model_loaded = load_model("best_model.h5")

    # 👉 List of Yes/No Columns (adjust as per your dataset!)
    yes_no_cols = ["extracurricular activities", "Extra_Activities"]

    if predict_option == "Manual Input":
        st.subheader("Enter values to predict:")

        user_input = {}

        for col in X.columns:
            if col.lower() in [x.lower() for x in yes_no_cols]:
                user_input[col] = 1 if st.radio(f"{col}", ["Yes", "No"]) == "Yes" else 0
            else:
                user_input[col] = st.number_input(f"{col}", value=0, step=1, format="%d")

        if st.button("Predict Now!"):
            input_df = pd.DataFrame([user_input])
            input_scaled = scaler.transform(input_df)
            pred = model_loaded.predict(input_scaled)
            pred_class = np.argmax(pred, axis=1)[0]
            class_map = {0: "Fail", 1: "Pass", 2: "Excellent"}
            st.success(f"🎉 Predicted Class: **{class_map[pred_class]}**")

    elif predict_option == "Upload CSV for Prediction":
        pred_file = st.file_uploader("Upload CSV for Prediction", type="csv", key="predict_csv")
        if pred_file is not None:
            new_data = pd.read_csv(pred_file)
            st.write("Uploaded Data:")
            st.dataframe(new_data.head())

            new_data = pd.get_dummies(new_data)
            new_data = new_data.reindex(columns=X.columns, fill_value=0)

            new_scaled = scaler.transform(new_data)
            preds = model_loaded.predict(new_scaled)
            pred_classes = np.argmax(preds, axis=1)

            result_df = new_data.copy()
            result_df['Predicted Class'] = pred_classes
            result_df['Performance Label'] = result_df['Predicted Class'].map({0: "Fail", 1: "Pass", 2: "Excellent"})

            st.write("## 📌 Predictions")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions", csv, "predictions.csv")
else:
    st.info("👉 Please train the model first to enable prediction.")

# ---------------------------------------------
# 📌 Footer
# ---------------------------------------------
st.sidebar.markdown("---")
st.sidebar.write("Built with ❤️ by Dua")
