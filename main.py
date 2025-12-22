import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import os
import torch.nn as nn
from notebook.architecture import GRUModel




def load_scaler(domain: str):
    path = f"notebook/{domain}_scaler.joblib"
    if not os.path.exists(path):
        st.error(f"Scaler file missing: {path}")
        return None
    return joblib.load(path)



def load_model(domain: str, X_seq: np.ndarray):

    input_size = X_seq.shape[2]
    path = f"models/{domain}_model.pth"
    if not os.path.exists(path):
        st.error(f"Model file missing: {path}")
        return None

    model = GRUModel(input_size=input_size)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def preprocess_test_file(file, scaler):
    df = pd.read_csv(file, sep=" ", header=None)

    df = df.iloc[:, :26]

    col_names = [
        "engine_id", "cycle",
        "op1", "op2", "op3",
        "s1","s2","s3","s4","s5",
        "s6","s7","s8","s9","s10",
        "s11","s12","s13","s14","s15",
        "s16","s17","s18","s19","s20","s21"
    ]
    df.columns = col_names

    sensor_cols = col_names[2:]
    df[sensor_cols] = scaler.transform(df[sensor_cols])

    dead_sensors = [c for c in sensor_cols if df[c].nunique() == 1]
    df = df.drop(columns=dead_sensors)

    return df


def create_test_sequences(df, sequence_length=50):
    X = []
    engine_ids = df['engine_id'].unique()
    for eid in engine_ids:
        engine_df = df[df['engine_id'] == eid].sort_values("cycle")
        engine_features = engine_df.drop(columns=['engine_id', 'cycle']).values
        n_cycles = engine_features.shape[0]

        if n_cycles >= sequence_length:
            X.append(engine_features[-sequence_length:])
        else:
            pad = np.zeros((sequence_length - n_cycles, engine_features.shape[1]))
            X.append(np.vstack([pad, engine_features]))
    return np.array(X, dtype=np.float32)



def classify_alert(rul):
    if rul <= 10:
        return "CRITICAL"
    elif rul <= 30:
        return "WARNING"
    else:
        return "NORMAL"


def predict_rul(X_seq, model):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_seq, dtype=torch.float32)).numpy().flatten()
    return preds


st.set_page_config(page_title="RUL Predictor", layout="wide")
st.title("RUL Predictor")

domain = st.sidebar.selectbox("Select Domain", ["FD001", "FD002", "FD003", "FD004"])


test_file = st.file_uploader("Upload Data", type=["txt"])
rul_file = st.file_uploader("Upload RUL", type=["txt"])

st.write("---")

if st.button("Run Prediction"):
    if test_file is None or rul_file is None:
        st.error("Please upload both test and RUL files.")
    else:
        st.info("Loading...")
        scaler = load_scaler(domain)
        if scaler is None:
            st.stop()

        # Preprocess test data
        df_test = preprocess_test_file(test_file, scaler)
        true_rul = pd.read_csv(rul_file, header=None).values.flatten()
        true_rul = np.minimum(true_rul, 125)

        # Create sequences
        X_test = create_test_sequences(df_test, sequence_length=50)

        st.info(f"Loading model for {domain}...")
        model = load_model(domain, X_test)
        if model is None:
            st.stop()

        # Predict
        preds = predict_rul(X_test, model)

        # Alerts
        alerts = [classify_alert(p) for p in preds]

        result_df = pd.DataFrame({
            "Unit": np.arange(1, len(true_rul)+1),
            "True RUL": true_rul,
            "Predicted RUL": preds,
            "Alert": alerts
        })

        st.success("Prediction completed!")
        st.subheader("Alerts Table")
        st.dataframe(result_df, use_container_width=True)

        critical_df = result_df[result_df["Alert"] == "CRITICAL"]
        if not critical_df.empty:
            st.error("⚠️ Critical Engines Detected!")
            st.dataframe(critical_df, use_container_width=True)
        else:
            st.success("No critical engines detected.")
