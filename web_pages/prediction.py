import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
from src.feature_engineering import feature_engineering_func
import streamlit as st


def model_prediction():

    # ======================
    # Load and preprocess data
    # ======================
    merged, features = feature_engineering_func()

    merged['Target_Remark'] = merged['Remark_sem2'].apply(
        lambda x: 1 if x == 'PASS' else 0
    )

    target_sgpa = merged['SGPA_sem2']

    X = merged[features]
    y = merged['Target_Remark']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ======================
    # Feature Scaling
    # ======================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ======================
    # MODEL 1: Random Forest Classifier
    # ======================
    st.write("### Model 1: Random Forest Classifier (PASS / FAIL Prediction)")

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    # ======================
    # Model Performance Summary (YOUR ENHANCEMENT)
    # ======================
    accuracy = accuracy_score(y_test, y_pred)
    cv_score = np.mean(cross_val_score(clf, X, y, cv=5))

    st.markdown("### 📊 Model Performance Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Classification Accuracy", f"{accuracy:.2f}")
    with col2:
        st.metric("Cross Validation (5-Fold)", f"{cv_score:.2f}")

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.info("""
    The Random Forest model classifies students as PASS or FAIL based on
    historical academic performance and engineered features.
    Higher accuracy indicates better prediction capability on unseen data.
    """)

    # ======================
    # Feature Importance
    # ======================
    st.write("### Feature Importance (Random Forest)")

    feature_importance = clf.feature_importances_

    fig = go.Figure([
        go.Bar(
            x=feature_importance,
            y=features,
            orientation='h'
        )
    ])

    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Features"
    )

    st.plotly_chart(fig)

    # ======================
    # MODEL 2: Linear Regression (SGPA Prediction)
    # ======================
    st.write("### Model 2: Linear Regression (SGPA Prediction)")

    X_train_sgpa, X_test_sgpa, y_train_sgpa, y_test_sgpa = train_test_split(
        X, target_sgpa, test_size=0.2, random_state=42
    )

    reg = LinearRegression()
    reg.fit(X_train_sgpa, y_train_sgpa)

    y_pred_sgpa = reg.predict(X_test_sgpa)

    mse = mean_squared_error(y_test_sgpa, y_pred_sgpa)
    rmse = np.sqrt(mse)
    r2_score = reg.score(X_test_sgpa, y_test_sgpa)

    st.write("RMSE:", round(rmse, 3))
    st.write("R² Score:", round(r2_score, 3))

    # ======================
    # Actual vs Predicted Plot
    # ======================
    st.write("### Actual vs Predicted SGPA")

    scatter_fig = go.Figure()

    scatter_fig.add_trace(go.Scatter(
        x=y_test_sgpa,
        y=y_pred_sgpa,
        mode='markers',
        marker=dict(opacity=0.6),
        name='Predicted vs Actual'
    ))

    scatter_fig.add_trace(go.Scatter(
        x=[0, 10],
        y=[0, 10],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Reference Line'
    ))

    scatter_fig.update_layout(
        title="Actual vs Predicted SGPA",
        xaxis_title="Actual SGPA",
        yaxis_title="Predicted SGPA"
    )

    st.plotly_chart(scatter_fig)
