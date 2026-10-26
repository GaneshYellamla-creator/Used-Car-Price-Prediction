
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

MODEL_FILE = Path("models/car_price_model.joblib")
INFO_FILE = Path("models/model_info.json")
PREDICTIONS_FILE = Path("models/test_predictions.csv")

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚚",
    layout="wide"
)

st.title(" 🚚 Used Car Price Prediction")
st.caption("It helps to estimate the used cars price predict using Machine Learning and it uses linear algorithm")
if not MODEL_FILE.exists() or not INFO_FILE.exists():
    st.warning("The model has not been trained yet.")
    st.code("python train_model.py")
    st.info(
        "Run the training command in your VS Code terminal, "
        "then refresh this page."
    )
    st.stop()

model = joblib.load(MODEL_FILE)

with open(INFO_FILE, "r", encoding="utf-8") as file:
    info = json.load(file)

metrics = info["metrics"]
features = info["features"]
feature_info = info["feature_info"]

st.sidebar.header("About this project")
st.sidebar.write(
    "This application estimates used-car selling prices "
    "from the features in the training dataset."
)
st.sidebar.write("**Algorithm:** Linear Regression")
st.sidebar.write(f"**Target:** {info['target_col']}")
st.sidebar.write(f"**Rows used:** {info['rows_used']}")

predict_tab, evaluation_tab, details_tab = st.tabs([
    "💸 Predict Price",
    "📈 📉 Model Evaluation",
    "📃 Project Details"
])

with predict_tab:
    st.subheader("Enter car details")
    st.write("Complete the fields below and click Predict Price.")

    with st.form("car_prediction_form"):
        inputs = {}
        columns = st.columns(2)

        for i, feature in enumerate(features):
            meta = feature_info[feature]
            label = feature.replace("_", " ").replace("-", " ").title()

            with columns[i % 2]:
                if meta["type"] == "numeric":
                    default = float(meta["default"])
                    inputs[feature] = st.number_input(
                        label,
                        value=default,
                        format="%.2f"
                    )
                else:
                    options = meta.get("options", [])
                    default = str(meta.get("default", "Unknown"))

                    # Large-cardinality fields such as model names
                    # are better handled as text inputs.
                    if meta.get("unique_count", 0) > 50:
                        inputs[feature] = st.text_input(
                            label, value=default
                        )
                    elif options:
                        if default not in options:
                            options = [default] + options
                        inputs[feature] = st.selectbox(
                            label, options=options
                        )
                    else:
                        inputs[feature] = st.text_input(
                            label, value=default
                        )

        submitted = st.form_submit_button(
            "Predict Price", type="primary", use_container_width=True
        )

    if submitted:
        input_df = pd.DataFrame([inputs], columns=features)

        try:
            prediction = float(model.predict(input_df)[0])

            if prediction < 0:
                st.warning(
                    "The model returned a negative estimate. "
                    "This may indicate that the entered details are "
                    "outside the range represented by the training data."
                )
            else:
                st.success("Prediction completed!")
                st.metric(
                    "Estimated Selling Price",
                    f"₹{prediction:,.2f}"
                )
                st.caption(
                    "This is a model estimate, not a guaranteed market price."
                )
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.info(
                "Check the dataset column types and retrain the model."
            )

with evaluation_tab:
    st.subheader("Model performance on the test set")

    col1, col2, col3 = st.columns(3)

    col1.metric("Mean Absolute Error (MAE)", f"{metrics['mae']:,.2f}")
    col2.metric("Root Mean Squared Error (RMSE)", f"{metrics['rmse']:,.2f}")
    col3.metric("R² Score", f"{metrics['r2']:.4f}")

    st.caption(
        "MAE and RMSE are expressed in the target price's units. "
        "Lower error is generally better. R² closer to 1 generally "
        "indicates a better fit, but a negative score is possible."
    )

    if PREDICTIONS_FILE.exists():
        results = pd.read_csv(PREDICTIONS_FILE)

        st.subheader("Actual vs Predicted Prices")
        fig1, ax1 = plt.subplots()
        ax1.scatter(
            results["Actual Price"],
            results["Predicted Price"],
            alpha=0.65
        )

        low = min(
            results["Actual Price"].min(),
            results["Predicted Price"].min()
        )
        high = max(
            results["Actual Price"].max(),
            results["Predicted Price"].max()
        )

        ax1.plot([low, high], [low, high], linestyle="--")
        ax1.set_xlabel("Actual Price")
        ax1.set_ylabel("Predicted Price")
        ax1.set_title("Actual vs Predicted")
        st.pyplot(fig1)
        plt.close(fig1)

        st.subheader("Prediction Error Distribution")
        results["Error"] = (
            results["Actual Price"] - results["Predicted Price"]
        )

        fig2, ax2 = plt.subplots()
        ax2.hist(results["Error"], bins=20)
        ax2.set_xlabel("Actual Price - Predicted Price")
        ax2.set_ylabel("Number of Cars")
        ax2.set_title("Distribution of Prediction Errors")
        st.pyplot(fig2)
        plt.close(fig2)

        with st.expander("View test-set predictions"):
            st.dataframe(results, use_container_width=True)
    else:
        st.info("Run train_model.py to generate evaluation graphs.")

with details_tab:
    st.subheader("Project information")

    st.write("**Project:** Used Car Price Prediction")
    st.write("**Algorithm:** Linear Regression")
    st.write("**Dataset:** data/used_cars.csv")
    st.write(f"**Target column:** {info['target_col']}")
    st.write(f"**Number of features:** {len(features)}")
    st.write(f"**Rows used for training/testing:** {info['rows_used']}")

    st.write("**Input features:**")
    st.write(", ".join(features))

    st.write("**Training/testing split:** 80% training and 20% testing.")
