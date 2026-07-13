import pandas as pd
import streamlit as st

from telco_churn.predict import score_customers
from telco_churn.utils import MODELS_DIR, get_feature_columns, load_model


# Page settings
st.set_page_config(
    page_title="Telco Churn Predictor",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    """Load the trained model and preprocessing files."""

    model = load_model(MODELS_DIR / "final_model.pkl")
    scaler = load_model(MODELS_DIR / "scaler.pkl")
    feature_columns = get_feature_columns()

    return model, scaler, feature_columns


def get_risk_label(probability):
    """Return the risk level and recommended action."""

    if probability >= 0.70:
        return "High Risk", "Contact this customer as soon as possible."

    if probability >= 0.40:
        return "Medium Risk", "Monitor this customer and consider a follow-up."

    return "Low Risk", "No urgent action is needed."


def build_customer_form():
    """Get one customer's information from the user."""

    col1, col2, col3 = st.columns(3)

    # Personal information
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

        tenure = st.number_input(
            "Tenure",
            min_value=0,
            max_value=100,
            value=12,
        )

    # Phone and internet services
    with col2:
        phone = st.selectbox("Phone Service", ["Yes", "No"])

        if phone == "Yes":
            multiple_lines_options = ["No", "Yes"]
        else:
            multiple_lines_options = ["No phone service"]

        multiple_lines = st.selectbox(
            "Multiple Lines",
            multiple_lines_options,
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )

        if internet == "No":
            internet_options = ["No internet service"]
        else:
            internet_options = ["No", "Yes"]

        online_security = st.selectbox(
            "Online Security",
            internet_options,
        )

        online_backup = st.selectbox(
            "Online Backup",
            internet_options,
        )

        device_protection = st.selectbox(
            "Device Protection",
            internet_options,
        )

    # Contract and other services
    with col3:
        tech_support = st.selectbox(
            "Tech Support",
            internet_options,
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            internet_options,
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            internet_options,
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    charges_col1, charges_col2 = st.columns(2)

    with charges_col1:
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0,
        )

    with charges_col2:
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=float(monthly_charges * max(tenure, 1)),
            step=10.0,
        )

    # Store the customer's information in a dictionary
    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # Convert one customer into a one-row DataFrame
    return pd.DataFrame([customer])


# Main page
st.title("Customer Retention Intelligence")

st.write(
    "Predict churn for one customer or upload a CSV file "
    "to score multiple customers."
)


# Load the model files once
model, scaler, feature_columns = load_artifacts()


# Create two tabs
single_tab, batch_tab = st.tabs(
    ["Single Customer", "Batch Scoring"]
)


# Single customer section
with single_tab:
    st.subheader("Single Customer Prediction")

    customer = build_customer_form()

    if st.button("Score Customer", type="primary"):
        result = score_customers(
            customer,
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
        )

        probability = float(
            result["churn_probability"].iloc[0]
        )

        risk_label, action = get_risk_label(probability)

        col1, col2 = st.columns(2)

        col1.metric(
            "Churn Probability",
            f"{probability:.1%}",
        )

        col2.metric(
            "Risk Level",
            risk_label,
        )

        if probability >= 0.70:
            st.error(action)

        elif probability >= 0.40:
            st.warning(action)

        else:
            st.success(action)

        st.dataframe(
            result,
            hide_index=True,
            use_container_width=True,
        )


# Multiple customer section
with batch_tab:
    st.subheader("Batch Customer Scoring")

    uploaded_file = st.file_uploader(
        "Upload a customer CSV file",
        type=["csv"],
    )

    top_n = st.slider(
        "Number of customers to display",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
    )

    if uploaded_file is not None:
        customers = pd.read_csv(uploaded_file)

        scores = score_customers(
            customers,
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
        )

        # Show customers with the highest risk first
        scores = scores.sort_values(
            by="churn_probability",
            ascending=False,
        )

        customer_count = len(scores)

        average_risk = scores[
            "churn_probability"
        ].mean()

        high_risk_count = (
            scores["churn_probability"] >= 0.70
        ).sum()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Scored Customers",
            f"{customer_count:,}",
        )

        col2.metric(
            "Average Churn Risk",
            f"{average_risk:.1%}",
        )

        col3.metric(
            "High Risk Customers",
            f"{high_risk_count:,}",
        )

        st.dataframe(
            scores.head(top_n),
            hide_index=True,
            use_container_width=True,
        )

        # Convert results to CSV for downloading
        csv_file = scores.to_csv(index=False)

        st.download_button(
            label="Download Scores",
            data=csv_file,
            file_name="customer_churn_scores.csv",
            mime="text/csv",
        )