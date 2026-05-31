import pandas as pd
import streamlit as st


from telco_churn.predict import score_customers
from telco_churn.utils import MODELS_DIR, get_feature_columns, load_model


st.set_page_config(
    page_title="Telco Churn Predictor",
    layout="wide",
)


def risk_label(probability: float) -> tuple[str, str]:
    """Return a simple risk label and action message."""

    if probability >= 0.7:
        return "High risk", "Prioritize this customer for retention outreach."

    if probability >= 0.4:
        return "Medium risk", "Monitor this customer and consider a light follow-up."

    return "Low risk", "No urgent action needed."


def load_artifacts():
    """Load the saved model files."""

    model = load_model(MODELS_DIR / "final_model.pkl")
    scaler = load_model(MODELS_DIR / "scaler.pkl")
    feature_columns = get_feature_columns()

    return model, scaler, feature_columns


def build_customer_row() -> pd.DataFrame:
    """Collect one customer's information from the form."""

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure", min_value=0, max_value=100, value=12)

    with col2:
        phone = st.selectbox("Phone Service", ["Yes", "No"])

        if phone == "No":
            multiple_options = ["No phone service"]
        else:
            multiple_options = ["No", "Yes"]

        multiple = st.selectbox("Multiple Lines", multiple_options)

        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        if internet == "No":
            internet_options = ["No internet service"]
        else:
            internet_options = ["No", "Yes"]

        online_security = st.selectbox("Online Security", internet_options)
        online_backup = st.selectbox("Online Backup", internet_options)
        device = st.selectbox("Device Protection", internet_options)

    with col3:
        tech = st.selectbox("Tech Support", internet_options)
        tv = st.selectbox("Streaming TV", internet_options)
        movies = st.selectbox("Streaming Movies", internet_options)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
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
        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0,
        )

    with charges_col2:
        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=float(monthly * max(tenure, 1)),
            step=10.0,
        )

    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    return pd.DataFrame([customer])


st.title("Customer Retention Intelligence")
st.write("Score one customer or upload a CSV file to find customers with higher churn risk.")


# Load saved model files only once.
if "artifacts_loaded" not in st.session_state:
    model, scaler, feature_columns = load_artifacts()

    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.feature_columns = feature_columns
    st.session_state.artifacts_loaded = True


model = st.session_state.model
scaler = st.session_state.scaler
feature_columns = st.session_state.feature_columns


single_tab, batch_tab = st.tabs(["Single Customer", "Batch Scoring"])


with single_tab:
    customer = build_customer_row()

    if st.button("Score Customer", type="primary"):
        score = score_customers(
            customer,
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
        )

        probability = float(score["churn_probability"].iloc[0])
        label, action = risk_label(probability)

        metric_col, segment_col, action_col = st.columns([1, 1, 2])

        metric_col.metric("Churn Probability", f"{probability:.1%}")
        segment_col.metric("Risk Segment", label)
        action_col.metric("Recommended Action", action)

        if probability >= 0.7:
            st.error(action)
        elif probability >= 0.4:
            st.warning(action)
        else:
            st.success(action)

        st.dataframe(score, hide_index=True, use_container_width=True)


with batch_tab:
    uploaded = st.file_uploader("Customer CSV", type=["csv"])
    top_n = st.slider(
        "Customers to review",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
    )

    if uploaded is not None:
        customers = pd.read_csv(uploaded)

        scores = score_customers(
            customers,
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
        )

        high_risk = int((scores["risk_segment"] == "high").sum())
        average_risk = float(scores["churn_probability"].mean())

        count_col, avg_col, high_col = st.columns(3)

        count_col.metric("Scored Customers", f"{len(scores):,}")
        avg_col.metric("Average Churn Risk", f"{average_risk:.1%}")
        high_col.metric("High Risk Customers", f"{high_risk:,}")

        st.dataframe(
            scores.head(top_n),
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download Scores",
            data=scores.to_csv(index=False).encode("utf-8"),
            file_name="customer_churn_scores.csv",
            mime="text/csv",
        )