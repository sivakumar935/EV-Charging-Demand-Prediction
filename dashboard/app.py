# ============================================================
# EV CHARGING DEMAND PREDICTION & USAGE PATTERN ANALYSIS
# STREAMLIT DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import joblib
import warnings

from pathlib import Path

warnings.filterwarnings("ignore")


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EV Charging Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ EV Charging Demand Prediction")
st.subheader("Usage Pattern Analysis and Predictive Analytics")

st.markdown("""
This dashboard provides interactive analysis of EV charging sessions,
charging demand prediction, forecasting, and model explainability.
""")

st.markdown("---")


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_DIR /
    "data" /
    "cleaned" /
    "ChargingRecords_Featured.csv"
)

MODEL_DIR = PROJECT_DIR / "models"

RESULT_DIR = PROJECT_DIR / "results"

EXPLAIN_DIR = RESULT_DIR / "explainability"


# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    # Convert datetime columns

    for col in ["StartDatetime", "EndDatetime"]:

        if col in df.columns:

            df[col] = pd.to_datetime(
                df[col],
                errors="coerce"
            )

    # Convert numerical columns

    for col in ["Demand", "Duration"]:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df = df.dropna(
        subset=["StartDatetime", "Demand"]
    ).copy()

    return df


if not DATA_PATH.exists():

    st.error(
        f"Featured dataset not found: {DATA_PATH}"
    )

    st.stop()


df = load_data()


# ============================================================
# 4. LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_model(filename):

    path = MODEL_DIR / filename

    if path.exists():

        try:

            return joblib.load(path)

        except Exception as e:

            st.warning(
                f"Unable to load {filename}: {e}"
            )

    return None


regression_model = load_model(
    "best_regression_model.pkl"
)

classification_model = load_model(
    "best_classification_model.pkl"
)

forecast_model = load_model(
    "best_forecast_model.pkl"
)


# ============================================================
# 5. SIDEBAR FILTERS
# ============================================================

st.sidebar.title("Dashboard Filters")

st.sidebar.markdown("Filter the charging records.")


# Date filter

min_date = df["StartDatetime"].min().date()

max_date = df["StartDatetime"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",

    value=(min_date, max_date),

    min_value=min_date,

    max_value=max_date
)


filtered_df = df.copy()


if isinstance(date_range, (tuple, list)) and len(date_range) == 2:

    start_date, end_date = date_range

    filtered_df = filtered_df[
        filtered_df["StartDatetime"].dt.date.between(
            start_date,
            end_date
        )
    ]


# Location filter

def apply_filter(data, column, label):

    if column not in data.columns:

        return data

    options = sorted(
        data[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected = st.sidebar.multiselect(
        label,
        options,
        default=options
    )

    return data[
        data[column].astype(str).isin(selected)
    ]


filtered_df = apply_filter(
    filtered_df,
    "Location",
    "Location"
)


filtered_df = apply_filter(
    filtered_df,
    "ChargerType",
    "Charger Type"
)


filtered_df = apply_filter(
    filtered_df,
    "ChargerCompany",
    "Charger Company"
)


st.sidebar.markdown("---")

st.sidebar.write(
    f"Filtered Records: {len(filtered_df):,}"
)

if filtered_df.empty:

    st.warning(
        "No records found for the selected filters."
    )

    st.stop()


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def read_result(filename):

    path = RESULT_DIR / filename

    if path.exists():

        try:

            return pd.read_csv(path)

        except Exception:

            return None

    return None


def get_feature_defaults(data, expected_features):

    """
    Create a default input row using the featured dataset.

    Numerical features use their median.
    Categorical features use their most frequent value.
    """

    defaults = {}

    for col in expected_features:

        if col not in data.columns:

            defaults[col] = np.nan

            continue

        if pd.api.types.is_numeric_dtype(data[col]):

            defaults[col] = data[col].median()

        else:

            mode_values = data[col].dropna().mode()

            if len(mode_values) > 0:

                defaults[col] = mode_values.iloc[0]

            else:

                defaults[col] = "Unknown"

    return defaults


def create_prediction_input(
    model,
    data,
    location,
    charger_type,
    company,
    duration,
    prediction_datetime
):

    """
    Dynamically creates input using the features
    expected by the saved model pipeline.
    """

    if not hasattr(model, "named_steps"):

        raise ValueError(
            "The saved model is not a compatible sklearn pipeline."
        )

    if "preprocessor" not in model.named_steps:

        raise ValueError(
            "The model does not contain a preprocessor step."
        )

    preprocessor = model.named_steps["preprocessor"]

    if not hasattr(preprocessor, "feature_names_in_"):

        raise ValueError(
            "Cannot determine the model's expected input columns."
        )

    expected_features = list(
        preprocessor.feature_names_in_
    )

    # Start with dataset-based defaults

    row = get_feature_defaults(
        data,
        expected_features
    )

    # User-selected inputs

    overrides = {

        "Location": location,

        "ChargerType": charger_type,

        "ChargerCompany": company,

        "Duration": duration,

        "DurationHours": duration / 60,

        "Year": prediction_datetime.year,

        "Month": prediction_datetime.month,

        "Day": prediction_datetime.day,

        "DayOfWeek": prediction_datetime.dayofweek,

        "WeekOfYear": int(
            prediction_datetime.isocalendar().week
        ),

        "Quarter": prediction_datetime.quarter,

        "Hour": prediction_datetime.hour,

        "Minute": prediction_datetime.minute,

        "IsWeekend": int(
            prediction_datetime.dayofweek >= 5
        ),

        "HourSin": np.sin(
            2 * np.pi * prediction_datetime.hour / 24
        ),

        "HourCos": np.cos(
            2 * np.pi * prediction_datetime.hour / 24
        ),

        "MonthSin": np.sin(
            2 * np.pi * prediction_datetime.month / 12
        ),

        "MonthCos": np.cos(
            2 * np.pi * prediction_datetime.month / 12
        ),

        "DayOfWeekSin": np.sin(
            2 * np.pi * prediction_datetime.dayofweek / 7
        ),

        "DayOfWeekCos": np.cos(
            2 * np.pi * prediction_datetime.dayofweek / 7
        )

    }

    # Duration categories

    if duration <= 30:

        duration_category = "Short"

    elif duration <= 120:

        duration_category = "Medium"

    else:

        duration_category = "Long"

    overrides["DurationCategory"] = duration_category

    # Time of day

    hour = prediction_datetime.hour

    if 5 <= hour < 12:

        time_of_day = "Morning"

    elif 12 <= hour < 17:

        time_of_day = "Afternoon"

    elif 17 <= hour < 21:

        time_of_day = "Evening"

    else:

        time_of_day = "Night"

    overrides["TimeOfDay"] = time_of_day

    # Update available model features

    for col, value in overrides.items():

        if col in row:

            row[col] = value

    input_df = pd.DataFrame(
        [row],
        columns=expected_features
    )

    # Match categorical input types

    categorical_columns = []

    for name, transformer, columns in preprocessor.transformers_:

        if name == "cat":

            categorical_columns = list(columns)

    for col in categorical_columns:

        if col in input_df.columns:

            input_df[col] = input_df[col].astype(str)

    return input_df


# ============================================================
# 7. DASHBOARD TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([

    "Overview",

    "Usage Patterns",

    "Predictions",

    "Time Series",

    "Model Performance",

    "Explainability"

])


# ============================================================
# TAB 1: OVERALL OVERVIEW
# ============================================================

with tab1:

    st.header("Overall EV Charging Overview")

    total_sessions = len(filtered_df)

    total_demand = filtered_df["Demand"].sum()

    average_demand = filtered_df["Demand"].mean()

    average_duration = filtered_df["Duration"].mean()

    unique_locations = filtered_df["Location"].nunique()

    unique_chargers = filtered_df["ChargerID"].nunique()


    # KPI ROW 1

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Charging Sessions",
        f"{total_sessions:,}"
    )

    col2.metric(
        "Total Charging Demand",
        f"{total_demand:,.2f}"
    )

    col3.metric(
        "Average Demand",
        f"{average_demand:,.2f}"
    )


    # KPI ROW 2

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Average Duration",
        f"{average_duration:,.2f} min"
    )

    col5.metric(
        "Total Locations",
        f"{unique_locations:,}"
    )

    col6.metric(
        "Total Chargers",
        f"{unique_chargers:,}"
    )


    st.markdown("---")


    # Demand by location

    col1, col2 = st.columns(2)

    location_demand = (

        filtered_df.groupby("Location")["Demand"]

        .sum()

        .reset_index()

        .sort_values(
            "Demand",
            ascending=False
        )

    )

    with col1:

        fig = px.bar(

            location_demand,

            x="Location",

            y="Demand",

            title="Total Demand by Location",

            color="Demand"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Demand by charger type

    charger_demand = (

        filtered_df.groupby("ChargerType")["Demand"]

        .sum()

        .reset_index()

    )

    with col2:

        fig = px.pie(

            charger_demand,

            names="ChargerType",

            values="Demand",

            title="Demand Share by Charger Type",

            hole=0.4

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Daily demand trend

    daily_demand = (

        filtered_df.set_index("StartDatetime")

        .resample("D")["Demand"]

        .sum()

        .reset_index()

    )

    fig = px.line(

        daily_demand,

        x="StartDatetime",

        y="Demand",

        title="Daily Charging Demand Trend",

        markers=True

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# TAB 2: USAGE PATTERN ANALYSIS
# ============================================================

with tab2:

    st.header("EV Charging Usage Pattern Analysis")


    # Hourly demand

    col1, col2 = st.columns(2)

    hourly_demand = (

        filtered_df.groupby("Hour")["Demand"]

        .mean()

        .reset_index()

    )

    with col1:

        fig = px.bar(

            hourly_demand,

            x="Hour",

            y="Demand",

            title="Average Demand by Hour"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Weekday demand

    weekday_demand = (

        filtered_df.groupby("DayOfWeek")["Demand"]

        .mean()

        .reset_index()

    )

    weekday_names = {

        0: "Monday",

        1: "Tuesday",

        2: "Wednesday",

        3: "Thursday",

        4: "Friday",

        5: "Saturday",

        6: "Sunday"

    }

    weekday_demand["Day"] = (

        weekday_demand["DayOfWeek"]

        .map(weekday_names)

    )

    with col2:

        fig = px.bar(

            weekday_demand,

            x="Day",

            y="Demand",

            title="Average Demand by Weekday"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Company demand

    col3, col4 = st.columns(2)

    company_demand = (

        filtered_df.groupby("ChargerCompany")["Demand"]

        .sum()

        .reset_index()

    )

    with col3:

        fig = px.bar(

            company_demand,

            x="ChargerCompany",

            y="Demand",

            title="Demand by Charger Company"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Duration category

    with col4:

        if "DurationCategory" in filtered_df.columns:

            duration_demand = (

                filtered_df.groupby("DurationCategory")["Demand"]

                .mean()

                .reset_index()

            )

            fig = px.bar(

                duration_demand,

                x="DurationCategory",

                y="Demand",

                title="Average Demand by Duration Category"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # Monthly demand

    monthly_demand = (

        filtered_df.groupby("Month")["Demand"]

        .sum()

        .reset_index()

    )

    fig = px.line(

        monthly_demand,

        x="Month",

        y="Demand",

        title="Monthly Charging Demand",

        markers=True

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Data preview

    st.subheader("Charging Dataset Preview")

    st.dataframe(
        filtered_df.head(100),
        use_container_width=True
    )


# ============================================================
# TAB 3: REGRESSION AND CLASSIFICATION PREDICTIONS
# ============================================================

with tab3:

    st.header("EV Charging Demand Prediction")

    st.write(
        "Enter charging session details to predict demand "
        "and classify the expected demand category."
    )


    if regression_model is None:

        st.error(
            "Regression model not found. "
            "Check models/best_regression_model.pkl."
        )

    elif classification_model is None:

        st.error(
            "Classification model not found. "
            "Check models/best_classification_model.pkl."
        )

    else:

        # Dropdown values

        locations = sorted(
            df["Location"].dropna().astype(str).unique()
        )

        charger_types = sorted(
            df["ChargerType"].dropna().astype(str).unique()
        )

        companies = sorted(
            df["ChargerCompany"].dropna().astype(str).unique()
        )


        st.subheader("Enter Session Details")


        with st.form("prediction_form"):

            col1, col2, col3 = st.columns(3)

            location = col1.selectbox(
                "Location",
                locations
            )

            charger_type = col2.selectbox(
                "Charger Type",
                charger_types
            )

            company = col3.selectbox(
                "Charger Company",
                companies
            )


            col4, col5, col6 = st.columns(3)

            prediction_date = col4.date_input(
                "Session Date",
                value=max_date
            )

            prediction_hour = col5.slider(
                "Starting Hour",
                0,
                23,
                12
            )

            prediction_minute = col6.slider(
                "Starting Minute",
                0,
                59,
                0
            )


            duration = st.number_input(

                "Session Duration (Minutes)",

                min_value=1.0,

                max_value=1440.0,

                value=60.0,

                step=5.0

            )


            submit = st.form_submit_button(
                "Predict Charging Demand"
            )


        if submit:

            prediction_datetime = (

                pd.Timestamp(prediction_date)

                + pd.Timedelta(
                    hours=prediction_hour,
                    minutes=prediction_minute
                )

            )


            try:

                # --------------------------------------
                # REGRESSION PREDICTION
                # --------------------------------------

                regression_input = create_prediction_input(

                    regression_model,

                    df,

                    location,

                    charger_type,

                    company,

                    duration,

                    prediction_datetime

                )

                predicted_demand = regression_model.predict(
                    regression_input
                )[0]


                # --------------------------------------
                # CLASSIFICATION PREDICTION
                # --------------------------------------

                classification_input = create_prediction_input(

                    classification_model,

                    df,

                    location,

                    charger_type,

                    company,

                    duration,

                    prediction_datetime

                )

                predicted_category = classification_model.predict(
                    classification_input
                )[0]

                 # Convert model labels into readable demand categories
                category_labels = {
                    0: "Normal Demand",
                    1: "High Demand"
                }
                predicted_category_label = category_labels.get(
                    int(predicted_category),
                    str(predicted_category)
                )


                # --------------------------------------
                # DISPLAY RESULTS
                # --------------------------------------

                st.markdown("---")

                st.subheader("Prediction Results")

                col1, col2 = st.columns(2)

                col1.metric(

                    "Predicted Charging Demand",

                    f"{float(predicted_demand):,.2f}"

                )

                col2.metric(

                    "Predicted Demand Category",

                    str(predicted_category)

                )


                # Classification probability

                if hasattr(
                    classification_model,
                    "predict_proba"
                ):

                    probabilities = (
                        classification_model.predict_proba(
                            classification_input
                        )[0]
                    )

                    classes = classification_model.classes_

                    probability_df = pd.DataFrame({

                        "Demand Category": [
                            category_labels.get(int(category), str(category))
                            for category in classes
                        ],

                        "Probability": probabilities

                    })

                    fig = px.bar(

                        probability_df,

                        x="Demand Category",

                        y="Probability",

                        title="Classification Prediction Probability",

                        text_auto=".2%"

                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )


            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )

                st.info(
                    "Check that the model was trained using "
                    "the same feature columns and preprocessing "
                    "as the featured dataset."
                )


# ============================================================
# TAB 4: TIME SERIES ANALYSIS AND FORECASTING
# ============================================================

with tab4:

    st.header("Time Series Analysis and Demand Forecasting")


    # Historical daily demand

    historical_demand = (

        filtered_df.set_index("StartDatetime")

        .resample("D")["Demand"]

        .sum()

        .reset_index()

    )

    historical_demand.columns = [
        "Date",
        "Demand"
    ]


    st.subheader("Historical Demand")

    fig = px.line(

        historical_demand,

        x="Date",

        y="Demand",

        title="Historical Daily Charging Demand",

        markers=True

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Rolling average

    historical_demand["7_Day_Moving_Average"] = (

        historical_demand["Demand"]

        .rolling(window=7)

        .mean()

    )

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=historical_demand["Date"],

            y=historical_demand["Demand"],

            mode="lines",

            name="Actual Demand"

        )

    )

    fig.add_trace(

        go.Scatter(

            x=historical_demand["Date"],

            y=historical_demand["7_Day_Moving_Average"],

            mode="lines",

            name="7-Day Moving Average"

        )

    )

    fig.update_layout(
        title="Demand with 7-Day Moving Average"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Forecast file

    st.subheader("Future Demand Forecast")

    forecast_path = RESULT_DIR / "future_demand_forecast.csv"

    if forecast_path.exists():

        forecast_df = pd.read_csv(
            forecast_path
        )

        st.dataframe(
            forecast_df,
            use_container_width=True
        )


        # Detect forecast column names

        date_col = next(

            (
                col for col in forecast_df.columns

                if "date" in col.lower()
            ),

            None

        )

        prediction_col = next(

            (
                col for col in forecast_df.columns

                if "predict" in col.lower()
                or "forecast" in col.lower()
            ),

            None

        )


        if date_col and prediction_col:

            forecast_df[date_col] = pd.to_datetime(
                forecast_df[date_col],
                errors="coerce"
            )

            fig = go.Figure()


            fig.add_trace(

                go.Scatter(

                    x=historical_demand["Date"],

                    y=historical_demand["Demand"],

                    mode="lines",

                    name="Historical Demand"

                )

            )


            fig.add_trace(

                go.Scatter(

                    x=forecast_df[date_col],

                    y=forecast_df[prediction_col],

                    mode="lines",

                    name="Forecast",

                    line=dict(dash="dash")

                )

            )


            fig.update_layout(

                title="Historical Demand vs Future Forecast",

                xaxis_title="Date",

                yaxis_title="Demand"

            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.warning(
            "Future forecast CSV not found. "
            "Run Notebook 05 first."
        )


    # Time-series model comparison

    ts_results = read_result(
        "time_series_model_comparison.csv"
    )

    if ts_results is not None:

        st.subheader("Time Series Model Comparison")

        st.dataframe(
            ts_results,
            use_container_width=True
        )


# ============================================================
# TAB 5: MODEL PERFORMANCE COMPARISON
# ============================================================

with tab5:

    st.header("Machine Learning Model Performance")


    # --------------------------------------
    # CLASSIFICATION RESULTS
    # --------------------------------------

    st.subheader("Classification Model Comparison")

    classification_results = read_result(
        "classification_model_comparison.csv"
    )


    if classification_results is not None:

        st.dataframe(
            classification_results,
            use_container_width=True
        )

        # Automatically identify metric columns

        model_col = next(

            (
                c for c in classification_results.columns

                if c.lower() == "model"
            ),

            classification_results.columns[0]

        )

        metric_columns = [

            c for c in classification_results.columns

            if c.lower() in [
                "accuracy",
                "precision",
                "recall",
                "f1 score",
                "f1_score"
            ]

        ]

        if metric_columns:

            fig = px.bar(

                classification_results,

                x=model_col,

                y=metric_columns,

                barmode="group",

                title="Classification Performance"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.warning(
            "Classification comparison file not found."
        )


    st.markdown("---")


    # --------------------------------------
    # REGRESSION RESULTS
    # --------------------------------------

    st.subheader("Regression Model Comparison")

    regression_results = read_result(
        "regression_model_comparison.csv"
    )


    if regression_results is not None:

        st.dataframe(
            regression_results,
            use_container_width=True
        )

        model_col = next(

            (
                c for c in regression_results.columns

                if c.lower() == "model"
            ),

            regression_results.columns[0]

        )

        r2_col = next(

            (
                c for c in regression_results.columns

                if "r2" in c.lower()
            ),

            None

        )

        if r2_col:

            fig = px.bar(

                regression_results,

                x=model_col,

                y=r2_col,

                title="Regression R² Comparison",

                color=r2_col

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.warning(
            "Regression comparison file not found."
        )


    st.markdown("---")


    # --------------------------------------
    # TIME SERIES RESULTS
    # --------------------------------------

    st.subheader("Time Series Model Comparison")

    time_series_results = read_result(
        "time_series_model_comparison.csv"
    )


    if time_series_results is not None:

        st.dataframe(
            time_series_results,
            use_container_width=True
        )

    else:

        st.warning(
            "Time series comparison file not found."
        )


# ============================================================
# TAB 6: MODEL EXPLAINABILITY
# ============================================================

with tab6:

    st.header("Model Explainability")

    st.write(
        "Understand which features influence the model's "
        "predictions using feature importance, SHAP and LIME."
    )


    # --------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------

    st.subheader("Feature Importance")


    importance_csv = (
        EXPLAIN_DIR /
        "feature_importance.csv"
    )

    importance_png = (
        EXPLAIN_DIR /
        "feature_importance.png"
    )


    if importance_csv.exists():

        importance_df = pd.read_csv(
            importance_csv
        )

        st.dataframe(
            importance_df,
            use_container_width=True
        )


    if importance_png.exists():

        st.image(

            str(importance_png),

            caption="Model Feature Importance",

            use_container_width=True

        )


    st.markdown("---")


    # --------------------------------------
    # SHAP
    # --------------------------------------

    st.subheader("SHAP Explainability")


    shap_summary = (
        EXPLAIN_DIR /
        "shap_feature_summary.csv"
    )

    shap_bar = (
        EXPLAIN_DIR /
        "shap_bar_plot.png"
    )

    shap_beeswarm = (
        EXPLAIN_DIR /
        "shap_beeswarm_plot.png"
    )

    shap_dependence = (
        EXPLAIN_DIR /
        "shap_dependence_plot.png"
    )


    if shap_summary.exists():

        shap_df = pd.read_csv(
            shap_summary
        )

        st.write(
            "SHAP Feature Summary"
        )

        st.dataframe(
            shap_df,
            use_container_width=True
        )


        if (
            "Feature" in shap_df.columns

            and "Mean_Absolute_SHAP" in shap_df.columns
        ):

            top_shap = (

                shap_df.sort_values(

                    "Mean_Absolute_SHAP",

                    ascending=False

                ).head(15)

            )

            fig = px.bar(

                top_shap.sort_values(
                    "Mean_Absolute_SHAP"
                ),

                x="Mean_Absolute_SHAP",

                y="Feature",

                orientation="h",

                title="Top 15 SHAP Features"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    if shap_bar.exists():

        st.markdown("**SHAP Global Feature Importance**")

        st.image(
            str(shap_bar),
            use_container_width=True
        )


    if shap_beeswarm.exists():

        st.markdown("**SHAP Summary Plot**")

        st.image(
            str(shap_beeswarm),
            use_container_width=True
        )


    if shap_dependence.exists():

        st.markdown("**SHAP Dependence Plot**")

        st.image(
            str(shap_dependence),
            use_container_width=True
        )


    st.markdown("---")


    # --------------------------------------
    # LIME
    # --------------------------------------

    st.subheader("LIME Explainability")


    lime_png = (
        EXPLAIN_DIR /
        "lime_explanation.png"
    )

    lime_html = (
        EXPLAIN_DIR /
        "lime_explanation.html"
    )


    if lime_png.exists():

        st.image(

            str(lime_png),

            caption="LIME Local Explanation",

            use_container_width=True

        )


    if lime_html.exists():

        with open(
            lime_html,
            "rb"
        ) as file:

            st.download_button(

                label="Download Interactive LIME Explanation",

                data=file,

                file_name="lime_explanation.html",

                mime="text/html"

            )


    if not any([

        importance_csv.exists(),

        importance_png.exists(),

        shap_summary.exists(),

        shap_bar.exists(),

        shap_beeswarm.exists(),

        shap_dependence.exists(),

        lime_png.exists(),

        lime_html.exists()

    ]):

        st.warning(
            "Explainability results not found. "
            "Run Notebook 06 first."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(

    """
    <div style="text-align:center">

    <h4>EV Charging Demand Prediction & Usage Pattern Analysis</h4>

    <p>Machine Learning | Time Series Forecasting | SHAP | LIME | Streamlit</p>

    </div>
    """,

    unsafe_allow_html=True

)