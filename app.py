
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "hinglish_reviews.csv"
MODEL_PATH = BASE_DIR / "sentiment_model.joblib"

st.set_page_config(
    page_title="Hinglish Sentiment Studio",
    page_icon="ðŸ“Š",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101326, #17152e);
    color: #f5f5ff;
}
[data-testid="stSidebar"] {
    background-color: #0c1020;
}
h1, h2, h3 {
    color: #c4b5fd;
}
.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
}
[data-testid="stMetric"] {
    background: #20213b;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("ðŸ“Š Sentiment Studio")
st.sidebar.caption("Hinglish Review Analytics")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Analyze Review",
        "Batch Analysis",
        "Aspect Insights",
        "Model Performance",
        "Error Analysis"
    ]
)

st.title("HINGLISH SENTIMENT STUDIO")
st.write("Analyze customer feedback written in Hindi, English, and Hinglish.")
st.divider()

df = None
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)

saved_model = None
if MODEL_PATH.exists():
    saved_model = joblib.load(MODEL_PATH)

if page == "Overview":
    st.subheader("Project Overview")
    c1, c2, c3 = st.columns(3)

    c1.metric("Total Reviews", len(df) if df is not None else 0)
    c2.metric(
        "Sentiment Classes",
        df["sentiment"].nunique() if df is not None and "sentiment" in df else 0
    )
    c3.metric(
        "Model Status",
        "Ready" if saved_model else "Not trained"
    )

    if df is not None and "sentiment" in df:
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Reviews"]
        fig = px.pie(
            counts,
            names="Sentiment",
            values="Reviews",
            title="Sentiment Distribution",
            hole=0.45
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dataset missing. Create hinglish_reviews.csv first.")


elif page == "Analyze Review":
    st.subheader("Analyze a Customer Review")
    st.write(
        "Predict overall sentiment and inspect sentiment for each aspect."
    )

    review = st.text_area(
        "Enter customer review",
        placeholder=(
            "Example: Quality achhi hai, lekin delivery late thi "
            "aur packaging kharab thi..."
        ),
        height=150
    )

    if st.button("Analyze Sentiment"):
        if not review.strip():
            st.warning("Please enter a review first.")
        elif not MODEL_PATH.exists():
            st.error("Model not found. Run python train_model.py first.")
        else:
            try:
                from aspect_analyzer import analyze_review

                result = analyze_review(review)

                st.divider()
                st.subheader("Overall Result")

                col1, col2 = st.columns(2)
                col1.metric(
                    "Overall Sentiment",
                    result["sentiment"].upper()
                )
                col2.metric(
                    "Model Confidence",
                    f'{result["confidence"]:.1%}'
                )

                st.divider()
                st.subheader("Aspect-wise Sentiment")

                aspect_results = result.get("aspect_results", [])

                if aspect_results:
                    aspect_df = pd.DataFrame([
                        {
                            "Aspect": item["aspect_name"],
                            "Sentiment": item["sentiment"].title(),
                            "Evidence": item["evidence"]
                        }
                        for item in aspect_results
                    ])

                    st.dataframe(
                        aspect_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    for item in aspect_results:
                        sentiment = item["sentiment"].lower()

                        if sentiment == "positive":
                            st.success(
                                f'{item["aspect_name"]}: Positive'
                            )
                        elif sentiment == "negative":
                            st.error(
                                f'{item["aspect_name"]}: Negative'
                            )
                        else:
                            st.info(
                                f'{item["aspect_name"]}: Neutral'
                            )
                else:
                    st.info("No aspects were detected.")

                st.caption(
                    "Aspect sentiments use a rule-based baseline. "
                    "They may be inaccurate for complex Hinglish, "
                    "negation, sarcasm, or mixed opinions."
                )

            except Exception as error:
                st.error(f"Analysis failed: {error}")



elif page == "Aspect Insights":
    st.subheader("Aspect Insights")
    st.write(
        "Explore how customers feel about different product aspects."
    )

    if df is not None and {"aspect", "sentiment"}.issubset(df.columns):
        insights_df = df.copy()

        insights_df["aspect"] = (
            insights_df["aspect"].astype(str).str.strip()
        )
        insights_df["sentiment"] = (
            insights_df["sentiment"].astype(str).str.lower().str.strip()
        )

        valid_sentiments = ["positive", "negative", "neutral"]
        insights_df = insights_df[
            insights_df["sentiment"].isin(valid_sentiments)
        ]

        if insights_df.empty:
            st.warning("No valid sentiment records found in the dataset.")
        else:
            total = len(insights_df)
            positive = int((insights_df["sentiment"] == "positive").sum())
            negative = int((insights_df["sentiment"] == "negative").sum())
            neutral = int((insights_df["sentiment"] == "neutral").sum())

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", total)
            col2.metric("Positive", positive)
            col3.metric("Negative", negative)
            col4.metric("Neutral", neutral)

            st.divider()

            aspect_counts = (
                insights_df.groupby(["aspect", "sentiment"])
                .size()
                .reset_index(name="Reviews")
            )

            fig = px.bar(
                aspect_counts,
                x="aspect",
                y="Reviews",
                color="sentiment",
                barmode="group",
                title="Sentiment Distribution by Aspect",
                labels={
                    "aspect": "Product Aspect",
                    "Reviews": "Number of Reviews",
                    "sentiment": "Sentiment"
                },
                color_discrete_map={
                    "positive": "#16a34a",
                    "negative": "#dc2626",
                    "neutral": "#64748b"
                }
            )

            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Aspect Summary")

            summary = (
                insights_df.groupby("aspect")
                .agg(
                    Total_Reviews=("sentiment", "size"),
                    Positive=("sentiment", lambda x: (x == "positive").sum()),
                    Negative=("sentiment", lambda x: (x == "negative").sum()),
                    Neutral=("sentiment", lambda x: (x == "neutral").sum())
                )
                .reset_index()
            )

            summary["Positive %"] = (
                summary["Positive"] / summary["Total_Reviews"] * 100
            ).round(1)

            summary["Negative %"] = (
                summary["Negative"] / summary["Total_Reviews"] * 100
            ).round(1)

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

            st.divider()
            st.subheader("Quick Insights")

            most_reviewed = summary.loc[
                summary["Total_Reviews"].idxmax()
            ]
            st.write(
                f"**Most discussed aspect:** {most_reviewed['aspect']} "
                f"({int(most_reviewed['Total_Reviews'])} records)"
            )

            if summary["Negative %"].max() > 0:
                most_negative = summary.loc[summary["Negative %"].idxmax()]
                st.write(
                    f"**Highest negative share:** {most_negative['aspect']} "
                    f"({most_negative['Negative %']:.1f}%)"
                )

            st.caption(
                "Insights reflect the labels in your dataset. "
                "They are not automatically calculated from newly uploaded "
                "reviews unless those reviews are added to the dataset."
            )

            st.subheader("Dataset Records")
            st.dataframe(
                insights_df,
                use_container_width=True,
                hide_index=True
            )
    else:
        st.warning(
            "Dataset missing or the required 'aspect' and 'sentiment' "
            "columns are unavailable."
        )



elif page == "Model Performance":
    st.subheader("ðŸ“Š Model Performance Dashboard")

    if saved_model:
        accuracy = saved_model.get("accuracy", 0)
        report = saved_model.get("classification_report", {})
        matrix = saved_model.get("confusion_matrix", [])
        total = saved_model.get("total_reviews", 0)
        train_size = saved_model.get("train_size", 0)
        test_size = saved_model.get("test_size", 0)

        st.write(
            "Evaluation results from the held-out test split."
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Test Accuracy", f"{accuracy:.1%}")
        col2.metric("Training Reviews", train_size)
        col3.metric("Testing Reviews", test_size)

        st.caption(
            f"Dataset size: {total} reviews. "
            "Results are preliminary because the dataset is small "
            "and manually written."
        )

        st.divider()
        st.subheader("Classification Metrics")

        metric_rows = []

        for label in ["negative", "neutral", "positive"]:
            values = report.get(label, {})
            metric_rows.append({
                "Sentiment": label.title(),
                "Precision": values.get("precision", 0),
                "Recall": values.get("recall", 0),
                "F1-score": values.get("f1-score", 0),
                "Support": values.get("support", 0)
            })

        metrics_df = pd.DataFrame(metric_rows)

        st.dataframe(
            metrics_df.style.format({
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-score": "{:.2%}",
                "Support": "{:.0f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.subheader("Confusion Matrix")

        if matrix:
            labels = ["Negative", "Neutral", "Positive"]

            matrix_df = pd.DataFrame(
                matrix,
                index=[f"Actual {label}" for label in labels],
                columns=[f"Predicted {label}" for label in labels]
            )

            fig = px.imshow(
                matrix_df,
                text_auto=True,
                aspect="auto",
                title="Actual vs Predicted Sentiment",
                color_continuous_scale="Purples"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Precision measures how often predictions for a class are correct. "
            "Recall measures how many actual examples of that class are found. "
            "F1-score balances precision and recall."
        )

    else:
        st.warning(
            "Model evaluation is unavailable. Run "
            "`python train_model.py` first."
        )
elif page == "Error Analysis":
    st.subheader("Prediction Error Analysis")

    error_path = BASE_DIR / "error_analysis.csv"

    if error_path.exists():
        error_df = pd.read_csv(error_path)

        if error_df.empty:
            st.success("No incorrect predictions in the latest test split.")
        else:
            st.metric("Incorrect Test Predictions", len(error_df))

            st.write(
                "These are reviews where the predicted sentiment "
                "did not match the actual sentiment."
            )

            st.dataframe(
                error_df,
                use_container_width=True,
                hide_index=True
            )

            csv_data = error_df.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                "Download Error Report",
                data=csv_data,
                file_name="error_analysis.csv",
                mime="text/csv"
            )
    else:
        st.warning(
            "Error report not found. Run python train_model.py first."
        )
elif page == "Batch Analysis":
    st.subheader("ðŸ“‚ Batch Review Intelligence")

    st.write(
        "Upload a CSV containing customer reviews to analyze "
        "sentiment for multiple records."
    )

    uploaded_file = st.file_uploader(
        "Upload review CSV",
        type=["csv"],
        help="Your CSV must contain a column named review."
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            review_column = next(
                (
                    column for column in uploaded_df.columns
                    if column.strip().lower() == "review"
                ),
                None
            )

            if review_column is None:
                st.error(
                    "CSV mein 'review' naam ka column hona chahiye."
                )

            elif not MODEL_PATH.exists():
                st.error(
                    "Model missing hai. Pehle python train_model.py run karo."
                )

            else:
                st.write(f"Uploaded rows: {len(uploaded_df)}")

                if len(uploaded_df) == 0:
                    st.warning("CSV file empty hai.")

                elif len(uploaded_df) > 5000:
                    st.warning(
                        "Is demo mein ek upload par maximum 5000 rows "
                        "process karo."
                    )

                elif st.button("Analyze All Reviews"):
                    from aspect_analyzer import analyze_review

                    results = []
                    progress = st.progress(0)
                    status = st.empty()

                    for index, review in enumerate(
                        uploaded_df[review_column].fillna("").astype(str)
                    ):
                        if not review.strip():
                            results.append({
                                "aspect_sentiments": "",
                                "sentiment": "not analyzed",
                                "confidence": None,
                                "aspects": "",
                                "error": "Empty review"

                            })
                        else:
                            try:
                                result = analyze_review(review)


                                results.append({
                                    "sentiment": result["sentiment"],
                                    "confidence": result["confidence"],
                                    "aspects": ", ".join(result["aspects"]),
                                    "aspect_sentiments": "; ".join(
                                    f'{item["aspect_name"]}: {item["sentiment"]}'
                                    for item in result.get("aspect_results", [])
                                ),
                                    "error": ""


                                })

                            except Exception as error:
                                results.append({
                                    "sentiment": "error",
                                    "confidence": None,
                                    "aspects": "",
                                    "error": str(error)
                                })

                        progress.progress((index + 1) / len(uploaded_df))
                        status.text(
                            f"Analyzed {index + 1} of {len(uploaded_df)} reviews"
                        )

                    results_df = pd.concat(
                        [
                            uploaded_df.reset_index(drop=True),
                            pd.DataFrame(results)
                        ],
                        axis=1
                    )

                    status.success("Batch analysis complete!")

                    valid_results = results_df[
                        results_df["sentiment"].isin(
                            ["positive", "negative", "neutral"]
                        )
                    ]

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Reviews", len(results_df))
                    col2.metric("Successfully Analyzed", len(valid_results))
                    col3.metric(
                        "Errors / Skipped",
                        len(results_df) - len(valid_results)
                    )

                    if not valid_results.empty:
                        sentiment_counts = (
                            valid_results["sentiment"]
                            .value_counts()
                            .rename_axis("Sentiment")
                            .reset_index(name="Reviews")
                        )

                        fig = px.bar(
                            sentiment_counts,
                            x="Sentiment",
                            y="Reviews",
                            color="Sentiment",
                            title="Batch Sentiment Distribution"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    st.subheader("Analysis Results")
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    csv_data = results_df.to_csv(
                        index=False
                    ).encode("utf-8-sig")

                    st.download_button(
                        "Download Analysis CSV",
                        data=csv_data,
                        file_name="hinglish_sentiment_results.csv",
                        mime="text/csv"
                    )

        except Exception as error:
            st.error(f"Could not read CSV: {error}")

