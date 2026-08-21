import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import shap
from cleaner.detector import detect_issues, explain_outliers, calculate_quality_score
from cleaner.fixer import fix_issues_with_log
from cleaner.rules import evaluate_rules
from cleaner.near_duplicates import detect_near_duplicates
from cleaner.pipeline import export_pipeline, load_pipeline

st.set_page_config(page_title="Data Cleaning Assistant", layout="wide", page_icon="🧹")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4149/4149678.png", width=80)
    st.title(" Data Cleaning Assistant")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("1.  Upload your CSV or Excel")
    st.markdown("2.  Detect issues automatically")
    st.markdown("3.  Check custom validation rules")
    st.markdown("4.  Scan for near-duplicates")
    st.markdown("5.  Explain outliers with SHAP")
    st.markdown("6.  Auto fix & download")
    st.markdown("---")

    st.markdown("###  Load a saved pipeline")
    pipeline_file = st.file_uploader("Pipeline JSON", type=["json"], key="pipeline_upload")
    if pipeline_file is not None:
        try:
            loaded_rules, loaded_threshold = load_pipeline(pipeline_file.read().decode("utf-8"))
            st.session_state["_loaded_rules_text"] = "\n".join(loaded_rules)
            st.session_state["_loaded_threshold"] = loaded_threshold
            st.success("Pipeline loaded — fields below are pre-filled.")
        except Exception as e:
            st.error(f"Could not load pipeline: {e}")

    st.markdown("---")
    st.caption("Built with Scikit-learn + SHAP + Streamlit")

# Header
st.title(" Data Cleaning Assistant")
st.markdown("##### Upload a messy CSV or Excel — we'll detect, explain and fix it automatically.")
st.markdown("---")

uploaded = st.file_uploader(" Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded:
    # File reading — CSV or Excel
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded, index_col=0)
    else:
        df = pd.read_excel(uploaded, index_col=0)

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📋 Total Rows", df.shape[0])
    col2.metric("📊 Total Columns", df.shape[1])
    col3.metric("🟡 Missing Values", int(df.isnull().sum().sum()))
    col4.metric("📁 File Size", f"{uploaded.size / 1024:.1f} KB")

    # Detect issues + quality score
    issues = detect_issues(df)
    score = calculate_quality_score(df, issues)

    # Quality Score card
    st.markdown("---")
    col_s1, col_s2, col_s3 = st.columns([1, 2, 1])
    with col_s2:
        if score >= 80:
            color = "green"
            label = "Good"
        elif score >= 50:
            color = "orange"
            label = "Needs Attention"
        else:
            color = "red"
            label = "Poor"

        st.markdown(f"""
        <div style='text-align:center; padding: 20px; border-radius: 12px; 
        border: 2px solid {color}'>
            <h1 style='color:{color}; font-size: 60px; margin:0'>{score}</h1>
            <h3 style='color:{color}; margin:0'>Data Quality Score / 100</h3>
            <p style='color:gray'>Status: <b>{label}</b></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Raw data
    with st.expander("📊 Preview Raw Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    # Detected Issues
    st.subheader("🔍 Detected Issues")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("**🟡 Missing Values**")
        if issues["missing_values"]:
            st.warning(f"⚠️ {len(issues['missing_values'])} columns affected")
        else:
            st.success("✅ None found!")

    with c2:
        st.markdown("**🔵 Type Issues**")
        if issues["type_issues"]:
            st.warning(f"⚠️ {len(issues['type_issues'])} columns affected")
        else:
            st.success("✅ None found!")

    with c3:
        st.markdown("**🔴 Outliers**")
        if issues["outliers"]["count"] > 0:
            st.warning(f"⚠️ {issues['outliers']['count']} rows detected")
        else:
            st.success("✅ None found!")

    with c4:
        st.markdown("**🟣 Duplicates**")
        if issues["duplicates"]["count"] > 0:
            st.warning(f"⚠️ {issues['duplicates']['count']} rows found")
        else:
            st.success("✅ None found!")

    # Full tables in expanders
    if issues["missing_values"]:
        with st.expander("🟡 View Missing Values Detail"):
            st.dataframe(pd.DataFrame.from_dict(issues["missing_values"],
                         orient="index", columns=["Missing Count"]), use_container_width=True)

    if issues["type_issues"]:
        with st.expander("🔵 View Type Issues Detail"):
            st.dataframe(pd.DataFrame.from_dict(issues["type_issues"],
                         orient="index", columns=["Issue"]), use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Custom validation rules
    # ------------------------------------------------------------------
    st.subheader("🧩 Custom Validation Rules")
    st.caption("One rule per line, referencing column names — e.g. `age > 0` or `price <= 100000`")
    default_rules_text = st.session_state.get("_loaded_rules_text", "")
    rules_text = st.text_area("Rules", value=default_rules_text, height=100, label_visibility="collapsed")

    if st.button("✅ Check Rules"):
        rule_list = [r for r in rules_text.splitlines() if r.strip()]
        if not rule_list:
            st.info("Add at least one rule above, then check again.")
        else:
            rule_results = evaluate_rules(df, rule_list)
            for rule, res in rule_results.items():
                if "error" in res:
                    st.error(f"⚠️ `{rule}` — {res['error']}")
                elif res["violations"] > 0:
                    st.warning(f"⚠️ `{rule}` — {res['violations']} row(s) violate this rule")
                    with st.expander(f"View violating rows — {rule}"):
                        st.dataframe(df.loc[res["index"]], use_container_width=True)
                else:
                    st.success(f"✅ `{rule}` — all rows pass")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Near-duplicate detection
    # ------------------------------------------------------------------
    st.subheader("🧬 Near-Duplicate Detection")
    st.caption("Finds rows that are highly similar but not identical (typos, casing, whitespace).")
    default_threshold = st.session_state.get("_loaded_threshold", 0.9)
    threshold = st.slider("Similarity threshold", 0.70, 0.99, float(default_threshold), 0.01)

    if st.button("🔎 Scan for Near-Duplicates"):
        with st.spinner("Comparing rows..."):
            near_dup_result = detect_near_duplicates(df, threshold=threshold)
        if near_dup_result["count"] > 0:
            st.warning(f"⚠️ {near_dup_result['count']} near-duplicate pair(s) found "
                       f"({near_dup_result['mode']} mode)")
            pair_df = pd.DataFrame(near_dup_result["pairs"], columns=["Row A", "Row B", "Similarity"])
            st.dataframe(pair_df, use_container_width=True)
        else:
            st.success(f"✅ No near-duplicates found ({near_dup_result['mode']} mode)")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Pipeline export
    # ------------------------------------------------------------------
    st.subheader("📦 Cleaning Pipeline")
    st.caption("Save your current validation rules + near-duplicate threshold to replay on another file.")
    current_rules = [r for r in rules_text.splitlines() if r.strip()]
    pipeline_json = export_pipeline(current_rules, threshold)
    st.download_button("⬇️ Download Pipeline (.json)", pipeline_json, "cleaning_pipeline.json",
                       use_container_width=True)

    st.markdown("---")

    # Charts side by side
    st.subheader("📊 Visual Analysis")
    ch1, ch2 = st.columns(2)

    with ch1:
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Count"]
        missing_df = missing_df[missing_df["Missing Count"] > 0]
        if not missing_df.empty:
            fig = px.bar(missing_df, x="Column", y="Missing Count",
                         color="Missing Count", title="Missing Values per Column")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values!")

    with ch2:
        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            col_to_plot = st.selectbox("Select column for boxplot", numeric_df.columns)
            fig2 = px.box(df, y=col_to_plot, title=f"Outlier Boxplot — {col_to_plot}")
            st.plotly_chart(fig2, use_container_width=True)

    # Correlation Heatmap
    st.markdown("---")
    st.subheader("🔗 Correlation Heatmap")
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        fig4 = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix",
            aspect="auto"
        )
        fig4.update_layout(height=500)
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("Values close to 1 or -1 = strong correlation. Close to 0 = weak/no correlation.")
    else:
        st.info("Need at least 2 numeric columns for correlation heatmap.")

    st.markdown("---")

    # SHAP
    st.subheader("🧠 Outlier Explanation (SHAP)")
    if st.button("Explain Outliers with SHAP"):
        with st.spinner("Running SHAP analysis... this may take a moment"):
            shap_values, numeric_df = explain_outliers(df)
            fig3, ax = plt.subplots()
            shap.summary_plot(shap_values, numeric_df, plot_type="bar", show=False)
            st.pyplot(fig3)
            st.caption("Higher SHAP value = that feature contributed more to flagging the row as outlier")

    st.markdown("---")

    # Fix
    st.subheader("🛠️ Auto Fix Dataset")
    if st.button("✨ Auto Fix Issues"):
        cleaned_df, imputation_log = fix_issues_with_log(df)

        st.markdown("### 📈 Improvement Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Missing Values",
                  int(cleaned_df.isnull().sum().sum()),
                  delta=f"-{int(df.isnull().sum().sum())} fixed",
                  delta_color="normal")
        m2.metric("Duplicate Rows",
                  int(cleaned_df.duplicated().sum()),
                  delta=f"-{int(df.duplicated().sum())} removed",
                  delta_color="normal")
        new_score = calculate_quality_score(cleaned_df, detect_issues(cleaned_df))
        m3.metric("Quality Score",
                  f"{new_score}/100",
                  delta=f"+{round(new_score - score, 1)} improvement",
                  delta_color="normal")

        # Imputation strategy log
        with st.expander("🧠 View Smart Imputation Decisions"):
            log_df = pd.DataFrame.from_dict(imputation_log, orient="index", columns=["Strategy Used"])
            log_df.index.name = "Column"

            def color_strategy(val):
                if val == "mean":
                    return "background-color: #d4edda"
                elif val == "median":
                    return "background-color: #fff3cd"
                else:
                    return "background-color: #d1ecf1"

            st.dataframe(log_df.style.applymap(color_strategy), use_container_width=True)
            st.caption("🟢 Mean = normal distribution | 🟡 Median = skewed | 🔵 Mode = categorical")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Before**")
            st.dataframe(df.head(5), use_container_width=True)
        with c2:
            st.markdown("**After**")
            st.dataframe(cleaned_df.head(5), use_container_width=True)

        csv = cleaned_df.to_csv(index=False)
        st.download_button("⬇️ Download Cleaned CSV", csv, "cleaned.csv", use_container_width=True)

else:
    # Empty state
    st.info("👆 Upload a CSV or Excel file to get started!")
    st.image("https://cdn-icons-png.flaticon.com/512/4149/4149678.png", width=150)