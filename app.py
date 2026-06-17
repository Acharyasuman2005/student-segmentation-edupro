import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(page_title="EduPro Learner Intelligence", layout="wide")

# ----------------------------------------------------------------------------
# LOAD SAVED ARTIFACTS (cached so they only load once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le_gender = joblib.load("le_gender.pkl")
    le_category = joblib.load("le_category.pkl")
    le_level = joblib.load("le_level.pkl")
    return kmeans, scaler, le_gender, le_category, le_level


@st.cache_data
def load_data():
    learner = pd.read_csv("learner_profiles.csv")
    course_cluster_pop = pd.read_csv("course_cluster_popularity.csv")
    courses = pd.read_csv("courses_clean.csv")
    user_courses = pd.read_csv("user_courses.csv")
    return learner, course_cluster_pop, courses, user_courses


kmeans, scaler, le_gender, le_category, le_level = load_model_artifacts()
learner, course_cluster_pop, courses, user_courses = load_data()

# Human-readable names for each cluster, based on the behavioral analysis done in Colab.
# If you re-run KMeans with different data, double check these mappings still match
# by comparing against the cluster_summary table from Step 6.
CLUSTER_NAMES = {
    0: "Niche Specialists",
    1: "Power Learners",
    2: "Premium One-Time Buyers",
    3: "Casual Beginners",
}
learner["SegmentName"] = learner["Cluster"].map(CLUSTER_NAMES)

# ----------------------------------------------------------------------------
# RECOMMENDATION LOGIC (mirrors Step 7 from Colab)
# ----------------------------------------------------------------------------
def recommend_courses(user_id, top_n=5, category_filter=None, level_filter=None):
    cluster = learner.loc[learner["UserID"] == user_id, "Cluster"].values[0]
    already_taken = set(user_courses.loc[user_courses["UserID"] == user_id, "CourseID"])

    candidates = course_cluster_pop[course_cluster_pop["Cluster"] == cluster]
    candidates = candidates[~candidates["CourseID"].isin(already_taken)]
    candidates = candidates.merge(courses, on="CourseID")

    if category_filter and category_filter != "All":
        candidates = candidates[candidates["CourseCategory"] == category_filter]
    if level_filter and level_filter != "All":
        candidates = candidates[candidates["CourseLevel"] == level_filter]

    candidates = candidates.sort_values("Score", ascending=False).head(top_n)
    return candidates[["CourseID", "CourseName", "CourseCategory", "CourseLevel", "CourseRating", "Score"]]


# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
st.sidebar.title("EduPro Dashboard")
page = st.sidebar.radio(
    "Go to",
    ["Learner Profile Explorer", "Cluster Dashboard", "Recommendations", "Segment Comparison"],
)

# ----------------------------------------------------------------------------
# PAGE 1: LEARNER PROFILE EXPLORER
# ----------------------------------------------------------------------------
if page == "Learner Profile Explorer":
    st.title("Learner Profile Explorer")
    st.write("Select a learner to see their full behavioral profile and assigned segment.")

    user_id = st.selectbox("Select Learner (UserID)", learner["UserID"].sort_values())
    profile = learner[learner["UserID"] == user_id].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Segment", profile["SegmentName"])
    col2.metric("Total Courses Enrolled", int(profile["TotalCoursesEnrolled"]))
    col3.metric("Avg Spending", f"₹{profile['AvgSpending']:.2f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Diversity Score", int(profile["DiversityScore"]))
    col5.metric("Avg Course Rating", f"{profile['AvgCourseRating']:.2f}")
    col6.metric("Learning Depth Index", f"{profile['LearningDepthIndex']:.2f}")

    st.divider()
    st.subheader("Full Profile Data")
    st.dataframe(profile.to_frame().T, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE 2: CLUSTER VISUALIZATION DASHBOARD
# ----------------------------------------------------------------------------
elif page == "Cluster Dashboard":
    st.title("Cluster Visualization Dashboard")
    st.write("Overview of all learner segments discovered through K-Means clustering.")

    cluster_sizes = learner["SegmentName"].value_counts().reset_index()
    cluster_sizes.columns = ["Segment", "Count"]

    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(cluster_sizes, names="Segment", values="Count", title="Segment Sizes")
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        fig_bar = px.bar(cluster_sizes, x="Segment", y="Count", title="Segment Sizes (Bar)")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Segment Behavior Summary")
    summary = learner.groupby("SegmentName").agg(
        AvgAge=("Age", "mean"),
        AvgCoursesEnrolled=("TotalCoursesEnrolled", "mean"),
        AvgSpending=("AvgSpending", "mean"),
        AvgRating=("AvgCourseRating", "mean"),
        AvgDiversity=("DiversityScore", "mean"),
        AvgDepthIndex=("LearningDepthIndex", "mean"),
    ).round(2)
    st.dataframe(summary, use_container_width=True)

    st.subheader("Spending vs Engagement by Segment")
    fig_scatter = px.scatter(
        learner, x="TotalCoursesEnrolled", y="AvgSpending", color="SegmentName",
        hover_data=["UserID"], title="Courses Enrolled vs Avg Spending"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE 3: PERSONALIZED RECOMMENDATIONS
# ----------------------------------------------------------------------------
elif page == "Recommendations":
    st.title("Personalized Course Recommendations")
    st.write("Get course recommendations tailored to a learner's segment, filterable by category or level.")

    user_id = st.selectbox("Select Learner (UserID)", learner["UserID"].sort_values(), key="rec_user")
    segment = learner.loc[learner["UserID"] == user_id, "SegmentName"].values[0]
    st.info(f"This learner belongs to segment: **{segment}**")

    col1, col2 = st.columns(2)
    with col1:
        category_filter = st.selectbox("Filter by Category", ["All"] + sorted(courses["CourseCategory"].unique().tolist()))
    with col2:
        level_filter = st.selectbox("Filter by Level", ["All"] + sorted(courses["CourseLevel"].unique().tolist()))

    top_n = st.slider("Number of recommendations", 1, 10, 5)

    recs = recommend_courses(user_id, top_n=top_n, category_filter=category_filter, level_filter=level_filter)

    if recs.empty:
        st.warning("No matching courses found for these filters. Try widening the category/level filter.")
    else:
        st.dataframe(recs, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE 4: SEGMENT COMPARISON PANEL
# ----------------------------------------------------------------------------
elif page == "Segment Comparison":
    st.title("Segment Comparison Panel")
    st.write("Compare two learner segments side by side.")

    segments = sorted(learner["SegmentName"].unique())
    col1, col2 = st.columns(2)
    with col1:
        seg_a = st.selectbox("Segment A", segments, index=0)
    with col2:
        seg_b = st.selectbox("Segment B", segments, index=1 if len(segments) > 1 else 0)

    metrics = ["Age", "TotalCoursesEnrolled", "AvgSpending", "AvgCourseRating", "DiversityScore", "LearningDepthIndex"]
    compare = learner.groupby("SegmentName")[metrics].mean().round(2)
    compare_subset = compare.loc[[seg_a, seg_b]].T
    compare_subset.columns = [seg_a, seg_b]

    st.subheader(f"{seg_a} vs {seg_b}")
    st.dataframe(compare_subset, use_container_width=True)

    fig = px.bar(
        compare_subset.reset_index().melt(id_vars="index", var_name="Segment", value_name="Value"),
        x="index", y="Value", color="Segment", barmode="group",
        title="Metric-by-Metric Comparison", labels={"index": "Metric"}
    )
    st.plotly_chart(fig, use_container_width=True)
