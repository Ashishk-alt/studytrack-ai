import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Initialize login state
#if "logged_in" not in st.session_state:
 #   st.session_state.logged_in = False

# User credentials 
#USERS = {
 #  "admin@gmail.com": "admin@123",
  #  "rahul@321": "rahul321#",
   # "bob123@gmail.com": "bob123#"
#}

#def login_page():
 #   st.title("🔐 Login to StudyTrack AI")

  #  username = st.text_input("Username")
   # password = st.text_input("Password", type="password")

    #if st.button("Login"):
     #   if username in USERS and USERS[username] == password:
      #      st.session_state.logged_in = True
       #     st.session_state.username = username
        #    st.success("✅ Login successful!")
         #   st.rerun()
        #else:
         #   st.error("❌ Invalid username or password")

# If not logged in → show login page ONLY
#if not st.session_state.logged_in:
 #   login_page()
  #  st.stop()

# STOP DASHBOARD IF NOT LOGGED IN
#if not st.session_state.logged_in:
 #   login_page()
  #  st.stop()

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="StudyTrack-AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------
# GLOBAL CSS (UNCHANGED)
# ------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-left: 4rem;
    padding-right: 1.5rem;
}
.main-title {
    margin-top: 10px;
    padding-left: 11rem;
    font-size: 44px;
    font-weight: 700;
    color: white;
    margin-bottom: 4px;
}
.sub-title {
    font-size: 18px;
    padding-left: 10rem;
    color: #c7cbd1;
    margin-top: 0px;
    margin-bottom: 16px;
}
.section-title {
    font-size: 30px;
    font-weight: 600;
    margin-top: 8px;
    margin-bottom: 8px;
}
hr {
    margin-top: 10px;
    margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# SIDEBAR NAVIGATION (UNCHANGED)
# ------------------------------------------------
st.sidebar.title("📘 Navigation")
menu = st.sidebar.radio(
    "",
    ["🏠 Home", "🧠 Model Training", "📊 Data Insights", "🎓 Student", "📈 Recommendation", "📄 Documentation"]
)

#if st.sidebar.button("🚪 Logout"):
 #   st.session_state.logged_in = False
  #  st.rerun()

# ------------------------------------------------
# HOME PAGE (UNCHANGED)
# ------------------------------------------------
if menu == "🏠 Home":

    col1, col2 = st.columns([0.7, 9.3])

    with col2:
        st.markdown("<div class='main-title'> 🚀StudyTrack AI </div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sub-title'>Tracking, Predicting, and Improving Student Performance</div>",
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown("<div class='section-title'>Project Overview</div>", unsafe_allow_html=True)

    st.write("""
    **StudyTrack AI – Personal Dashboard** analyzes student academic data using multiple
    behavioral and lifestyle parameters to generate insights, predictions, and
    personalized recommendations.
    The system supports CSV uploads, interactive dashboards, and AI-based performance prediction to help students
     make data-driven decisions.
    """)

    st.image(
        "Images.png",
        use_container_width=True
    )

    st.markdown("<div class='section-title'>Project Objectives</div>", unsafe_allow_html=True)

    st.markdown("""
    - Analyze student performance using multiple parameters  
    - Predict academic outcomes  
    - Provide personalized recommendations  
    - Visualize academic trends  
    - Enable data-driven decision making  
    """)

# ------------------------------------------------
# MODEL TRAINING (ONLY TEXT CLARIFIED)
# ------------------------------------------------
elif menu == "🧠 Model Training":
    st.title("🧠 Model Training")

    file = st.file_uploader("Upload Student Dataset (CSV or Excel)", type=["csv", "xlsx"])

    if file:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        if st.button("Train Model"):
            # ---------------- DATA PREPROCESSING ----------------
            df = df.drop_duplicates()

            # Select numeric columns
            numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

            # Convert all numeric columns safely
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Handle missing values
            if df[numeric_cols].isnull().sum().sum() > 0:
                st.warning("⚠️ Missing values found. Filling with column mean.")
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

            df = df.reset_index(drop=True)    

            # ---------- REQUIRED COLUMNS ----------
            required_cols = [
                "Study_Hours",
                "Sleep_Hours",
                "Attendance_Percentage",
                "Attention_Level",
                "Previous_Marks",
                "Final_Marks"
            ]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.fillna(df.mean(numeric_only=True))
            # ---------------------------------------------------    

            required_cols = [
                "Study_Hours",
                "Sleep_Hours",
                "Attendance_Percentage",
                "Attention_Level",
                "Previous_Marks",
                "Final_Marks"
            ]

            if not all(col in df.columns for col in required_cols):
                st.error("❌ Dataset missing required columns!")
                st.stop()

            # ---------------- MODEL 1 (WITH PREVIOUS MARKS) ----------------
            X_full = df[["Study_Hours", "Sleep_Hours", "Attendance_Percentage", "Attention_Level", "Previous_Marks"]]
            y = df["Final_Marks"]

            model_full = LinearRegression()
            model_full.fit(X_full, y)

            # ---------------- MODEL 2 (WITHOUT PREVIOUS MARKS) ----------------
            X_habit = df[["Study_Hours", "Sleep_Hours", "Attendance_Percentage", "Attention_Level"]]

            scaler = StandardScaler()
            X_habit_scaled = scaler.fit_transform(X_habit)

            model_habit = LinearRegression()
            model_habit.fit(X_habit_scaled, y)

            # ---------------- PREDICTIONS (FOR INSIGHTS) ----------------
            df["Predicted_Marks"] = model_full.predict(X_full)

            # ---------------- PERFORMANCE LEVEL ----------------
            def performance_level(marks):
                if marks >= 85:
                    return "Excellent Performer"
                elif marks >= 70:
                    return "Good Performer"
                elif marks >= 55:
                    return "Average Performer"
                else:
                    return "Needs Improvement"

            df["Performance_Level"] = df["Predicted_Marks"].apply(performance_level)

            # ---------------- STORE MODELS & DATA ----------------
            st.session_state["model_full"] = model_full
            st.session_state["model_habit"] = model_habit
            st.session_state["scaler_habit"] = scaler
            st.session_state["trained_data"] = df

            st.success("✅ Both models trained successfully!")

            # ---------------- EVALUATION ----------------
            y_pred = model_full.predict(X_full)
            r2 = r2_score(y, y_pred)

            st.markdown("### 📊 Model Performance (With Previous Marks)")
            st.write(f"**R² Score:** {r2 * 100:.2f}%")


            
# ------------------------------------------------
# STUDENT PAGE (MULTI-PARAMETERS ADDED HERE)
# ------------------------------------------------
elif menu == "🎓 Student":
    st.title("🎓 Student Analysis")

    # ---------- CHECK MODEL ----------
    if "model_habit" not in st.session_state:
        st.warning("⚠️ Please train the model first in Model Training section.")
        st.stop()

    model_habit = st.session_state["model_habit"]

    # =====================================================
    # 🔹 PART 1: INDIVIDUAL STUDENT PREDICTION
    # =====================================================
    st.subheader("🧍 Individual Student Prediction")

    name = st.text_input("Student Name")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider("📘 Study Hours", 0, 12, 5)
        sleep_hours = st.slider("😴 Sleep Hours", 0, 10, 7)

    with col2:
        attendance = st.slider("🏫 Attendance Percentage", 0, 100, 80)
        attention = st.slider("🧠 Attention Level", 0, 100, 70)

    if st.button("Analyze Individual Student"):

        # -------- ML Prediction --------
        input_data = [[study_hours, sleep_hours, attendance, attention]]

        scaler = st.session_state["scaler_habit"]
        input_scaled = scaler.transform(input_data)

        predicted_marks = model_habit.predict(input_scaled)[0]
        # Clamp between 0 and 100
        predicted_marks = max(0, min(100, predicted_marks))


        # -------- Performance Level --------
        if predicted_marks >= 85:
            performance = "Excellent Performer"
        elif predicted_marks >= 70:
            performance = "Good Performer"
        elif predicted_marks >= 55:
            performance = "Average Performer"
        else:
            performance = "Needs Improvement"

        # -------- Recommendation Logic --------
        def generate_recommendation_individual():
            rec = []

            if performance == "Excellent Performer":
                rec.append("Maintain current study routine")
                if attention < 70:
                    rec.append("Improve focus consistency")
                if sleep_hours < 7:
                    rec.append("Ensure adequate sleep")

            elif performance == "Average Performer":
                rec.append("Increase academic consistency")
                if study_hours < 6:
                    rec.append("Increase study hours")
                if attention < 65:
                    rec.append("Reduce distractions and improve focus")
                if attendance < 80:
                    rec.append("Improve class attendance")

            else:
                rec.append("Immediate academic intervention required")
                if study_hours < 6:
                    rec.append("Significantly increase study hours")
                if sleep_hours < 7:
                    rec.append("Improve sleep routine")
                if attention < 60:
                    rec.append("Work on concentration techniques")
                if attendance < 75:
                    rec.append("Attend classes regularly")

            return " | ".join(rec)

        recommendation = generate_recommendation_individual()

        # -------- OUTPUT --------
        st.success(f"Analysis completed for {name}")
        st.markdown("### 📊 Prediction Result")
        st.write(f"**Predicted Marks:** {predicted_marks:.2f}")
        st.write(f"**Performance Level:** {performance}")
        st.write(f"**Recommendation:** {recommendation}")

    # =====================================================
    # 🔹 PART 2: BULK STUDENT PREDICTION (CSV UPLOAD)
    # =====================================================
    st.divider()
    st.subheader("📂 Bulk Student Prediction (CSV Upload)")

    bulk_file = st.file_uploader("Upload Student CSV for Bulk Prediction", type=["csv", "xlsx"])

    if bulk_file:
        if bulk_file.name.endswith(".csv"):
            bulk_df = pd.read_csv(bulk_file)
        else:
            bulk_df = pd.read_excel(bulk_file)    

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(bulk_df.head())

        required_cols = [
            "Student_ID",
            "Student_Name",
            "Study_Hours",
            "Sleep_Hours",
            "Attendance_Percentage",
            "Attention_Level"
        ]

        if not all(col in bulk_df.columns for col in required_cols):
            st.error("❌ CSV missing required columns!")
            st.stop()

        if st.button("Predict Bulk Data"):

            X_bulk = bulk_df[
                ["Study_Hours", "Sleep_Hours", "Attendance_Percentage", "Attention_Level"]
            ]

            scaler = st.session_state["scaler_habit"]
            X_bulk_scaled = scaler.transform(X_bulk)

            bulk_df["Predicted_Marks"] = model_habit.predict(X_bulk_scaled)
            # Clamp between 0 and 100
            bulk_df["Predicted_Marks"] = bulk_df["Predicted_Marks"].apply(lambda x: max(0, min(100, x))).round(2)

            # -------- Performance Level --------
            def perf_level(marks):
                if marks >= 85:
                    return "Excellent Performer"
                elif marks >= 70:
                    return "Good Performer"
                elif marks >= 55:
                    return "Average Performer"
                else:
                    return "Needs Improvement"

            bulk_df["Performance_Level"] = bulk_df["Predicted_Marks"].apply(perf_level)

            # -------- Recommendation --------
            def generate_recommendation(row):
                rec = []

                if row["Performance_Level"] == "Excellent Performer":
                    rec.append("Maintain current study routine")
                    if row["Attention_Level"] < 70:
                        rec.append("Improve focus consistency")
                    if row["Sleep_Hours"] < 7:
                        rec.append("Ensure adequate sleep")

                elif row["Performance_Level"] == "Average Performer":
                    rec.append("Increase academic consistency")
                    if row["Study_Hours"] < 6:
                        rec.append("Increase study hours")
                    if row["Attention_Level"] < 65:
                        rec.append("Reduce distractions and improve focus")
                    if row["Attendance_Percentage"] < 80:
                        rec.append("Improve class attendance")

                else:
                    rec.append("Immediate academic intervention required")
                    if row["Study_Hours"] < 6:
                        rec.append("Significantly increase study hours")
                    if row["Sleep_Hours"] < 7:
                        rec.append("Improve sleep routine")
                    if row["Attention_Level"] < 60:
                        rec.append("Work on concentration techniques")
                    if row["Attendance_Percentage"] < 75:
                        rec.append("Attend classes regularly")

                return " | ".join(rec)

            bulk_df["Recommendation"] = bulk_df.apply(generate_recommendation, axis=1)

            st.success("✅ Bulk prediction & recommendation completed successfully")

            st.subheader("📊 Bulk Prediction Results")
            st.dataframe(bulk_df, use_container_width=True)

            csv = bulk_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Bulk Prediction Result",
                csv,
                "bulk_student_predictions.csv",
                "text/csv"
            )
# DATA INSIGHTS
# ------------------------------------------------
elif menu == "📊 Data Insights":
    st.title("📊 Data Insights Dashboard")

    # CHECK IF MODEL IS TRAINED
    if "trained_data" not in st.session_state:
        st.warning("⚠️ Please upload data and train the model first.")
    else:
        df = st.session_state["trained_data"]

        st.success("✅ Data loaded successfully for insights!")

        # -----------------------------
        # DATA PREVIEW
        # -----------------------------
        st.subheader("📄 Trained Data")
        st.dataframe(df)
        st.subheader("⬇️ Download Trained Model Data")

        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
           label="📥 Download Trained Dataset (CSV)",
           data=csv_data,
           file_name="studytrack_trained_data.csv",
           mime="text/csv"
        )

        # -----------------------------
        # 1️⃣ Study Hours vs Predicted Marks (FIXED)
        # -----------------------------
        st.subheader("🎯 Study Hours vs Predicted Marks")

        fig1 = px.scatter(
            df,
            x="Study_Hours",
            y="Predicted_Marks",
            color="Performance_Level",
            size="Attendance_Percentage"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # -----------------------------
        # 2️⃣ Average Predicted Marks by Study Hours
        # -----------------------------
        st.subheader("📊 Average Predicted Marks by Study Hours")

        avg_df = df.groupby("Study_Hours", as_index=False)["Predicted_Marks"].mean()

        fig2 = px.bar(
            avg_df,
            x="Study_Hours",
            y="Predicted_Marks",
            color="Predicted_Marks"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # -----------------------------
        # 3️⃣ Performance Level Distribution (PIE CHART)
        # -----------------------------
        st.subheader("🥧 Performance Level Distribution")

        perf_df = df["Performance_Level"].value_counts().reset_index()
        perf_df.columns = ["Performance_Level", "Count"]

        fig3 = px.pie(
            perf_df,
            names="Performance_Level",
            values="Count",
            title="Distribution of Student Performance Levels"
        )
        st.plotly_chart(fig3, use_container_width=True)

        # -----------------------------
        # 4️⃣ Correlation Heatmap
        # -----------------------------
        st.subheader("🔥 Correlation Heatmap")

        numeric_df = df.select_dtypes(include="number")
        corr = numeric_df.corr()

        fig4 = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig4, use_container_width=True)

        # -----------------------------
        # 5️⃣ Actual vs Predicted Marks
        # -----------------------------
        st.subheader("📈 Actual vs Predicted Marks")

        # Check required columns
        required_cols = ["Final_Marks", "Predicted_Marks"]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
           st.warning("⚠️ Actual marks not available to compare.")
        else:
           compare_df = df[["Final_Marks", "Predicted_Marks"]].copy()
           compare_df.columns = ["Actual Marks", "Predicted Marks"]

        # Show table
           st.dataframe(compare_df, use_container_width=True)

        # Line chart
           fig5 = px.line(
               compare_df,
               y=["Actual Marks", "Predicted Marks"],
               markers=True,
               title="Actual vs Predicted Marks Comparison"
            )
        st.plotly_chart(fig5, use_container_width=True)

        st.divider()
        st.subheader("💡 Key Insights Summary")

        avg_study = df["Study_Hours"].mean()
        avg_marks = df["Predicted_Marks"].mean()
        avg_attention = df["Attention_Level"].mean()

        topper = df.loc[df["Predicted_Marks"].idxmax(), "Student_Name"]

        st.markdown(f"""
        - 📘 **Average Study Hours:** {avg_study:.2f} hrs/day  
        - 🎯 **Average Predicted Marks:** {avg_marks:.2f}%  
        - 🏆 **Top Performing Student:** {topper}  
        - 🧠 **Average Attention Level:** {avg_attention:.2f}  
        """)

        #Clustering-------------------------
        st.divider()
        st.subheader("📌 Student Clusters based on Performance Level (0–3)")
        # 1️⃣ Map Performance Level to Cluster IDs
        cluster_map = {
            "Needs Improvement": 0,
            "Average Performer": 1,
            "Good Performer": 2,
            "Excellent Performer": 3
        }
        df["Cluster_ID"] = df["Performance_Level"].map(cluster_map)

        # 2️⃣ Create Cluster Distribution Table
        cluster_count = df["Cluster_ID"].value_counts().sort_index().reset_index()
        cluster_count.columns = ["Cluster_ID", "Student_Count"]
        
        st.subheader("🔢 Cluster Distribution Table")
        st.dataframe(cluster_count, use_container_width=True)

        # 3️⃣ Bar Chart for Clusters
        fig_cluster = px.bar(
            cluster_count,
            x="Cluster_ID",
            y="Student_Count",
            text="Student_Count",
            color="Cluster_ID",
            title="Student Clusters based on Performance Level (0 = Needs Improvement, 3 = Excellent)"
        )

        fig_cluster.update_layout(
            xaxis_title="Cluster ID (0=Needs Improvement, 1=Average, 2=Good, 3=Excellent)",
            yaxis_title="Number of Students",
            showlegend=False
        )

        st.plotly_chart(fig_cluster, use_container_width=True)

        
# ------------------------------------------------
# RECOMMENDATION PAGE (LOGIC BASED ON PARAMETERS)

elif menu == "📈 Recommendation":
    st.title("📄 Trained Model – Student Recommendations")

    # ---------- SAFETY CHECK ----------
    if "trained_data" not in st.session_state:
        st.warning("⚠️ Please train the model first in Model Training section.")
        st.stop()

    df = st.session_state["trained_data"]

    st.subheader("📊 Student Wise Recommendations (From Trained Data)")

    # ---------- GENERATE RECOMMENDATIONS ----------
    def generate_recommendation(row):
        if row["Performance_Level"] == "Excellent Performer":
            return "keep up the great work and maintain consistency."
        elif row["Performance_Level"] == "Good Performer":
            return "slight improvement in study routine can boost results."
        elif row["Performance_Level"] == "Average Performer":
            return " increase focus, revision, and practice regularly."
        else:
            return "dedicate more time to studies and seek guidance."

    # Create Recommendation column
    df["Recommendation"] = df.apply(generate_recommendation, axis=1)

    # ---------- FINAL DISPLAY TABLE ----------
    display_df = df[
        ["Student_ID", "Student_Name", "Predicted_Marks", "Performance_Level", "Recommendation"]
    ]

    st.dataframe(display_df, use_container_width=True)

    # ---------- DOWNLOAD BUTTON ----------
    csv = display_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Recommendations as CSV",
        data=csv,
        file_name="student_recommendations.csv",
        mime="text/csv"
    )

# ------------------------------------------------
# DOCUMENTATION PAGE (FULL WORKFLOW)
# ------------------------------------------------
elif menu == "📄 Documentation":
    st.title("📄 Model Workflow Documentation")

    st.markdown("### StudyTrack AI – Workflow")

    # OPTION 1: Local image (recommended)
    st.image(
        "Image2.png",   # put image in same folder as app.py
        width=600
    )

#---------------------------------------------
#-------------(streamlit footer)--------------
#---------------------------------------------
st.markdown(
    """
    <hr>
    <div style="text-align:center; color:gray; font-size:14px;">
       Infosys Springboard 6.0 Internship | © 2025 StudyTrack AI | Designed by Ashish Kumar | Trained by Anil Kumar

    </div>
    """,
    unsafe_allow_html=True
)