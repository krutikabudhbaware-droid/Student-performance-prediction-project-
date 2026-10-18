import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="EduPredict | Student Performance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #173f5f, #20639b);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 18px;
    margin-top: 5px;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result-card {
    padding: 30px;
    border-radius: 18px;
    background: #ffffff;
    text-align: center;
    box-shadow: 0 4px 18px rgba(0,0,0,0.10);
}

.score {
    font-size: 52px;
    font-weight: bold;
    margin: 10px;
}

.footer {
    text-align: center;
    color: #777;
    padding: 25px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "student_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Model could not be loaded: {e}")
    st.stop()
    

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🎓 EduPredict")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Prediction",
            "📊 Student Analysis",
            "ℹ️ About Project"
        ]
    )

    st.markdown("---")

    st.caption("Student Performance Prediction")
    st.caption("Machine Learning + Streamlit")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">

<h1>🎓 EduPredict</h1>

<p>Student Performance Prediction & Academic Analytics</p>

</div>
""", unsafe_allow_html=True)

# ==================================================
# PREDICTION PAGE
# ==================================================

if page == "🏠 Prediction":

    st.markdown("## Student Prediction")

    st.write(
        "Enter the student's academic information below to estimate "
        "their performance."
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # Student information

    col1, col2 = st.columns(2)

    with col1:

        student_name = st.text_input(
            "Student Name",
            placeholder="Enter student name"
        )

        study_hours = st.number_input(
            "📚 Study Hours per Day",
            min_value=0.0,
            max_value=24.0,
            value=4.0,
            step=0.5
        )

    with col2:

        attendance = st.number_input(
            "📅 Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0
        )

        previous_score = st.number_input(
            "📝 Previous Score",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0
        )

    sleep_hours = st.number_input(
        "😴 Sleep Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # Prediction button

    predict = st.button(
        "🔮 Predict Student Performance",
        use_container_width=True
    )

    if predict:

        # Model input
        input_data = [[
            study_hours,
            attendance,
            previous_score,
            sleep_hours
        ]]

        prediction = model.predict(input_data)[0]

        # Keep result between 0 and 100
        prediction = max(0, min(100, float(prediction)))

        # Performance category

        if prediction >= 75:
            category = "High Performance"
            message = "The student shows strong predicted performance."

        elif prediction >= 50:
            category = "Moderate Performance"
            message = "The student has moderate predicted performance."

        else:
            category = "Needs Improvement"
            message = "The student may benefit from additional academic support."

        st.markdown("---")

        # Result

        name_display = student_name if student_name else "Student"

        st.markdown(
            f"""
            <div class="result-card">

            <h2>Prediction Result</h2>

            <p><b>{name_display}</b></p>

            <div class="score">{prediction:.2f}</div>

            <h3>{category}</h3>

            <p>{message}</p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 📊 Student Performance Overview")

        # Metrics

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Study Hours",
            f"{study_hours:.1f} hrs"
        )

        c2.metric(
            "Attendance",
            f"{attendance:.0f}%"
        )

        c3.metric(
            "Previous Score",
            f"{previous_score:.0f}"
        )

        c4.metric(
            "Sleep",
            f"{sleep_hours:.1f} hrs"
        )

        # Chart

        chart_data = pd.DataFrame(
            {
                "Parameter": [
                    "Attendance",
                    "Previous Score",
                    "Predicted Performance"
                ],
                "Score": [
                    attendance,
                    previous_score,
                    prediction
                ]
            }
        )

        st.markdown("### 📈 Performance Comparison")

        st.bar_chart(
            chart_data.set_index("Parameter")
        )

        # Suggestions

        st.markdown("### 💡 Recommendations")

        recommendations = []

        if study_hours < 4:
            recommendations.append(
                "Increase focused study time gradually."
            )

        if attendance < 75:
            recommendations.append(
                "Try to maintain regular class attendance."
            )

        if previous_score < 60:
            recommendations.append(
                "Review previous topics and practice regularly."
            )

        if sleep_hours < 7:
            recommendations.append(
                "Maintain a consistent and adequate sleep schedule."
            )

        if not recommendations:
            recommendations.append(
                "Keep following your current study routine and maintain consistency."
            )

        for recommendation in recommendations:
            st.info("💡 " + recommendation)

# ==================================================
# STUDENT ANALYSIS PAGE
# ==================================================

elif page == "📊 Student Analysis":

    st.markdown("## 📊 Student Analysis")

    st.write(
        "This section provides an overview of the factors "
        "used by the prediction system."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="card">

            <h3>📚 Academic Factors</h3>

            <p><b>Study Hours</b><br>
            Daily study time provided by the student.</p>

            <p><b>Previous Score</b><br>
            Previous academic performance.</p>

            <p><b>Attendance</b><br>
            Percentage of classes attended.</p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="card">

            <h3>🌱 Lifestyle Factor</h3>

            <p><b>Sleep Hours</b><br>
            Average daily sleep duration.</p>

            <p>These inputs are provided to the trained
            machine-learning model for prediction.</p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 🔄 Prediction Workflow")

    st.info(
        "Student Data → Preprocessing → Machine Learning Model → "
        "Performance Prediction → Analysis & Recommendations"
    )

# ==================================================
# ABOUT PAGE
# ==================================================

elif page == "ℹ️ About Project":

    st.markdown("## ℹ️ About EduPredict")

    st.markdown(
        """
        <div class="card">

        <h3>Student Performance Prediction</h3>

        <p>
        EduPredict is a machine-learning based student performance
        prediction project. It uses student-related academic and
        lifestyle information to estimate predicted performance.
        </p>

        <h4>Technologies Used</h4>

        <ul>
            <li>Python</li>
            <li>Pandas</li>
            <li>Scikit-learn</li>
            <li>Joblib</li>
            <li>Streamlit</li>
        </ul>

        <h4>Project Objective</h4>

        <p>
        The objective is to demonstrate how machine learning can
        be used to analyze student information and generate a
        performance prediction.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 👩‍💻 Project Team")

    st.write("**Krutika Budhbaware**")
    st.write("**Prerna Misale**")
    st.write("**Riya Kapse**")
    st.write("**Sanika Bhure**")

    st.markdown("### 👩‍🏫 Project Guide")

    st.write("**Ashwini Pise Mam**")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
    EduPredict • Student Performance Prediction System<br>
    Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)
