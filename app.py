import streamlit as st

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fake News Detector - God Mode",
    page_icon="🛡️",
    layout="centered"
)

# -----------------------------
# Custom Styling (SAFE CSS)
# -----------------------------
st.markdown("""
<style>

body {
    background-color: #0e1117;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
    margin-bottom: 10px;
}

.sub-text {
    text-align: center;
    color: #bbbbbb;
    margin-bottom: 30px;
}

.input-box textarea {
    border-radius: 10px !important;
}

.stButton>button {
    background: linear-gradient(90deg, #4CAF50, #2E8B57);
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
}

.result-real {
    padding: 15px;
    border-radius: 10px;
    background-color: #1e4620;
    color: #90ee90;
    font-size: 20px;
    text-align: center;
}

.result-fake {
    padding: 15px;
    border-radius: 10px;
    background-color: #4a1c1c;
    color: #ff7b7b;
    font-size: 20px;
    text-align: center;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header Section
# -----------------------------
st.markdown('<div class="main-title">🛡️ Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">AI Powered Misinformation Detection System</div>', unsafe_allow_html=True)

# -----------------------------
# Input Section
# -----------------------------
user_input = st.text_area("Enter News Text Below:", height=150)

# -----------------------------
# Prediction Section
# -----------------------------
if st.button("🔍 Analyze News"):

    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        # 🔴 Replace this with your real model prediction
        # Example dummy logic:
        if "fake" in user_input.lower():
            prediction = "FAKE"
        else:
            prediction = "REAL"

        # -----------------------------
        # Display Result
        # -----------------------------
        if prediction == "REAL":
            st.markdown(
                '<div class="result-real">✅ This News Appears to be REAL</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-fake">🚨 This News Appears to be FAKE</div>',
                unsafe_allow_html=True
            )

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    '<div class="footer">Built by Jd Vardhan | Exhibition 2026</div>',
    unsafe_allow_html=True
)
