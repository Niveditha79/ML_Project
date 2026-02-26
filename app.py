import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Email Spam Detection",
    layout="centered"
)

# --------------------------------------------------
# BACKGROUND CSS (LIGHT BLUE + LIGHT RED)
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg, #e3f2fd, #fdecea);
    font-family: Arial, sans-serif;
}

/* Main white card */
.main-card {
    background-color: white;
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0 6px 15px rgba(0,0,0,0.1);
    margin-top: 20px;
}

/* Section cards */
.section-card {
    background-color: #f9fafb;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    border-left: 5px solid #60a5fa;
}

/* Titles */
.title {
    text-align: center;
    font-size: 34px;
    font-weight: bold;
    color: #1f2937;
}

.subtitle {
    text-align: center;
    font-size: 15px;
    color: #6b7280;
    margin-bottom: 25px;
}

/* Buttons */
div.stButton > button {
    background-color: #ef4444;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    padding: 8px 20px;
    border: none;
}

div.stButton > button:hover {
    background-color: #dc2626;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# UI START
# --------------------------------------------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

st.markdown("<div class='title'>📧 Email Spam Detection</div>", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
try:
    data = pd.read_csv("spam.csv", encoding="latin-1")
except:
    st.error("spam.csv file not found. Place it in the same folder as app.py")
    st.stop()


# Smart column handling
if {'v1', 'v2'}.issubset(data.columns):
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']

elif {'label', 'text'}.issubset(data.columns):
    data = data[['label', 'text']]
    data.columns = ['label', 'message']

elif {'label', 'message'}.issubset(data.columns):
    data = data[['label', 'message']]

elif {'Category', 'Message'}.issubset(data.columns):
    data = data[['Category', 'Message']]
    data.columns = ['label', 'message']

else:
    st.error("Dataset column format not recognized")
    st.stop()

data['label'] = data['label'].astype(str).str.lower()
data['label'] = data['label'].map({
    'spam': 1,
    'ham': 0,
    'not spam': 0,
    'not_spam': 0
})
data.dropna(inplace=True)

# st.success("Dataset loaded successfully")
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
# st.subheader("🤖 Model Training")

X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊Performance")
st.write(f"Accuracy: **{accuracy * 100:.2f}%**")

with st.expander("Classification Report"):
    st.text(classification_report(y_test, y_pred))

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("✉️ Testing")

user_message = st.text_area(
    "Enter Email / SMS Message",
    placeholder="Congratulations! You have won a free prize..."
)

if st.button("Predict"):
    if user_message.strip() == "":
        st.warning("Please enter a message")
    else:
        prediction = model.predict([user_message])[0]
        if prediction == 1:
            st.error("🚫 SPAM MESSAGE")
        else:
            st.success("✅ NOT SPAM")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.caption("Mini Project | Machine Learning | Email Spam Detection")