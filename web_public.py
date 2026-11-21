
from pathlib import Path
import sqlite3
import hashlib
import secrets
import hmac
from datetime import datetime

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SecureCheck | Password Strength",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).resolve().parent
DATASET_PATH = BASE / "data" / "dataset.csv"
DB_PATH = BASE / "users.db"

LABELS = {
    0: "Weak",
    1: "Medium",
    2: "Strong"
}


# --------------------------------------------------
# SOFT MODERN BEIGE + CHARCOAL DESIGN
# --------------------------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F5F1E9;
    color: #34332F;
    font-family: Arial, sans-serif;
}

/* Main content */
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Headings */
h1, h2, h3 {
    color: #34332F;
    font-weight: 650;
}

/* Paragraphs */
p, label, span {
    color: #55534D;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #E9E1D4;
    border-right: 1px solid #D8CDBD;
}

/* Buttons */
.stButton > button {
    background-color: #49463F;
    color: #FFFFFF;
    border: none;
    border-radius: 9px;
    padding: 0.65rem 1rem;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background-color: #686257;
    color: #FFFFFF;
    border: none;
}

/* Input fields */
.stTextInput input {
    background-color: #FFFEFB;
    border: 1px solid #D8D0C4;
    border-radius: 8px;
    color: #34332F;
}

/* Select boxes */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #FFFEFB;
    border-radius: 8px;
    border-color: #D8D0C4;
}

/* Cards */
[data-testid="stMetric"],
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFEFB;
    border: 1px solid #E3DBCF;
    border-radius: 12px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #E3DBCF;
    border-radius: 10px;
}

/* Horizontal divider */
hr {
    border-color: #DED5C8;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}

.stTabs [data-baseweb="tab"] {
    color: #49463F;
}

/* Mobile responsive */
@media screen and (max-width: 768px) {
    .block-container {
        padding: 1rem 0.8rem 2rem 0.8rem;
    }

    h1 {
        font-size: 1.8rem;
    }

    h2 {
        font-size: 1.4rem;
    }

    section[data-testid="stSidebar"] {
        min-width: 0;
    }

    .stButton > button {
        width: 100%;
    }
}

@media screen and (max-width: 480px) {
    .block-container {
        padding-left: 0.6rem;
        padding-right: 0.6rem;
    }
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATABASE SETUP
# --------------------------------------------------

def create_database():
    connection = sqlite3.connect(DB_PATH)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        600000
    ).hex()


def signup(username, email, password):
    username = username.strip()
    email = email.strip().lower()

    if not username or not email or not password:
        return False, "Please fill in all fields."

    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)

    try:
        connection = sqlite3.connect(DB_PATH)

        connection.execute(
            """
            INSERT INTO users
            (username, email, password_hash, salt)
            VALUES (?, ?, ?, ?)
            """,
            (username, email, password_hash, salt)
        )

        connection.commit()
        connection.close()

        return True, "Account created successfully."

    except sqlite3.IntegrityError:
        return False, "Username or email already exists."


def login(username, password):
    connection = sqlite3.connect(DB_PATH)

    result = connection.execute(
        """
        SELECT password_hash, salt
        FROM users
        WHERE username = ?
        """,
        (username.strip(),)
    ).fetchone()

    connection.close()

    if result is None:
        return False

    stored_hash, salt = result
    entered_hash = hash_password(password, salt)

    return hmac.compare_digest(stored_hash, entered_hash)


create_database()


# --------------------------------------------------
# PASSWORD FEATURE EXTRACTION
# --------------------------------------------------

COMMON_VALUES = {
    "password",
    "password123",
    "123456",
    "12345678",
    "qwerty",
    "admin",
    "welcome",
    "letmein"
}


def extract_features(password):
    return {
        "length": len(password),
        "uppercase": sum(c.isupper() for c in password),
        "lowercase": sum(c.islower() for c in password),
        "digits": sum(c.isdigit() for c in password),
        "symbols": sum(not c.isalnum() for c in password),
        "unique_chars": len(set(password)),
        "has_common": int(password.lower() in COMMON_VALUES)
    }


# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

@st.cache_resource
def train_model(csv_path):
    data = pd.read_csv("data\dataset.csv", on_bad_lines="skip")

    required_columns = {"password", "strength"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "Dataset must contain password and strength columns."
        )

    data = data.dropna(
        subset=["password", "strength"]
    ).reset_index(drop=True)

    if len(data) > 50000:
        data = data.sample(
            50000,
            random_state=42
        ).reset_index(drop=True)

    y = pd.to_numeric(
        data["strength"],
        errors="coerce"
    )

    valid = y.isin([0, 1, 2])

    data = data.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)

    X = pd.DataFrame(
        [extract_features(str(p)) for p in data["password"]]
    ).reset_index(drop=True)

    if X.empty:
        raise ValueError("No valid training data found.")

    if y.nunique() < 2:
        raise ValueError(
            "Dataset must contain at least two strength classes."
        )

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    return model


def check_password(model, password):
    features = extract_features(password)

    input_data = pd.DataFrame([features])

    prediction = int(model.predict(input_data)[0])

    probabilities = model.predict_proba(input_data)[0]
    confidence = float(max(probabilities)) * 100

    return LABELS[prediction], confidence


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "saved_results" not in st.session_state:
    st.session_state.saved_results = []

if "last_strength" not in st.session_state:
    st.session_state.last_strength = None

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None


# --------------------------------------------------
# LOGIN AND SIGNUP PAGE
# --------------------------------------------------

if not st.session_state.logged_in:

    st.markdown(
        "<h1 style='text-align:center;'>SecureCheck</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center;'>"
        "A simple password strength analysis project"
        "</p>",
        unsafe_allow_html=True
    )

    st.write("")

    left, center, right = st.columns([1, 2, 1])

    with center:

        st.markdown("### Welcome")

        login_tab, signup_tab = st.tabs(
            ["Login", "Create account"]
        )

        with login_tab:

            with st.form("login_form"):

                login_username = st.text_input(
                    "Username"
                )

                login_password = st.text_input(
                    "Account password",
                    type="password"
                )

                login_button = st.form_submit_button(
                    "Login",
                    use_container_width=True
                )

                if login_button:

                    if login(login_username, login_password):

                        st.session_state.logged_in = True
                        st.session_state.username = login_username.strip()
                        st.session_state.saved_results = []
                        st.session_state.last_strength = None
                        st.session_state.last_confidence = None

                        st.success("Login successful!")
                        st.rerun()

                    else:
                        st.error("Incorrect username or password.")

        with signup_tab:

            with st.form("signup_form"):

                new_username = st.text_input(
                    "Choose a username"
                )

                new_email = st.text_input(
                    "Email address"
                )

                new_password = st.text_input(
                    "Create an account password",
                    type="password"
                )

                confirm_password = st.text_input(
                    "Confirm account password",
                    type="password"
                )

                signup_button = st.form_submit_button(
                    "Create account",
                    use_container_width=True
                )

                if signup_button:

                    if new_password != confirm_password:
                        st.error("Passwords do not match.")

                    else:
                        success, message = signup(
                            new_username,
                            new_email,
                            new_password
                        )

                        if success:
                            st.success(message)
                        else:
                            st.error(message)

    st.stop()


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

try:
    if not DATASET_PATH.exists():
        st.error(
            "Dataset not found. Please check that "
            "data/dataset.csv exists."
        )
        st.stop()

    model = train_model(str(DATASET_PATH))

except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


# --------------------------------------------------
# SIDEBAR DASHBOARD
# --------------------------------------------------

with st.sidebar:

    st.markdown("## SecureCheck")

    st.caption("Password strength checker")

    st.divider()

    st.write("Welcome,")

    st.markdown(
        f"**{st.session_state.username}**"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Password checker",
            "Saved results"
        ]
    )

    st.divider()

    st.caption("Account")

    if st.button(
        "Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.saved_results = []
        st.session_state.last_strength = None
        st.session_state.last_confidence = None

        st.rerun()


# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

if page == "Dashboard":

    st.title("Dashboard")

    st.write(
        "Welcome to your password strength analysis dashboard."
    )

    st.write("")

    total_results = len(
        st.session_state.saved_results
    )

    st.metric(
        "Saved analyses",
        total_results
    )

    st.divider()

    st.subheader("About SecureCheck")

    st.write(
        """
        SecureCheck is an educational project that uses
        machine learning to classify password strength.

        It analyses features such as length, uppercase
        letters, lowercase letters, digits and symbols.
        """
    )

    st.info(
        "Use the sidebar to open the password checker "
        "or view your saved analysis results."
    )


# --------------------------------------------------
# PASSWORD CHECKER PAGE
# --------------------------------------------------

elif page == "Password checker":

    st.title("Password checker")

    st.write(
        "Enter a made-up sample password to analyse its strength."
    )

    st.info(
        "For this student project, do not enter your real "
        "passwords or passwords used for personal accounts."
    )

    with st.container(border=True):

        st.subheader("Analyse a password")

        password_input = st.text_input(
            "Sample password",
            type="password",
            placeholder="Enter a sample password"
        )

        check_button = st.button(
            "Check strength",
            use_container_width=True
        )

        if check_button:

            if not password_input:
                st.warning("Please enter a sample password.")

            else:

                strength, confidence = check_password(
                    model,
                    password_input
                )

                st.session_state.last_strength = strength
                st.session_state.last_confidence = confidence

                st.rerun()

    if st.session_state.last_strength is not None:

        st.write("")

        st.subheader("Analysis result")

        strength = st.session_state.last_strength
        confidence = st.session_state.last_confidence

        if strength == "Weak":
            st.error("Strength: Weak")

        elif strength == "Medium":
            st.warning("Strength: Medium")

        else:
            st.success("Strength: Strong")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted strength",
                strength
            )

        with col2:
            st.metric(
                "Model confidence",
                f"{confidence:.1f}%"
            )

        st.caption(
            "Confidence is the model's prediction confidence, "
            "not a guarantee of password security."
        )

        st.divider()

        st.subheader("Save this result")

        nickname = st.text_input(
            "Result name",
            placeholder="Example: Sample password 1",
            key="result_nickname"
        )

        if st.button(
            "Save result",
            use_container_width=True
        ):

            if not nickname.strip():
                st.warning("Please enter a result name.")

            else:

                st.session_state.saved_results.append({
                    "name": nickname.strip(),
                    "strength": strength,
                    "confidence": round(confidence, 1),
                    "date": datetime.now().strftime(
                        "%d-%m-%Y %H:%M"
                    )
                })

                st.success("Result saved successfully!")

                st.session_state.last_strength = None
                st.session_state.last_confidence = None

                st.rerun()


# --------------------------------------------------
# SAVED RESULTS PAGE
# --------------------------------------------------

elif page == "Saved results":

    st.title("Saved results")

    st.write(
        "Your saved analysis summaries are shown below."
    )

    results = st.session_state.saved_results

    if not results:

        st.info("No saved results yet.")

    else:

        st.metric(
            "Total saved results",
            len(results)
        )

        st.write("")

        results_df = pd.DataFrame(results)

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Delete an individual result")

        result_names = [
            f"{i + 1}. {item['name']}"
            for i, item in enumerate(results)
        ]

        selected_result = st.selectbox(
            "Select a result",
            result_names
        )

        if st.button(
            "Delete selected result",
            use_container_width=True
        ):

            selected_index = result_names.index(
                selected_result
            )

            st.session_state.saved_results.pop(
                selected_index
            )

            st.success("Selected result deleted.")
            st.rerun()

        st.divider()

        if st.button(
            "Clear all saved results",
            use_container_width=True
        ):

            st.session_state.saved_results = []

            st.success("All saved results cleared.")
            st.rerun()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "SecureCheck | Educational Machine Learning Project"
)
