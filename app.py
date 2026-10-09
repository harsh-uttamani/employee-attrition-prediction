"""
app.py
Day 11-12: Flask Web Application
----------------------------------
This Flask app:
1. Loads our saved Random Forest model + encoders (trained by train_model.py)
2. Shows a form where a user enters employee details
3. Encodes that input EXACTLY the way training data was encoded
4. Predicts attrition risk and shows the result + top influencing factors
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from bson.objectid import ObjectId
import datetime

app = Flask(__name__)
app.secret_key = "super_secret_attrition_key"  # Required for sessions

# MongoDB Setup
# Connect to local MongoDB instance
client = MongoClient('mongodb://localhost:27017/')
db = client['attrition_db']
companies_collection = db['companies']
users_collection = db['users']

# Ensure unique indexes
companies_collection.create_index("company_email", unique=True)
users_collection.create_index("email", unique=True)

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ---------------------------------------------------------
# Load all saved model artifacts ONCE when the server starts
# (not on every request -- that would be slow and wasteful)
# ---------------------------------------------------------
model = joblib.load("model/attrition_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

# Options for dropdowns -- these MUST exactly match the text values
# that existed in the training data, because label_encoders were fit on them.
DROPDOWN_OPTIONS = {
    "BusinessTravel": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
    "Department": ["Sales", "Research & Development", "Human Resources"],
    "EducationField": ["Life Sciences", "Other", "Medical", "Marketing",
                        "Technical Degree", "Human Resources"],
    "Gender": ["Male", "Female"],
    "JobRole": ["Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources"],
    "MaritalStatus": ["Single", "Married", "Divorced"],
    "OverTime": ["Yes", "No"],
}


# ==========================================
# AUTHENTICATION ROUTES
# ==========================================

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        company_name = request.form.get("company_name")
        company_email = request.form.get("company_email")
        admin_name = request.form.get("admin_name")
        username = request.form.get("username")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if password != confirm_password:
            return render_template("signup.html", error="Passwords do not match.")

        # Check if company or user already exists
        if companies_collection.find_one({"company_email": company_email}):
            return render_template("signup.html", error="Company email already registered.")
        if users_collection.find_one({"email": username}):
            return render_template("signup.html", error="Username (email) already exists.")

        # Create company
        company_doc = {
            "company_name": company_name,
            "company_email": company_email,
            "created_at": datetime.datetime.now(datetime.timezone.utc)
        }
        company_id = companies_collection.insert_one(company_doc).inserted_id

        # Create admin user for this company
        user_doc = {
            "company_id": company_id,
            "name": admin_name,
            "email": username,
            "password_hash": generate_password_hash(password),
            "role": "Company Admin",
            "status": "Active",
            "created_at": datetime.datetime.now(datetime.timezone.utc)
        }
        users_collection.insert_one(user_doc)

        return redirect(url_for('login'))
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = users_collection.find_one({"email": username})
        if user and check_password_hash(user["password_hash"], password):
            # Fetch company to store in session
            company = companies_collection.find_one({"_id": user["company_id"]})
            
            # Setup session
            session["user_id"] = str(user["_id"])
            session["company_id"] = str(user["company_id"])
            session["user_name"] = user["name"]
            session["company_name"] = company["company_name"] if company else "Unknown"
            session["role"] = user["role"]
            
            return redirect(url_for('home'))
        else:
            return render_template("login.html", error="Invalid credentials.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/methodology")
def methodology():
    return render_template("methodology.html")


# ==========================================
# PREDICTION ROUTES
# ==========================================

@app.route("/")
@login_required
def home():
    """Show the empty input form."""
    return render_template("index.html", dropdowns=DROPDOWN_OPTIONS, feature_columns=feature_columns)


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    """
    Read form data, encode it the same way as training data,
    run it through the model, and show the result.
    """
    # STEP 1: Read every form field into a dictionary.
    # request.form gives us submitted values as strings; we convert types
    # column by column so the model receives the same data types it trained on.
    form_data = {}
    for col in feature_columns:
        value = request.form.get(col)
        if col in DROPDOWN_OPTIONS:
            form_data[col] = value  # keep as text for now, encode below
        else:
            form_data[col] = float(value)  # numeric fields

    # STEP 2: Put it into a single-row DataFrame with columns in the EXACT
    # order the model was trained on -- order matters for scikit-learn models.
    input_df = pd.DataFrame([form_data], columns=feature_columns)

    # STEP 3: Encode categorical (text) columns using the SAME encoders
    # that were fit during training -- this guarantees "Sales" always
    # becomes the same number it was during training.
    for col, encoder in label_encoders.items():
        input_df[col] = encoder.transform(input_df[col])

    # STEP 4: Predict.
    # predict() gives the class (0 = stay, 1 = leave)
    # predict_proba() gives the confidence/probability behind that class
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  # probability of "Yes, will leave"

    result = "Likely to LEAVE" if prediction == 1 else "Likely to STAY"
    confidence = round(probability * 100, 2) if prediction == 1 else round((1 - probability) * 100, 2)

    # STEP 5: Get the model's overall top 5 most influential features
    # (global importance -- same for every prediction, but tells the user
    # WHAT the model generally pays most attention to).
    importances = pd.Series(model.feature_importances_, index=feature_columns)
    top_factors = importances.sort_values(ascending=False).head(5)
    top_factors_list = [(name, round(score, 3)) for name, score in top_factors.items()]

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        top_factors=top_factors_list,
        prediction=int(prediction)
    )


if __name__ == "__main__":
    # debug=True auto-reloads the server when we edit code -- great for
    # development, but should be turned OFF (debug=False) in production.
    app.run(debug=True)
