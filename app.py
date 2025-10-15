import streamlit as st
import sqlite3
import pandas as pd
import os
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------------
# PAGE SETUP
# -------------------------------
st.set_page_config(page_title="AI Car Reselling Platform", layout="wide")
st.title("🚗 Smart Car Reselling Portal")
st.markdown("> *Buy and sell smarter — negotiate directly with other users!*")
st.markdown("---")

DB_PATH = "car_resell.db"
CSV_PATH = "advanced_car_reselling_dataset_india (1).csv"

# -------------------------------
# DATABASE INITIALIZATION
# -------------------------------
def init_db():
    new_db = not os.path.exists(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT UNIQUE,
                 password TEXT,
                 role TEXT CHECK(role IN ('buyer','seller')) NOT NULL)''')

    c.execute('''CREATE TABLE IF NOT EXISTS cars (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 seller_id INTEGER,
                 car_name TEXT,
                 brand TEXT,
                 year INTEGER,
                 engine_cc REAL,
                 horsepower REAL,
                 kms_driven INTEGER,
                 fuel_type TEXT,
                 transmission TEXT,
                 predicted_price REAL,
                 asking_price REAL,
                 image_path TEXT,
                 status TEXT DEFAULT 'Available')''')

    c.execute('''CREATE TABLE IF NOT EXISTS proposals (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 buyer_id INTEGER,
                 seller_id INTEGER,
                 car_id INTEGER,
                 proposed_price REAL,
                 status TEXT DEFAULT 'Pending',
                 created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS messages (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 proposal_id INTEGER,
                 sender_id INTEGER,
                 message TEXT,
                 timestamp TEXT DEFAULT CURRENT_TIMESTAMP)''')

    conn.commit()
    conn.close()
    if new_db:
        st.info("🆕 New database created!")

init_db()

# -------------------------------
# HELPERS
# -------------------------------
def get_db_conn():
    return sqlite3.connect(DB_PATH)

def safe_label_encode(enc: LabelEncoder, value: str):
    if value is None:
        return 0
    try:
        if value in enc.classes_:
            return int(enc.transform([value])[0])
        else:
            return 0
    except:
        return 0

# -------------------------------
# LOAD & TRAIN MODEL
# -------------------------------
@st.cache_data
def load_csv(csv_path):
    if not os.path.exists(csv_path):
        st.warning(f"CSV not found at {csv_path}. Price prediction disabled.")
        return None
    return pd.read_csv(csv_path)

@st.cache_resource
def train_model(csv_path):
    df_raw = load_csv(csv_path)
    if df_raw is None:
        return None, {}, 0.0, 0.0

    df = df_raw.copy()
    rename_map = {
        "Car_Brand": "brand",
        "Car_Name": "car_name",
        "Year": "year",
        "Engine_CC": "engine_cc",
        "Horsepower_HP": "horsepower",
        "Total_Kilometres": "kms_driven",
        "Fuel_Type": "fuel_type",
        "Transmission": "transmission",
        "Selling_Price": "predicted_price"
    }
    df = df.rename(columns=rename_map)
    target_col = "predicted_price"
    expected_cols = ["brand", "car_name", "year", "engine_cc", "horsepower",
                     "kms_driven", "fuel_type", "transmission", target_col]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        return None, {}, 0.0, 0.0

    df = df[expected_cols]
    df = df.fillna({
        "year": df["year"].median(),
        "engine_cc": df["engine_cc"].median(),
        "horsepower": df["horsepower"].median(),
        "kms_driven": df["kms_driven"].median(),
        "brand": "Unknown",
        "car_name": "Unknown",
        "fuel_type": "Unknown",
        "transmission": "Unknown"
    })

    encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred) if len(y_test) > 0 else 0.0
    mae = mean_absolute_error(y_test, y_pred) if len(y_test) > 0 else 0.0

    return model, encoders, r2, mae

model, encoders, r2, mae = train_model(CSV_PATH)

# -------------------------------
# DATABASE FUNCTIONS
# -------------------------------
def create_user(username, password, role):
    conn = get_db_conn()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, password, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT id, username, role FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

def add_car(seller_id, car_name, brand, year, engine_cc, horsepower, kms_driven, fuel_type, transmission, image_path, predicted_price, asking_price):
    conn = get_db_conn()
    c = conn.cursor()
    c.execute('''INSERT INTO cars (seller_id, car_name, brand, year, engine_cc, horsepower,
                                   kms_driven, fuel_type, transmission, predicted_price, asking_price, image_path)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (seller_id, car_name, brand, year, engine_cc, horsepower, kms_driven,
               fuel_type, transmission, predicted_price, asking_price, image_path))
    conn.commit()
    conn.close()

def get_user_cars(seller_id):
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM cars WHERE seller_id=?", (seller_id,))
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]

def get_cars(exclude_seller_id=None, search_brand=None):
    conn = get_db_conn()
    c = conn.cursor()
    query = "SELECT * FROM cars WHERE status='Available'"
    params = []
    if exclude_seller_id:
        query += " AND seller_id != ?"
        params.append(exclude_seller_id)
    if search_brand:
        query += " AND LOWER(brand) LIKE ?"
        params.append(f"%{search_brand.lower()}%")
    c.execute(query, tuple(params))
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]

def delete_car(car_id, seller_id):
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("DELETE FROM cars WHERE id=? AND seller_id=?", (car_id, seller_id))
    conn.commit()
    conn.close()

def create_proposal(buyer_id, car_id, proposed_price):
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT seller_id FROM cars WHERE id=? AND status='Available'", (car_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False
    seller_id = row[0]
    c.execute("INSERT INTO proposals (buyer_id, seller_id, car_id, proposed_price) VALUES (?, ?, ?, ?)",
              (buyer_id, seller_id, car_id, proposed_price))
    conn.commit()
    conn.close()
    return True

def get_proposals_for_seller(seller_id):
    conn = get_db_conn()
    c = conn.cursor()
    c.execute('''SELECT p.id, p.buyer_id, u.username as buyer_name, p.car_id, c.car_name, p.proposed_price, p.status, p.created_at
                 FROM proposals p
                 JOIN users u ON p.buyer_id = u.id
                 JOIN cars c ON p.car_id = c.id
                 WHERE p.seller_id=? ORDER BY p.created_at DESC''', (seller_id,))
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]

def get_proposals_for_buyer(buyer_id):
    conn = get_db_conn()
    c = conn.cursor()
    c.execute('''SELECT p.id, p.car_id, c.car_name, p.proposed_price, p.status, p.created_at
                 FROM proposals p
                 JOIN cars c ON p.car_id = c.id
                 WHERE p.buyer_id=? ORDER BY p.created_at DESC''', (buyer_id,))
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]

def add_message(proposal_id, sender_id, message):
    conn = get_db_conn()
    c = conn.cursor()
    ts = datetime.utcnow().isoformat()
    c.execute("INSERT INTO messages (proposal_id, sender_id, message, timestamp) VALUES (?, ?, ?, ?)",
              (proposal_id, sender_id, message, ts))
    conn.commit()
    conn.close()

def get_messages(proposal_id):
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT id, proposal_id, sender_id, message, timestamp FROM messages WHERE proposal_id=? ORDER BY id ASC", (proposal_id,))
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]

def update_proposal_status(proposal_id, status):
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("UPDATE proposals SET status=? WHERE id=?", (status, proposal_id))
    if status == "Accepted":
        c.execute("SELECT car_id FROM proposals WHERE id=?", (proposal_id,))
        r = c.fetchone()
        if r:
            car_id = r[0]
            c.execute("UPDATE cars SET status='Sold' WHERE id=?", (car_id,))
    conn.commit()
    conn.close()

# -------------------------------
# AUTHENTICATION
# -------------------------------
if "user" not in st.session_state:
    st.session_state["user"] = None

if not st.session_state["user"]:
    st.sidebar.header("🔐 Login / Signup")
    mode = st.sidebar.radio("Choose Action", ["Login", "Signup"])
    if mode == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            user = authenticate_user(username, password)
            if user:
                st.session_state["user"] = {"id": user[0], "username": user[1], "role": user[2]}
                st.experimental_rerun = lambda: None
                st.stop()
            else:
                st.error("Invalid credentials.")
    else:
        username = st.sidebar.text_input("New Username")
        password = st.sidebar.text_input("New Password", type="password")
        role = st.sidebar.selectbox("Role", ["buyer", "seller"])
        if st.sidebar.button("Create Account"):
            ok = create_user(username, password, role)
            if ok:
                st.success("Account created! Please log in.")
            else:
                st.error("Username already exists.")
    st.stop()

# -------------------------------
# DASHBOARDS
# -------------------------------
user = st.session_state["user"]
role = user["role"]
user_id = user["id"]

st.sidebar.write(f"Logged in as: **{user['username']} ({role})**")
if st.sidebar.button("Logout"):
    st.session_state["user"] = None
    st.experimental_rerun = lambda: None
    st.stop()

# =======================
# SELLER DASHBOARD
# =======================
if role == "seller":
    st.header("🧾 Seller Dashboard")
    st.subheader("Add a new car listing")
    with st.form("add_car_form"):
        car_name = st.text_input("Car Name")
        brand = st.text_input("Car Brand")
        year = st.number_input("Year", 1980, 2050, 2018)
        engine_cc = st.number_input("Engine CC", 50, 10000, 1500)
        horsepower = st.number_input("Horsepower", 10, 5000, 120)
        kms_driven = st.number_input("Kms Driven", 0, 2000000, 20000)
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric", "Other"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Other"])
        image_path = st.text_input("Image Path (optional)")

        predicted_price_display = None
        if model is not None:
            if st.form_submit_button("🔍 Predict Price"):
                feat_names = list(model.feature_names_in_)
                input_row = {fn: 0 for fn in feat_names}
                input_row["year"] = int(year)
                input_row["engine_cc"] = float(engine_cc)
                input_row["horsepower"] = float(horsepower)
                input_row["kms_driven"] = int(kms_driven)
                if "brand" in encoders: input_row["brand"] = safe_label_encode(encoders["brand"], brand)
                if "car_name" in encoders: input_row["car_name"] = safe_label_encode(encoders["car_name"], car_name)
                if "fuel_type" in encoders: input_row["fuel_type"] = safe_label_encode(encoders["fuel_type"], fuel_type)
                if "transmission" in encoders: input_row["transmission"] = safe_label_encode(encoders["transmission"], transmission)
                input_df = pd.DataFrame([input_row], columns=feat_names)
                predicted_price_display = float(model.predict(input_df)[0])
                

        asking_price = st.number_input("Your Asking Price", min_value=0.0, value=0.0, step=0.01)

        if st.form_submit_button("Create Listing"):
            predicted_to_store = predicted_price_display if predicted_price_display is not None else 0.0
            add_car(user_id, car_name, brand, year, engine_cc, horsepower, kms_driven, fuel_type, transmission, image_path, predicted_to_store, asking_price)
            st.success(f"Listing created for '{car_name}' with asking price ₹{asking_price:.2f}")

    st.subheader("📦 My Listings")
    my_cars = get_user_cars(user_id)
    if my_cars:
        for car in my_cars:
            cols = st.columns([6, 1])
            with cols[0]:
                st.markdown(f"**ID {car.get('id')} — {car.get('brand','')} {car.get('car_name','')} ({car.get('year','')})**")
                st.write(f"- Engine: {car.get('engine_cc','N/A')}cc | {car.get('horsepower','N/A')}HP")
                st.write(f"- Asking Price: ₹{float(car.get('asking_price',0)):.2f} • Model Suggestion: ₹{float(car.get('predicted_price',0)):.2f}")
                st.write(f"- Status: {car.get('status','Available')}")
            with cols[1]:
                if st.button("🗑️ Delete", key=f"delete_{car.get('id')}"):
                    delete_car(car.get('id'), user_id)
                    st.success("Listing deleted.")
            st.markdown("---")

# =======================
# BUYER DASHBOARD
# =======================
else:
    st.header("🚘 Buyer Dashboard")
    st.write(f"Welcome, **{user['username']}** — browse available listings and send proposals below.")

    st.subheader("🔍 Browse & Search Listings")
    search_brand = st.text_input("Search by brand (partial match)", "")
    listings = get_cars(exclude_seller_id=user_id, search_brand=search_brand.strip() if search_brand else None)

    if listings:
        for car in listings:
            st.markdown(f"### {car.get('brand','')} {car.get('car_name','')} ({car.get('year','')})")
            st.write(f"- Engine: {car.get('engine_cc','N/A')}cc | {car.get('horsepower','N/A')}HP")
            st.write(f"- Asking Price: ₹{float(car.get('asking_price',0)):.2f} • Model Suggestion: ₹{float(car.get('predicted_price',0)):.2f}")
            st.write(f"- Fuel: {car.get('fuel_type','N/A')} • Transmission: {car.get('transmission','N/A')}")
            st.write(f"- Driven: {car.get('kms_driven','N/A')} km")

            # Proposal form
            with st.form(f"proposal_form_{car['id']}", clear_on_submit=True):
                proposed_price = st.number_input("Your proposal price", min_value=0.0, value=float(car.get('asking_price',0)), step=0.01, key=f"prop_{car['id']}")
                if st.form_submit_button("Send Proposal"):
                    ok = create_proposal(user_id, car['id'], proposed_price)
                    if ok:
                        st.success(f"Proposal ₹{proposed_price:.2f} sent to seller.")
                    else:
                        st.error("Failed to send proposal (car may no longer be available).")
            st.markdown("---")
    else:
        st.info("No listings found matching your search.")

    # My proposals + chat
    st.subheader("💬 My Proposals")
    my_props = get_proposals_for_buyer(user_id)
    if my_props:
        for p in my_props:
            st.markdown(f"**Proposal #{p['id']}** — Car: {p['car_name']} (ID:{p['car_id']})")
            st.write(f"- Proposed price: ₹{p['proposed_price']:.2f} • Status: {p['status']} • Created: {p['created_at']}")
            
            # Display chat messages
            msgs = get_messages(p['id'])
            if msgs:
                st.markdown("**Chat:**")
                for m in msgs:
                    sender_name = "You" if m['sender_id'] == user_id else "Seller"
                    st.write(f"- **{sender_name}** [{m['timestamp']}]: {m['message']}")
            else:
                st.info("No messages yet.")

            # Send message form
            with st.form(f"msg_form_buyer_{p['id']}", clear_on_submit=True):
                text = st.text_input("Message to seller", key=f"msg_buyer_{p['id']}")
                if st.form_submit_button("Send Message"):
                    if text.strip():
                        add_message(p['id'], user_id, text.strip())
                        st.success("Message sent.")
            st.markdown("---")
    else:
        st.info("You have not sent any proposals yet.")
