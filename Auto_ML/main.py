import streamlit as st
import sqlite3
import os

# Database connection

conn = sqlite3.connect("users.db")
c = conn.cursor()

# Create users table if not exists
c.execute("""CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY, username TEXT, password TEXT)""")
conn.commit()

# Initialization check
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""


def authenticate_user(username, password):
    c.execute(
        "SELECT * FROM users WHERE username = ? AND password = ?", (username, password)
    )
    result = c.fetchone()
    if result:
        return True
    else:
        return False


def user_exists(username):
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    if result:
        return True
    else:
        return False


def create_user(username, password):
    c.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)", (username, password)
    )
    conn.commit()


def show_login_signup():
    st.title("Login/Sign Up")

    page = st.radio("Go to", ["Login", "Sign Up"])

    if page == "Login":
        show_login()
    elif page == "Sign Up":
        show_signup()


def show_login():
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate_user(username, password):
            st.success(f"Logged in as {username}")
            st.session_state.username = username
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")


def show_signup():
    st.title("Sign Up")

    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")

    if st.button("Sign Up"):
        if user_exists(new_username):
            st.error("Username already exists")
        else:
            create_user(new_username, new_password)
            st.success("User created successfully")
            st.session_state.username = new_username
            st.session_state.logged_in = True
            st.experimental_rerun()


def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.experimental_rerun()


def main():
    if st.session_state.logged_in:
        username = st.session_state.username
        st.title("Helix AI - An Auto-ML Tool")
        st.write(
            "Welcome to Helix AI, an Auto-ML tool that helps you build machine learning models with ease."
        )
        st.write(f"Logged in as {username}")
        if st.button("Logout"):
            logout()

    else:
        show_login_signup()

    if st.session_state.logged_in:
        if st.button("Supervised Learning"):
            os.system("streamlit run app.py")
        elif st.button("Unsupervised Learning"):
            os.system("streamlit run app2.py")
        elif st.button("NLP By BERT"):
            os.system("streamlit run app3.py")
        elif st.button("Auto Neural Network"):
            os.system("streamlit run app4.py")


if __name__ == "__main__":
    main()
