import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

st.set_page_config(page_title="Food Ordering Behavior Dashboard", layout="wide")

st.title("Phân Tích Hành Vi Đặt Đồ Ăn 🍔")

# Load dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("food_ordering_behavior_dataset.csv")
    return df

df = load_data()

# Sidebar lọc dữ liệu
city = st.sidebar.multiselect("Chọn thành phố:", options=df["city"].unique(), default=df["city"].unique())
df_selection = df[df["city"].isin(city)]

import streamlit as st
import pandas as pd

def init_db():
    conn = sqlite3.connect('customer_data.db')
    c = conn.cursor()
    # Tạo bảng nếu chưa tồn tại
    c.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT,
            name TEXT,
            age INTEGER,
            cuisine TEXT,
            meal_type TEXT,
            order_value REAL,
            discount_applied INTEGER,
            mood TEXT,
            hunger_level TEXT,
            company TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_data(city, name, age, cuisine, meal_type, order_value, discount_applied, mood, hunger_level, company):
    conn = sqlite3.connect('customer_data.db')
    c = conn.cursor()
    query = '''
        INSERT INTO orders (city, name, age, cuisine, meal_type, order_value, discount_applied, mood, hunger_level, company)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
    c.execute(query, (city, name, age, cuisine, meal_type, order_value, discount_applied, mood, hunger_level, company))
    conn.commit()
    conn.close()

# --- GIAO DIỆN STREAMLIT ---
def main():
    st.set_page_config(page_title="Quản lý đơn hàng", layout="wide")
    init_db() # Khởi tạo DB ngay khi chạy app

    st.title("🍽️ Hệ Thống Ghi Nhận Đơn Hàng")
    
    with st.form("order_form", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Tên khách hàng")
            age = st.number_input("Tuổi", min_value=1, max_value=100, value=25)
            city = st.text_input("Thành phố")
            cuisine = st.selectbox("Cuisine", ["Vietnamese", "Italian", "Japanese", "Korean", "Western"])
            meal_type = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner", "Snack"])

        with col2:
            order_value = st.number_input("Order Value (VNĐ)", min_value=0, step=1000)
            discount_applied = st.selectbox("Discount Applied", options=[1, 0], 
                                           format_func=lambda x: "Có (1)" if x == 1 else "Không (0)")
            mood = st.selectbox("Mood", ["Happy", "Neutral", "Sad", "Stressed"], index=0)
            hunger_level = st.select_slider("Hunger Level", options=["Low", "Medium", "High"], value="Medium")
            company = st.selectbox("Company", ["Alone", "Friends", "Family", "Partner"], index=0)

        submitted = st.form_submit_button("Lưu vào cơ sở dữ liệu")

        if submitted:
            if name and city:
                add_data(city, name, age, cuisine, meal_type, order_value, discount_applied, mood, hunger_level, company)
                st.success(f"✅ Đã thêm dữ liệu của {name} vào database!")
            else:
                st.error("Vui lòng điền các thông tin bắt buộc (Tên, Thành phố)!")

    # --- HIỂN THỊ DỮ LIỆU ĐÃ LƯU ---
    if st.checkbox("Hiển thị danh sách đơn hàng đã lưu"):
        conn = sqlite3.connect('customer_data.db')
        df = pd.read_sql_query("SELECT * FROM orders ORDER BY id DESC", conn)
        st.dataframe(df)
        conn.close()

if __name__ == "__main__":
    main()