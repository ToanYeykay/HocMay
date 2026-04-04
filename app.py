import streamlit as st
import sqlite3
import pandas as pd

# --- 1. CẤU HÌNH DANH SÁCH LỰA CHỌN ---
CITIES = ['Pune', 'Mumbai', 'Delhi', 'Chandigarh', 'Bangalore', 'Hyderabad']
CUISINES = ['Chinese', 'South Indian', 'Biryani', 'Fast Food', 'North Indian', 'Desserts']
MEAL_TYPES = ['Dinner', 'Breakfast', 'Snacks', 'Lunch']
MOODS = ['Celebrating', 'Lazy', 'Happy', 'Stressed']
HUNGER_LEVELS = ['Low', 'Medium', 'High']
COMPANIES = ['Partner', 'Family', 'Friends', 'Alone']

# --- 2. CÁC HÀM XỬ LÝ DATABASE ---
def init_db():
    conn = sqlite3.connect('food_orders_v2.db') # Đổi tên db để tránh xung đột cấu trúc cũ
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            name TEXT,
            age INTEGER,
            city TEXT,
            cuisine TEXT,
            meal_type TEXT,
            order_value REAL,
            discount_applied INTEGER,
            mood TEXT,
            hunger_level TEXT,
            company TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_order(data):
    conn = sqlite3.connect('food_orders_v2.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO orders (user_id, name, age, city, cuisine, meal_type, order_value, discount_applied, mood, hunger_level, company)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (data['user_id'], data['name'], data['age'], data['city'], data['cuisine'], data['meal_type'], 
          data['order_value'], data['discount_applied'], data['mood'], data['hunger_level'], data['company']))
    conn.commit()
    conn.close()

# --- 3. GIAO DIỆN NGƯỜI DÙNG (UI) ---
def main():
    st.set_page_config(page_title="Food Ordering System", layout="wide")
    init_db()

    st.title("🍴 Hệ Thống Quản Lý Đơn Hàng v2.0")
    st.markdown("---")

    with st.form("main_form", clear_on_submit=True):
        st.subheader("Nhập thông tin khách hàng & Đơn hàng")
        
        # Tạo 3 cột để giao diện gọn gàng hơn
        col_id, col_name, col_age = st.columns([1, 2, 1])
        with col_id:
            user_id = st.text_input("Mã User ID", placeholder="VD: USR-501")
        with col_name:
            name = st.text_input("Họ và tên", placeholder="Nguyễn Văn A")
        with col_age:
            age = st.number_input("Tuổi", min_value=1, max_value=100, value=25)

        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            city = st.selectbox("Thành phố", options=CITIES)
            cuisine = st.selectbox("Loại ẩm thực", options=CUISINES)
            meal_type = st.selectbox("Bữa ăn", options=MEAL_TYPES)
            order_value = st.number_input("Giá trị đơn hàng (VNĐ)", min_value=0.0, step=1000.0)

        with col2:
            discount_applied = st.radio("Sử dụng mã giảm giá?", options=[1, 0], 
                                       format_func=lambda x: "Có (1)" if x == 1 else "Không (0)", horizontal=True)
            mood = st.selectbox("Tâm trạng", options=MOODS, index=MOODS.index('Happy'))
            hunger_level = st.select_slider("Mức độ đói", options=HUNGER_LEVELS, value='Medium')
            company = st.selectbox("Đi cùng ai", options=COMPANIES, index=COMPANIES.index('Alone'))

        submitted = st.form_submit_button("Lưu vào hệ thống")

        if submitted:
            if not user_id or not name:
                st.error("⚠️ Vui lòng nhập đầy đủ Mã User ID và Tên!")
            else:
                order_data = {
                    "user_id": user_id, "name": name, "age": age, "city": city, 
                    "cuisine": cuisine, "meal_type": meal_type, "order_value": order_value,
                    "discount_applied": discount_applied, "mood": mood,
                    "hunger_level": hunger_level, "company": company
                }
                save_order(order_data)
                st.success(f"Đã ghi nhận đơn hàng cho ID: {user_id} ({name})")

    # --- 4. HIỂN THỊ DỮ LIỆU ---
    st.divider()
    if st.checkbox("Xem toàn bộ Database (Bao gồm User ID)"):
        conn = sqlite3.connect('food_orders_v2.db')
        df_view = pd.read_sql_query("SELECT * FROM orders ORDER BY id DESC", conn)
        conn.close()
        
        if not df_view.empty:
            st.dataframe(df_view, use_container_width=True)
        else:
            st.info("Database hiện đang trống.")

if __name__ == "__main__":
    main()