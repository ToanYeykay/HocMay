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
    # Sử dụng v3 vì cấu trúc bảng lại thay đổi (mất cột name)
    conn = sqlite3.connect('food_orders_v3.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
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
    conn = sqlite3.connect('food_orders_v3.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO orders (user_id, age, city, cuisine, meal_type, order_value, discount_applied, mood, hunger_level, company)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (data['user_id'], data['age'], data['city'], data['cuisine'], data['meal_type'], 
          data['order_value'], data['discount_applied'], data['mood'], data['hunger_level'], data['company']))
    conn.commit()
    conn.close()

# --- 3. GIAO DIỆN NGƯỜI DÙNG (UI) ---
def main():
    st.set_page_config(page_title="Hệ Thống Quản Lý Đơn Hàng", layout="wide")
    init_db()

    st.title("🍴 Quản Lý Đơn Hàng (Anonymous Mode)")
    st.caption("Dữ liệu hiện tại chỉ quản lý qua User ID để đảm bảo tính bảo mật.")

    with st.form("main_form", clear_on_submit=True):
        col_id, col_age = st.columns([3, 1])
        with col_id:
            user_id = st.text_input("Mã định danh User ID", placeholder="Nhập mã khách hàng (VD: 1001, USR-X,...)")
        with col_age:
            age = st.number_input("Tuổi", min_value=1, max_value=100, value=25)

        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            city = st.selectbox("Thành phố", options=CITIES)
            cuisine = st.selectbox("Loại ẩm thực", options=CUISINES)
            meal_type = st.selectbox("Bữa ăn", options=MEAL_TYPES)

        with col2:
            order_value = st.number_input("Giá trị đơn hàng (VNĐ)", min_value=0.0, step=1000.0)
            discount_applied = st.selectbox("Sử dụng mã giảm giá?", options=[1, 0], 
                                           format_func=lambda x: "Có (1)" if x == 1 else "Không (0)")
            
            # Group các trạng thái tâm lý/xã hội
            with st.expander("Thông tin thêm (Mood, Hunger, Company)", expanded=True):
                mood = st.selectbox("Tâm trạng", options=MOODS, index=MOODS.index('Happy'))
                hunger_level = st.select_slider("Mức độ đói", options=HUNGER_LEVELS, value='Medium')
                company = st.selectbox("Đi cùng ai", options=COMPANIES, index=COMPANIES.index('Alone'))

        submitted = st.form_submit_button("Xác nhận lưu dữ liệu")

        if submitted:
            if not user_id:
                st.error("Không được để trống User ID!")
            else:
                order_data = {
                    "user_id": user_id, "age": age, "city": city, 
                    "cuisine": cuisine, "meal_type": meal_type, "order_value": order_value,
                    "discount_applied": discount_applied, "mood": mood,
                    "hunger_level": hunger_level, "company": company
                }
                save_order(order_data)
                st.success(f"Đã lưu dữ liệu cho ID: **{user_id}**")

    # --- 4. HIỂN THỊ DỮ LIỆU ---
    st.divider()
    if st.checkbox("Hiển thị bảng dữ liệu hiện tại"):
        conn = sqlite3.connect('food_orders_v3.db')
        df_view = pd.read_sql_query("SELECT * FROM orders ORDER BY id DESC", conn)
        conn.close()
        
        if not df_view.empty:
            st.dataframe(df_view, use_container_width=True)
        else:
            st.info("Chưa có đơn hàng nào được nhập.")

if __name__ == "__main__":
    main()