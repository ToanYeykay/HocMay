import streamlit as st
import sqlite3
import pandas as pd

# --- 1. CẤU HÌNH DANH SÁCH LỰA CHỌN (Từ dữ liệu thực tế của bạn) ---
CITIES = ['Pune', 'Mumbai', 'Delhi', 'Chandigarh', 'Bangalore', 'Hyderabad']
CUISINES = ['Chinese', 'South Indian', 'Biryani', 'Fast Food', 'North Indian', 'Desserts']
MEAL_TYPES = ['Dinner', 'Breakfast', 'Snacks', 'Lunch']
MOODS = ['Celebrating', 'Lazy', 'Happy', 'Stressed']
HUNGER_LEVELS = ['High', 'Low', 'Medium']
COMPANIES = ['Partner', 'Family', 'Friends', 'Alone']

# --- 2. CÁC HÀM XỬ LÝ DATABASE ---
def init_db():
    conn = sqlite3.connect('food_orders.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    conn = sqlite3.connect('food_orders.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO orders (name, age, city, cuisine, meal_type, order_value, discount_applied, mood, hunger_level, company)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (data['name'], data['age'], data['city'], data['cuisine'], data['meal_type'], 
          data['order_value'], data['discount_applied'], data['mood'], data['hunger_level'], data['company']))
    conn.commit()
    conn.close()

# --- 3. GIAO DIỆN NGƯỜI DÙNG (UI) ---
def main():
    st.set_page_config(page_title="Food Ordering System", layout="wide")
    init_db()

    st.title("🍴 Hệ Thống Quản Lý Đơn Hàng Foodie")
    st.info("Nhập thông tin khách hàng mới và đơn hàng vào hệ thống bên dưới.")

    # Sử dụng st.form để gom nhóm dữ liệu nhập
    with st.form("main_form", clear_on_submit=True):
        st.subheader("Thông tin chi tiết")
        
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Tên khách hàng", placeholder="Nhập họ và tên...")
            age = st.number_input("Tuổi", min_value=1, max_value=100, value=25)
            city = st.selectbox("Thành phố", options=CITIES)
            cuisine = st.selectbox("Loại ẩm thực", options=CUISINES)
            meal_type = st.selectbox("Bữa ăn", options=MEAL_TYPES)

        with col2:
            order_value = st.number_input("Giá trị đơn hàng (VNĐ)", min_value=0.0, step=1000.0)
            discount_applied = st.radio("Sử dụng mã giảm giá?", options=[1, 0], 
                                       format_func=lambda x: "Có (1)" if x == 1 else "Không (0)", horizontal=True)
            
            # Thiết lập các giá trị mặc định theo yêu cầu của bạn
            mood = st.selectbox("Tâm trạng", options=MOODS, index=MOODS.index('Happy'))
            hunger_level = st.select_slider("Mức độ đói", options=HUNGER_LEVELS, value='Medium')
            company = st.selectbox("Đi cùng ai", options=COMPANIES, index=COMPANIES.index('Alone'))

        # Nút xác nhận
        submitted = st.form_submit_button("Xác nhận và Lưu đơn hàng")

        if submitted:
            if name.strip() == "":
                st.error("Lỗi: Bạn chưa nhập tên khách hàng!")
            else:
                order_data = {
                    "name": name, "age": age, "city": city, "cuisine": cuisine,
                    "meal_type": meal_type, "order_value": order_value,
                    "discount_applied": discount_applied, "mood": mood,
                    "hunger_level": hunger_level, "company": company
                }
                save_order(order_data)
                st.success(f"Đã lưu đơn hàng của khách hàng: **{name}**")

    # --- 4. HIỂN THỊ DỮ LIỆU ---
    st.divider()
    if st.checkbox("Xem danh sách đơn hàng vừa nhập"):
        conn = sqlite3.connect('food_orders.db')
        df_view = pd.read_sql_query("SELECT * FROM orders ORDER BY id DESC", conn)
        conn.close()
        
        if not df_view.empty:
            st.dataframe(df_view, use_container_width=True)
        else:
            st.warning("Chưa có dữ liệu nào trong database.")

if __name__ == "__main__":
    main()