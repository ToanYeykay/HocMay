import streamlit as st
import sqlite3

# --- 1. Cấu hình danh sách lựa chọn (Từ dữ liệu duy nhất của bạn) ---
cities = ['Pune', 'Mumbai', 'Delhi', 'Chandigarh', 'Bangalore', 'Hyderabad']
cuisines = ['Chinese', 'South Indian', 'Biryani', 'Fast Food', 'North Indian', 'Desserts']
meal_types = ['Dinner', 'Breakfast', 'Snacks', 'Lunch']
moods = ['Celebrating', 'Lazy', 'Happy', 'Stressed']
hunger_levels = ['High', 'Low', 'Medium']
companies = ['Partner', 'Family', 'Friends', 'Alone']

def main():
    st.title("📝 Hệ Thống Nhập Liệu Khách Hàng")
    st.markdown("---")

    with st.form("order_form", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Tên khách hàng", placeholder="Ví dụ: Toàn")
            age = st.number_input("Tuổi", min_value=1, max_value=100, value=25)
            
            # Sử dụng st.selectbox cho các trường đã có danh sách
            city = st.selectbox("Thành phố (City)", options=cities)
            cuisine = st.selectbox("Loại ẩm thực (Cuisine)", options=cuisines)
            meal_type = st.selectbox("Bữa ăn (Meal Type)", options=meal_types)

        with col2:
            order_value = st.number_input("Giá trị đơn hàng (Order Value)", min_value=0.0)
            discount_applied = st.selectbox("Áp dụng giảm giá", options=[1, 0], 
                                           format_func=lambda x: "Có (1)" if x == 1 else "Không (0)")
            
            # Các trường có giá trị mặc định theo yêu cầu của bạn
            mood = st.selectbox("Tâm trạng (Mood)", options=moods, index=moods.index('Happy'))
            hunger_level = st.selectbox("Mức độ đói (Hunger Level)", options=hunger_levels, index=hunger_levels.index('Medium'))
            company = st.selectbox("Đi cùng ai (Company)", options=companies, index=companies.index('Alone'))

        submitted = st.form_submit_button("Lưu vào cơ sở dữ liệu")

        if submitted:
            if name:
                # Code lưu vào database (giữ nguyên logic SQLite trước đó)
                save_to_sqlite(city, name, age, cuisine, meal_type, order_value, discount_applied, mood, hunger_level, company)
                st.success(f"✅ Đã thêm dữ liệu cho {name}!")
            else:
                st.error("Vui lòng nhập tên khách hàng!")

def save_to_sqlite(city, name, age, cuisine, meal_type, order_value, discount_applied, mood, hunger_level, company):
    conn = sqlite3.connect('customer_data.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO orders (city, name, age, cuisine, meal_type, order_value, discount_applied, mood, hunger_level, company)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (city, name, age, cuisine, meal_type, order_value, discount_applied, mood, hunger_level, company))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()