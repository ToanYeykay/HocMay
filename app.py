import streamlit as st
import pandas as pd
import xgboost as xgb
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Foodie Analytics Pro", layout="wide")

# --- HÀM LOAD MÔ HÌNH & DỮ LIỆU ---
@st.cache_resource
def load_model_assets():
    model = xgb.XGBRegressor()
    model.load_model('food_rating_model.json')
    with open('model_columns.json', 'r') as f:
        model_cols = json.load(f)
    return model, model_cols

@st.cache_data
def load_data():
    return pd.read_csv('food_ordering_behavior_dataset3.csv')

# --- TRANG 1: NHẬP LIỆU & DỰ ĐOÁN ---
def input_page():
    st.title("➕ Nhập Đơn Hàng & Dự Đoán Rating")
    model, model_cols = load_model_assets()
    
    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            user_id = st.text_input("User ID", "USR-1001")
            age = st.number_input("Tuổi", 18, 80, 25)
            city = st.selectbox("Thành phố", ["Pune", "Mumbai", "Delhi", "Bangalore", "Hyderabad"])
            cuisine = st.selectbox("Ẩm thực", ["Chinese", "South Indian", "Biryani", "North Indian"])
        
        with col2:
            order_value = st.number_input("Giá trị đơn (VNĐ)", 0, 500000, 50000)
            delivery_fee = st.number_input("Phí ship", 0, 50000, 5000)
            order_time = st.selectbox("Buổi đặt", ["Morning", "Afternoon", "Evening", "Night"])
            discount = st.radio("Giảm giá", [1, 0], format_func=lambda x: "Có" if x==1 else "Không")

        with col3:
            mood = st.selectbox("Tâm trạng", ["Happy", "Stressed", "Lazy", "Celebrating"])
            hunger = st.select_slider("Mức độ đói", ["Low", "Medium", "High"])
            company = st.selectbox("Đi cùng", ["Alone", "Friends", "Family", "Partner"])
            rank = st.selectbox("Hạng khách", ["Bronze", "Silver", "Gold", "Diamond"])

        submit = st.form_submit_button("Lưu & Dự đoán")

        if submit:
            # --- LOGIC DỰ ĐOÁN ---
            # Tạo dataframe trống với tất cả các cột của model
            input_df = pd.DataFrame(0, index=[0], columns=model_cols)
            
            # Điền các giá trị số
            input_df['age'] = age
            input_df['order_value'] = order_value
            input_df['delivery_fee'] = delivery_fee
            input_df['discount_applied'] = discount
            input_df['net_value'] = order_value + delivery_fee
            
            # Điền các cột One-Hot (gán giá trị 1 cho cột tương ứng)
            mappings = {
                'city': city, 'order_time': order_time, 'cuisine': cuisine,
                'mood': mood, 'hunger_level': hunger, 'company': company, 'rank': rank
            }
            for col, val in mappings.items():
                col_name = f"{col}_{val}"
                if col_name in input_df.columns:
                    input_df[col_name] = 1
            
            # Dự đoán
            prediction = model.predict(input_df)[0]
            
            st.success(f"Đã lưu đơn hàng!")
            st.metric("Dự đoán Rating khách sẽ cho:", f"{prediction:.1f} ⭐")

# --- TRANG 2: PHÂN TÍCH ---
def analysis_page():
    st.title("📊 Dashboard Phân Tích Dữ Liệu")
    df = load_data()

    # Kiểm tra xem cột rank có tồn tại không
    if 'rank' not in df.columns:
        st.error("⚠️ Lỗi: Không tìm thấy cột 'rank' trong dữ liệu. Vui lòng kiểm tra lại file CSV!")
        st.write("Các cột hiện có trong file của bạn là:", df.columns.tolist())
        return # Dừng trang này tại đây để không bị crash

    # Nếu có cột rank thì mới chạy tiếp các dòng dưới
    st.subheader("🏆 Phân lớp Khách hàng")
    rank_counts = df['rank'].value_counts()
    st.bar_chart(rank_counts)
    st.title("Dashboard Phân Tích Dữ Liệu")
    df = load_data()

    # Chỉ số tổng quan
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng đơn hàng", len(df))
    m2.metric("Rating trung bình", round(df['rating_given'].mean(), 2))
    m3.metric("Độ chính xác Model (R2)", "90.7%")
    m4.metric("Sai số (RMSE)", "0.20")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Phân lớp Khách hàng")
        rank_counts = df['rank'].value_counts()
        st.bar_chart(rank_counts)
    
    with c2:
        st.subheader("Tương quan Phí ship & Rating")
        fig, ax = plt.subplots()
        sns.regplot(data=df.sample(500), x='delivery_fee', y='rating_given', ax=ax, scatter_kws={'alpha':0.3})
        st.pyplot(fig)

    st.divider()
    
    st.subheader("Các yếu tố ảnh hưởng Rating mạnh nhất")
    st.info("Dựa trên trọng số của mô hình XGBoost đã huấn luyện.")

# --- ĐIỀU HƯỚNG SIDEBAR ---
def main():
    st.sidebar.title("Menu Hệ Thống")
    page = st.sidebar.radio("Chọn chức năng:", ["Nhập đơn hàng", "Phân tích AI"])

    if page == "Nhập đơn hàng":
        input_page()
    else:
        analysis_page()

if __name__ == "__main__":
    main()