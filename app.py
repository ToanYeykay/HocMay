import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 0. CẤU HÌNH TRANG ---
st.set_page_config(page_title="Phân tích khách hàng Foodie", layout="wide")

# --- 1. HÀM LOAD TÀI NGUYÊN (Dùng Cache) ---
@st.cache_data
def load_data():
    # Tên file phải khớp chính xác trên GitHub
    file_path = 'food_ordering_behavior_dataset.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path).head(5000)
    return pd.DataFrame() 

@st.cache_resource
def load_models():
    model_dir = 'models'
    
    # Load danh sách cột
    col_path = os.path.join(model_dir, 'model_columns.pkl')
    with open(col_path, 'rb') as f:
        m_cols = pickle.load(f)
        
    # Load Model Rating
    r_path = os.path.join(model_dir, 'model_rating.pkl')
    with open(r_path, 'rb') as f:
        m_rating = pickle.load(f)
        
    # Load Model Repeat
    rep_path = os.path.join(model_dir, 'model_repeat.pkl')
    with open(rep_path, 'rb') as f:
        m_repeat = pickle.load(f)
        
    return m_rating, m_repeat, m_cols

# Thực thi load dữ liệu và model
df = load_data()
try:
    model_rating, model_repeat, model_columns = load_models()
except Exception as e:
    st.error(f"⚠️ Lỗi: Không tìm thấy thư mục 'models' hoặc file .pkl. Chi tiết: {e}")
    st.stop()

# --- 2. THANH ĐIỀU HƯỚNG ---
st.sidebar.title("Menu Chính")
page = st.sidebar.radio("Di chuyển đến:", [
    "🏠 Giới thiệu & EDA", 
    "🔮 Dự báo thông minh", 
    "📊 Đánh giá mô hình"
])

# ---------------------------------------------------------
# TRANG 1: GIỚI THIỆU & EDA
# ---------------------------------------------------------
if page == "🏠 Giới thiệu & EDA":
    st.title("📋 Khám phá Dữ liệu khách hàng")
    
    with st.expander("ℹ️ Thông tin dự án", expanded=True):
        st.write("**Đề tài:** Dự báo hành vi khách hàng ngành Food Delivery")
        st.write("**Mục tiêu:** Sử dụng XGBoost để dự đoán sự hài lòng (Rating) và khả năng quay lại.")

    st.subheader("1. Xem trước dữ liệu (5000 dòng đầu)")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("2. Phân tích trực quan")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Phân phối Rating**")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='rating_given', palette='magma', ax=ax1)
        st.pyplot(fig1)
        
    with col2:
        st.write("**Tỉ lệ khách hàng quay lại**")
        fig2, ax2 = plt.subplots()
        df['is_repeat_order'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], ax=ax2)
        st.pyplot(fig2)

# ---------------------------------------------------------
# TRANG 2: TRIỂN KHAI MÔ HÌNH
# ---------------------------------------------------------
elif page == "🔮 Dự báo thông minh":
    st.title("🔮 Dự báo hành vi khách hàng")
    
    st.markdown("---")
    c_input, c_result = st.columns([1, 1.5])
    
    with c_input:
        st.subheader("📝 Nhập thông tin")
        u_age = st.slider("Độ tuổi", 18, 70, 25)
        u_val = st.number_input("Giá trị đơn (INR)", 100, 5000, 450)
        u_fee = st.number_input("Phí ship (INR)", 0, 200, 40)
        u_city = st.selectbox("Thành phố", ["Pune", "Mumbai", "Delhi", "Bangalore"])
        u_mood = st.selectbox("Tâm trạng", ["Happy", "Lazy", "Hungry", "Celebrating"])
        u_rain = st.radio("Trời đang mưa?", ["No", "Yes"])
        
        predict_btn = st.button("🚀 Chạy dự báo", use_container_width=True)

    with c_result:
        st.subheader("📊 Kết quả dự báo")
        if predict_btn:
            # Tiền xử lý dữ liệu đầu vào
            input_df = pd.DataFrame(columns=model_columns).fillna(0)
            input_df.loc[0, 'age'] = u_age
            input_df.loc[0, 'order_value'] = u_val
            input_df.loc[0, 'delivery_fee'] = u_fee
            
            # Gán giá trị cho One-hot Encoding
            if f'city_{u_city}' in model_columns: input_df.loc[0, f'city_{u_city}'] = 1
            if f'mood_{u_mood}' in model_columns: input_df.loc[0, f'mood_{u_mood}'] = 1
            if f'rainy_weather_Yes' in model_columns and u_rain == "Yes": input_df.loc[0, 'rainy_weather_Yes'] = 1

            # Dự báo Rating
            r_raw = model_rating.predict(input_df)[0]
            r_final = np.clip(3.0 + (r_raw - 3.0) * 1.5, 1, 5) 
            
            # Dự báo Repeat Order (dựa trên Rating vừa đoán)
            input_df['predicted_rating'] = r_final
            prob_repeat = model_repeat.predict_proba(input_df)[:, 1][0]
            
            # Hiển thị kết quả trực quan
            st.metric("⭐️ Dự đoán số sao", f"{r_final:.1f} / 5.0")
            
            st.write(f"**Xác suất quay lại đặt hàng:** {prob_repeat*100:.2f}%")
            st.progress(float(prob_repeat))
            
            if prob_repeat >= 0.47:
                st.success("✅ Dự báo: Khách hàng CÓ khả năng quay lại.")
            else:
                st.warning("⚠️ Dự báo: Khách hàng ÍT có khả năng quay lại.")

# ---------------------------------------------------------
# TRANG 3: ĐÁNH GIÁ & HIỆU NĂNG
# ---------------------------------------------------------
else:
    st.title("📈 Đánh giá hiệu năng của XGBoost")
    
    col_eval1, col_eval2 = st.columns(2)
    with col_eval1:
        st.subheader("1. Chỉ số đo lường")
        st.write("- **MAE (Rating):** 1.2112")
        st.write("- **F1-Score (Repeat):** ~0.69")
        st.write("- **Ngưỡng tối ưu:** 0.47")

    with col_eval2:
        st.subheader("2. Ma trận nhầm lẫn")
        if os.path.exists("image_408e89.png"):
            st.image("image_408e89.png", caption="Confusion Matrix (Threshold 0.47)")
        else:
            st.info("💡 Mẹo: Upload ảnh Confusion Matrix của bạn lên GitHub để hiển thị tại đây.")

    st.subheader("3. Nhận định mô hình")
    st.markdown("""
    - **Ưu điểm:** Mô hình đã nhận diện được tầm quan trọng của Rating đối với hành vi đặt hàng lại.
    - **Hạn chế:** Sai số Rating còn bị ảnh hưởng bởi tính chủ quan của người dùng.
    - **Giải pháp:** Cần thu thập thêm dữ liệu về thời gian giao hàng thực tế để tăng độ chính xác.
    """)