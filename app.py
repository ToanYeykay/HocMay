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
    return pd.DataFrame() # Trả về df trống nếu ko thấy file

@st.cache_resource
def load_models():
    # Sử dụng os.path.join để an toàn trên Linux/Windows
    model_dir = 'models'
    
    # Load danh sách cột (Bắt buộc phải có để input đúng định dạng)
    col_path = os.path.join(model_dir, 'model_columns.pkl')
    with open(col_path, 'rb') as f:
        m_cols = pickle.load(f)
        
    # Load Model Rating (XGBoost Regressor)
    # Lưu ý: Nếu bạn dùng file .json thì đổi đuôi và dùng load_model()
    r_path = os.path.join(model_dir, 'model_rating.pkl')
    with open(r_path, 'rb') as f:
        m_rating = pickle.load(f)
        
    # Load Model Repeat (XGBoost Classifier)
    rep_path = os.path.join(model_dir, 'model_repeat.pkl')
    with open(rep_path, 'rb') as f:
        m_repeat = pickle.load(f)
        
    return m_rating, m_repeat, m_cols

# Thực thi load
df = load_data()
try:
    model_rating, model_repeat, model_columns = load_models()
except Exception as e:
    st.error(f"⚠️ Lỗi cấu hình: Hãy đảm bảo thư mục 'models' có đủ 3 file .pkl. Lỗi: {e}")
    st.stop()

# --- 2. THANH ĐIỀU HƯỚNG ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1532/1532688.png", width=100)
st.sidebar.title("Dự Báo Khách Hàng")
page = st.sidebar.radio("Chọn trang hiển thị:", [
    "🏠 Giới thiệu & EDA", 
    "🔮 Dự báo thông minh", 
    "📊 Đánh giá mô hình"
])

# ---------------------------------------------------------
# TRANG 1: GIỚI THIỆU & EDA
# ---------------------------------------------------------
if page == "🏠 Giới thiệu & EDA":
    st.title("📋 Giới thiệu Dự án & Khám phá Dữ liệu")
    
    # Phần thông tin SV
    with st.expander("ℹ️ Thông tin sinh viên & Đề tài", expanded=True):
        st.write("**Họ tên:** [Tên của bạn]")
        st.write("**MSSV:** [Mã số sinh viên]")
        st.write("**Giá trị:** Dự báo sự hài lòng giúp nhà hàng cải thiện dịch vụ kịp thời.")

    # Hiển thị dữ liệu
    st.subheader("1. Dữ liệu mẫu")
    st.dataframe(df.head(10), use_container_width=True)

    # Biểu đồ
    st.subheader("2. Phân tích trực quan")
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("**Phân phối Rating thực tế**")
        fig, ax = plt.subplots()
        sns.histplot(df['rating_given'], bins=5, color='orange', ax=ax)
        st.pyplot(fig)
        
    with c2:
        st.write("**Tương quan giữa Phí ship và Rating**")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x='rating_given', y='delivery_fee', data=df, palette='Set2', ax=ax2)
        st.pyplot(fig2)

    st.info("💡 **Nhận xét:** Dữ liệu cho thấy phí giao hàng càng cao, khách hàng có xu hướng đánh giá khắt khe hơn.")

# ---------------------------------------------------------
# TRANG 2: TRIỂN KHAI MÔ HÌNH
# ---------------------------------------------------------
elif page == "🔮 Dự báo thông minh":
    st.title("🔮 Hệ thống dự báo hành vi")
    
    st.markdown("---")
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        st.subheader("Nhập thông tin")
        u_age = st.slider("Tuổi", 18, 70, 25)
        u_val = st.number_input("Giá trị đơn (INR)", 100, 5000, 450)
        u_fee = st.number_input("Phí ship (INR)", 0, 200, 40)
        u_city = st.selectbox("Thành phố", ["Pune", "Mumbai", "Delhi", "Bangalore"])
        u_mood = st.selectbox("Tâm trạng", ["Happy", "Lazy", "Hungry", "Celebrating"])
        u_rain = st.radio("Trời mưa?", ["No", "Yes"])
        
        btn = st.button("🚀 Chạy dự báo", use_container_width=True)

    with col_b:
        st.subheader("Kết quả dự báo")
        if btn:
            # 1. Tiền xử lý Input (Khớp với One-hot Encoding)
            input_df = pd.DataFrame(columns=model_columns).fillna(0)
            input_df.loc[0, 'age'] = u_age
            input_df.loc[0, 'order_value'] = u_val
            input_df.loc[0, 'delivery_fee'] = u_fee
            
            # Gán 1 cho các cột One-hot
            if f'city_{u_city}' in model_columns: input_df.loc[0, f'city_{u_city}'] = 1
            if f'mood_{u_mood}' in model_columns: input_df.loc[0, f'mood_{u_mood}'] = 1
            if f'rainy_weather_Yes' in model_columns and u_rain == "Yes": input_df.loc[0, 'rainy_weather_Yes'] = 1

            # 2. Dự báo Rating
            r_pred_raw = model_rating.predict(input_df)[0]
            r_final = np.clip(3.0 + (r_pred_raw - 3.0) * 1.5, 1, 5) # Hàm stretch của bạn
            
            # 3. Dự báo Repeat Order
            input_df['predicted_rating'] = r_final
            prob_rep = model_repeat.predict_proba(input_df)[:, 1][0]
            
            # Hiển thị
            st.metric("⭐️ Đánh giá dự kiến", f"{r_final:.1f} / 5.0")
            
            prog_color = "green" if prob_rep >= 0.47 else "red"
            st.write(f"**Xác suất quay lại:** {prob_rep*100:.1f}%")
            st.progress(prob_rep)
            
            if prob_rep >= 0.47:
                st.success("✅ KẾT LUẬN: Khách hàng trung thành. Nên duy trì chăm sóc.")
            else:
                st.warning("⚠️ KẾT LUẬN: Nguy cơ rời bỏ cao. Cần tặng voucher giảm giá!")

# ---------------------------------------------------------
# TRANG 3: ĐÁNH GIÁ & HIỆU NĂNG
# ---------------------------------------------------------
else:
    st.title("📈 Phân tích hiệu năng mô hình")
    
    col_x, col_y = st.columns(2)
    with col_x:
        st.subheader("1. Chỉ số đo lường")
        st.write("- **Rating (XGBRegressor):** MAE = 1.2112")
        st.write("- **Repeat (XGBClassifier):** F1-Score = 0.69 (tại Threshold 0.47)")
        st.write("- **Độ chính xác ±1 sao:** 60.90%")

    with col_y:
        st.subheader("2. Ma trận nhầm lẫn")
        # Thay 'image_408e89.png' bằng tên file ảnh thực tế của bạn trên GitHub
        if os.path.exists("image_408e89.png"):
            st.image("image_408e89.png", caption="Confusion Matrix - Threshold 0.47")
        else:
            st.info("Vui lòng upload ảnh Confusion Matrix lên GitHub để hiển thị.")

    st.subheader("3. Phân tích & Hướng cải thiện")
    st.markdown("""
    * **Phân tích:** Biến `predicted_rating` đứng đầu về mức độ quan trọng, chứng tỏ sự hài lòng ảnh hưởng trực tiếp đến việc đặt lại.
    * **Hạn chế:** Sai số MAE 1.21 còn cao do dữ liệu thiếu các thông tin về chất lượng món ăn thực tế.
    * **Cải thiện:** Cần thêm các biến về thời gian giao hàng thực tế (Actual Delivery Time) để mô hình chính xác hơn.
    """)