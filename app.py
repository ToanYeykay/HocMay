import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Food Delivery Analysis", layout="wide")

# --- 1. SỬ DỤNG CACHE ĐỂ LOAD DỮ LIỆU & MODEL ---
@st.cache_data
def load_data():
    # Lưu ý: Bạn nên để file csv này trong thư mục dự án
    df = pd.read_csv('food_ordering_behavior_dataset.csv')
    return df

@st.cache_resource
def load_models():
    # 1. Load danh sách cột (vẫn dùng pickle vì nó là list)
    with open('models/model_columns.pkl', 'rb') as f:
        m_cols = pickle.load(f)
    
    # 2. Load Model Rating (.json)
    m_rating = xgb.XGBRegressor()
    m_rating.load_model('models/model_rating.pkl')
    
    # 3. Load Model Repeat (.json)
    m_repeat = xgb.XGBClassifier()
    m_repeat.load_model('models/model_repeat.pkl')
    
    return m_rating, m_repeat, m_cols

# Load tài nguyên
df = load_data()
model_rating, model_repeat, model_columns = load_models()

# --- THANH ĐIỀU HƯỚNG (SIDEBAR) ---
st.sidebar.title("Menu Điều Hướng")
page = st.sidebar.radio("Chọn trang:", [
    "Trang 1: Khám phá dữ liệu (EDA)", 
    "Trang 2: Triển khai mô hình", 
    "Trang 3: Đánh giá & Hiệu năng"
])

# ---------------------------------------------------------
# TRANG 1: GIỚI THIỆU & EDA
# ---------------------------------------------------------
if page == "Trang 1: Khám phá dữ liệu (EDA)":
    st.title("📊 Giới thiệu & Khám phá dữ liệu")
    
    st.info("""
    **Thông tin sinh viên:**
    - Họ và tên: [Tên của bạn]
    - MSSV: [MSSV của bạn]
    - Đề tài: Dự báo mức độ hài lòng và hành vi tái mua hàng trong ngành giao đồ ăn.
    """)
    
    st.subheader("1. Giá trị thực tiễn")
    st.write("Ứng dụng giúp nhà hàng nhận diện khách hàng không hài lòng và dự báo khả năng quay lại, từ đó tối ưu hóa các chiến dịch Marketing và cải thiện dịch vụ.")

    st.subheader("2. Dữ liệu thô (Raw Data)")
    st.dataframe(df.head(10))

    st.subheader("3. Biểu đồ phân tích")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Phân phối Rating (Nhãn mục tiêu 1)**")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='rating_given', palette='viridis', ax=ax1)
        st.pyplot(fig1)
        
    with col2:
        st.write("**Tỷ lệ khách hàng quay lại (Nhãn mục tiêu 2)**")
        fig2, ax2 = plt.subplots()
        df['is_repeat_order'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax2, colors=['#ff9999','#66b3ff'])
        st.pyplot(fig2)

    st.write("**Nhận xét:** Dữ liệu có sự phân bổ khá đồng đều ở các mức Rating. Tuy nhiên, các đặc trưng như `delivery_fee` và `order_value` có sự tương quan mạnh đến quyết định quay lại của khách.")

# ---------------------------------------------------------
# TRANG 2: TRIỂN KHAI MÔ HÌNH
# ---------------------------------------------------------
elif page == "Trang 2: Triển khai mô hình":
    st.title("🚀 Dự báo trực tiếp")
    
    with st.form("input_form"):
        st.subheader("Nhập thông tin đơn hàng")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Tuổi", 18, 80, 25)
            order_val = st.number_input("Giá trị đơn (INR)", 100, 5000, 500)
            delivery_fee = st.number_input("Phí ship", 0, 200, 30)
        with col2:
            city = st.selectbox("Thành phố", ["Pune", "Mumbai", "Delhi", "Bangalore"])
            mood = st.selectbox("Tâm trạng", ["Happy", "Hungry", "Lazy", "Stressed"])
            order_time = st.selectbox("Thời gian", ["Morning", "Afternoon", "Evening", "Night"])
        with col3:
            cuisine = st.selectbox("Ẩm thực", ["Chinese", "North Indian", "South Indian", "Fast Food"])
            hunger = st.selectbox("Mức đói", ["Low", "Medium", "High"])
            rain = st.selectbox("Trời mưa?", ["Yes", "No"])

        submit = st.form_submit_button("Dự báo ngay")

    if submit:
        # Tiền xử lý dữ liệu nhập (phải khớp với model_columns)
        input_data = pd.DataFrame(columns=model_columns).fillna(0)
        # Gán giá trị số
        input_data.loc[0, 'age'] = age
        input_data.loc[0, 'order_value'] = order_val
        input_data.loc[0, 'delivery_fee'] = delivery_fee
        # Gán One-hot (ví dụ đơn giản)
        if f'city_{city}' in model_columns: input_data.loc[0, f'city_{city}'] = 1
        # ... (làm tương tự cho các biến selectbox khác)

        # Dự đoán chuỗi
        rating_raw = model_rating.predict(input_data)[0]
        # Hàm stretch như trong code gốc
        rating_pred = np.clip(3.0 + (rating_raw - 3.0) * 1.5, 1, 5)
        
        input_data['predicted_rating'] = rating_pred
        prob_repeat = model_repeat.predict_proba(input_data)[:, 1][0]
        
        # Hiển thị
        st.success(f"⭐ **Dự đoán đánh giá:** {rating_pred:.1f} sao")
        st.info(f"🔁 **Xác suất quay lại:** {prob_repeat*100:.2f}%")
        
        if prob_repeat >= 0.47:
            st.balloons()
            st.write("👉 **Kết luận:** Khách hàng này **CÓ** khả năng cao sẽ quay lại!")
        else:
            st.write("👉 **Kết luận:** Khách hàng này **ÍT** có khả năng quay lại.")

# ---------------------------------------------------------
# TRANG 3: ĐÁNH GIÁ & HIỆU NĂNG
# ---------------------------------------------------------
else:
    st.title("📈 Đánh giá hiệu năng mô hình")
    
    st.subheader("1. Chỉ số đo lường")
    c1, c2 = st.columns(2)
    c1.metric("Rating MAE (Hồi quy)", "1.2112")
    c2.metric("Repeat F1-Score (Phân loại)", "0.69")

    st.subheader("2. Ma trận nhầm lẫn (Confusion Matrix)")
    # Giả sử bạn đã tính