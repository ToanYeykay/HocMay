import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

# 1. Cấu hình trang
st.set_page_config(page_title="Food Delivery Predictor", layout="wide")

# 2. Load các mô hình đã lưu
@st.cache_resource
def load_models():
    with open('model_rating.pkl', 'rb') as f:
        m_rating = pickle.load(f)
    with open('model_repeat.pkl', 'rb') as f:
        m_repeat = pickle.load(f)
    with open('model_columns.pkl', 'rb') as f:
        cols = pickle.load(f)
    return m_rating, m_repeat, cols

model_rating, model_repeat, model_columns = load_models()

# 3. Giao diện Sidebar - Nhập thông tin đơn hàng
st.sidebar.header("Nhập thông tin đơn hàng")

def user_input_features():
    age = st.sidebar.slider("Tuổi khách hàng", 18, 70, 30)
    order_value = st.sidebar.number_input("Giá trị đơn hàng (INR)", 100, 5000, 500)
    delivery_fee = st.sidebar.number_input("Phí giao hàng (INR)", 0, 200, 50)
    
    # Các thông tin phân loại
    city = st.sidebar.selectbox("Thành phố", ["Pune", "Mumbai", "Delhi", "Chandigarh", "Bangalore", "Hyderabad", "Kolkata", "Chennai", "Ahmedabad", "Jaipur"])
    order_time = st.sidebar.selectbox("Thời gian đặt", ["Morning", "Afternoon", "Evening", "Night"])
    cuisine = st.sidebar.selectbox("Ẩm thực", ["Chinese", "South Indian", "Biryani", "North Indian", "Fast Food", "Continental", "Desserts", "Italian", "Mexican", "Japanese"])
    mood = st.sidebar.selectbox("Tâm trạng", ["Happy", "Lazy", "Celebrating", "Hungry", "Stressed"])
    
    # Tạo dictionary dữ liệu
    data = {
        'age': age,
        'order_value': order_value,
        'delivery_fee': delivery_fee,
        'city': city,
        'order_time': order_time,
        'cuisine': cuisine,
        'mood': mood,
        # Thêm các cột mặc định khác nếu cần thiết giống như lúc train...
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# 4. Tiền xử lý dữ liệu nhập vào (One-hot Encoding)
# Bước này cực kỳ quan trọng để khớp với model_columns
input_encoded = pd.get_dummies(input_df)
final_input = pd.DataFrame(columns=model_columns).fillna(0)

for col in input_encoded.columns:
    if col in final_input.columns:
        final_input[col] = input_encoded[col]

# 5. Dự đoán
st.title("🍔 Hệ thống dự báo hành vi khách hàng")
st.write("Dựa trên mô hình XGBoost đã huấn luyện.")

if st.button("Bắt đầu dự báo"):
    # BƯỚC 1: Dự đoán Rating
    rating_raw = model_rating.predict(final_input)
    # Áp dụng hàm stretch giống trong code của bạn
    rating_val = np.clip(3.0 + (rating_raw - 3.0) * 1.5, 1, 5)[0]
    
    # BƯỚC 2: Dự đoán Repeat Order
    # Thêm cột predicted_rating vào đầu vào của Model 2
    final_input_repeat = final_input.copy()
    final_input_repeat['predicted_rating'] = rating_val
    
    prob_repeat = model_repeat.predict_proba(final_input_repeat)[:, 1][0]
    is_repeat = prob_repeat >= 0.47
    
    # Hiển thị kết quả
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dự đoán Rating", f"{rating_val:.2f} ⭐")
    with col2:
        status = "Sẽ quay lại ✅" if is_repeat else "Không quay lại ❌"
        st.metric("Khả năng đặt lại", status, f"{prob_repeat*100:.1f}% xác suất")

    if is_repeat:
        st.success("Khách hàng này có tiềm năng cao! Hãy gửi thêm ưu đãi.")
    else:
        st.warning("Khách hàng có nguy cơ rời bỏ. Cần cải thiện dịch vụ.")