import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Dự đoán", layout="wide")

# Load mô hình và danh sách cột
@st.cache_resource
def load_model():
    model = joblib.load('model_xgb.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns

model, model_columns = load_model()

# --- SIDEBAR ĐIỀU HƯỚNG ---
st.sidebar.title("Menu Chính")
page = st.sidebar.radio("Chọn trang:", ["Thông tin đề tài", "Dự đoán Rating"])

# --- TRANG 1: THÔNG TIN ĐỀ TÀI & EDA ---
if page == "Thông tin đề tài":
    st.title("Phân tích dữ liệu Đặt hàng Đồ ăn")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info("""
        **Họ và tên:** Lê Tấn Toàn
        **MSSV:** 22T1020768
        **Đề tài:** Dự đoán mức độ hài lòng khách hàng F&B
        """)

    # Hiển thị dữ liệu mẫu (Giả sử bạn upload file data_sach.csv lên git)
    st.subheader("1. Dữ liệu thô (Trích đoạn)")
    try:
        df_sample = pd.read_csv('food_ordering_behavior_dataset3.csv')
        st.dataframe(df_sample.head(10))
        
        st.subheader("2. Biểu đồ phân tích (EDA)")
        c1, c2 = st.columns(2)
        
        with c1:
            st.write("**Phân bổ điểm Rating thực tế**")
            fig1, ax1 = plt.subplots()
            sns.countplot(data=df_sample, x='rating_given', palette='viridis', ax=ax1)
            st.pyplot(fig1)
            
        with c2:
            st.write("**Ma trận tương quan**")
            fig2, ax2 = plt.subplots()
            numeric_df = df_sample.select_dtypes(include=[np.number])
            sns.heatmap(numeric_df.corr(), cmap='RdBu', ax=ax2)
            st.pyplot(fig2)

        st.success("""
        **Nhận xét dữ liệu:**
        - Dữ liệu tập trung nhiều ở mức 3 và 4 sao (không bị lệch quá nặng).
        - Đặc trưng quan trọng nhất: **Tâm trạng (Mood)** và **Thời tiết (Rainy)**.
        """)
    except:
        st.warning("Hãy upload file 'data_sach.csv' lên GitHub để hiển thị biểu đồ.")

# --- TRANG 2: DỰ ĐOÁN RATING ---
elif page == "Dự đoán Rating":
    st.title("🤖 Dự đoán Rating thông minh")
    
    st.markdown("Nhập thông tin đơn hàng dưới đây để AI dự đoán số sao khách sẽ chấm:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mood = st.selectbox("Tâm trạng khách:", ['Happy', 'Stressed', 'Celebrating', 'Lazy'])
        net_value = st.number_input("Giá trị đơn hàng (VNĐ):", min_value=0, value=150000)
        day_type = st.selectbox("Loại ngày:", ["Weekday", "Weekend"])

    with col2:
        rainy = st.radio("Trời có mưa không?", ["No", "Yes"])
        delivery_fee = st.number_input("Phí Ship (VNĐ):", min_value=0, value=15000)
        is_repeat = st.radio("Khách quay lại?", ["No", "Yes"])

    with col3:
        rank = st.selectbox("Hạng khách hàng:", ['VIP (Kim cương)', 'Tiềm năng (Vàng)', 'Trung thành (Bạc)', 'Cần giữ chân (Đồng)'])
        time_taken = st.slider("Thời gian đặt hàng (phút):", 5, 120, 30)

    # --- XỬ LÝ LOGIC DỰ ĐOÁN ---
    if st.button("🚀 Dự đoán kết quả"):
        # 1. Tạo DataFrame từ input
        input_data = pd.DataFrame({
            'net_value': [net_value],
            'delivery_fee': [delivery_fee],
            'time_taken_to_order': [time_taken],
            'day_type': [1 if day_type == "Weekend" else 0],
            'rainy_weather': [1 if rainy == "Yes" else 0],
            'is_repeat_order': [1 if is_repeat == "Yes" else 0],
            'mood': [mood],
            'rank_category': [rank]
        })

        # 2. One-hot Encoding giống lúc train
        input_encoded = pd.get_dummies(input_data)
        
        # 3. Khớp cột với model_columns (Điền 0 cho các cột thiếu)
        final_input = pd.DataFrame(columns=model_columns)
        final_input = pd.concat([final_input, input_encoded], axis=0).fillna(0)
        final_input = final_input[model_columns] # Đảm bảo đúng thứ tự cột

        # 4. Dự đoán
        pred_raw = model.predict(final_input)[0]
        pred_int = int(np.clip(np.round(pred_raw), 1, 5))

        # 5. Hiển thị
        st.divider()
        st.subheader("KẾT QUẢ DỰ BÁO")
        
        star_html = "⭐" * pred_int
        st.markdown(f"### Dự đoán khách sẽ chấm: {star_html} ({pred_int} sao)")
        
        # Giả lập độ tin cậy (Dựa trên khoảng cách làm tròn)
        confidence = 100 - abs(pred_raw - pred_int) * 20 
        st.progress(confidence / 100)
        st.write(f"Độ tin cậy dự báo: {confidence:.1f}%")

        if pred_int <= 2:
            st.error("Cảnh báo: Khách hàng có khả năng cao sẽ không hài lòng!")
        elif pred_int >= 4:
            st.balloons()