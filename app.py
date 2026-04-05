import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Cấu hình giao diện
st.set_page_config(page_title="Dự đoán Food Rating - Lê Tấn Toàn", layout="wide")

# --- LOAD DỮ LIỆU ---
@st.cache_data
def load_data():
    df = pd.read_csv('food_ordering_behavior_dataset.csv')
    return df

try:
    df = load_data()
except:
    st.error("⚠️ Thiếu file 'food_ordering_behavior_dataset.csv'!")

# Sidebar menu
st.sidebar.title("Menu")
menu = st.sidebar.radio("Chọn trang:", ["Trang 1: EDA", "Trang 2: Dự đoán Rating"])

# ---------------------------------------------------------
# TRANG 1: GIỚI THIỆU & EDA
# ---------------------------------------------------------
if menu == "Trang 1: EDA":
    st.title("📊 Khám Phá Dữ Liệu Đặt Đồ Ăn")
    st.info(f"**Sinh viên:** Lê Tấn Toàn | **MSSV:** 22T1020768")
    
    st.markdown("### Giá trị thực tiễn\nDự báo mức độ hài lòng giúp nhà hàng tối ưu vận hành và chăm sóc khách hàng tốt hơn.")
    
    st.write("#### Dữ liệu mẫu")
    st.dataframe(df.head(10))

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Phân bổ Rating**")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='rating_given', data=df, palette='viridis', ax=ax1)
        st.pyplot(fig1)
    with col2:
        st.write("**Tương quan các biến số**")
        fig2, ax2 = plt.subplots()
        sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='RdBu', ax=ax2)
        st.pyplot(fig2)

# ---------------------------------------------------------
# TRANG 2: TRIỂN KHAI MÔ HÌNH (Đã bỏ Repeat & Order Count)
# ---------------------------------------------------------
else:
    st.title("🤖 Dự Đoán Rating Đơn Hàng")

    # Load model
    try:
        model = joblib.load('model_xgb.pkl')
    except:
        st.warning("⚠️ Không tìm thấy file 'model_xgb.pkl'.")
        st.stop()

    st.subheader("Nhập thông tin đơn hàng")
    
    # Giao diện nhập liệu mới (Bỏ Repeat và Order Count)
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            u_id = st.text_input("Mã khách hàng (User ID):", "22T102")
            order_val = st.number_input("Giá trị đơn hàng (Order Value):", 100, 2000, 500)
            mood = st.selectbox("Tâm trạng khách hàng:", ['Stressed', 'Lazy', 'Happy', 'Celebrating'])
        
        with c2:
            time_order = st.number_input("Thời gian giao hàng thực tế (phút):", 5, 120, 30)
            res_type = st.selectbox("Loại nhà hàng:", df['restaurant_type'].unique())
            weather = st.radio("Thời tiết lúc giao:", ["Không mưa (No)", "Có mưa (Yes)"], horizontal=True)

    # Nút dự đoán
    if st.button("🚀 Thực hiện dự đoán"):
        # 1. Xử lý Logic Tiền xử lý
        mood_map = {'Stressed': 1, 'Lazy': 2, 'Happy': 3, 'Celebrating': 4}
        
        # 2. CHUẨN BỊ DỮ LIỆU ĐẦU VÀO CHO XGBOOST
        # Lưu ý: Vì Toàn đã bỏ 'order_count', mình sẽ giả định giá trị mặc định là 1 đơn hàng
        # để mô hình không bị thiếu cột khi dự đoán.
        input_data = pd.DataFrame([[
            order_val,          # Tương ứng cột 'total_spent'
            mood_map[mood],     # Tương ứng cột 'avg_mood'
            1                   # GIÁ TRỊ MẶC ĐỊNH cho 'order_count' (Do đã bỏ widget)
        ]], columns=['total_spent', 'avg_mood', 'order_count'])
        
        # 3. Dự đoán
        pred = model.predict(input_data)[0]
        pred = max(1, min(5, pred)) # Chặn trong khoảng 1-5 sao

        # 4. Hiển thị kết quả
        st.divider()
        res1, res2 = st.columns(2)
        with res1:
            st.metric("⭐️ Rating dự báo:", f"{pred:.2f} / 5.0")
        with res2:
            status = "Hài lòng" if pred >= 4 else "Bình thường" if pred >= 3 else "Không hài lòng"
            st.write(f"**Trạng thái dự kiến:** {status}")
            st.progress(float(max(0.0, min(pred/5, 1.0))))

        if pred >= 4:
            st.success(f"Khách hàng {u_id} nhiều khả năng sẽ đánh giá tốt!")
        elif pred < 3:
            st.error("Cảnh báo: Đơn hàng này có nguy cơ nhận đánh giá thấp!")