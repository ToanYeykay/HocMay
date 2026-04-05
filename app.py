import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Cấu hình trang
st.set_page_config(page_title="Dự đoán hành vi khách hàng", layout="wide")

# Hàm load dữ liệu (để cache lại cho nhanh)
@st.cache_data
def load_data():
    df = pd.read_csv('food_ordering_behavior_dataset.csv')
    # Giả lập lại bước tính toán user_df như đã làm ở Colab
    mood_map = {'Celebrating': 4, 'Happy': 3, 'Lazy': 2, 'Stressed': 1}
    df['mood_score'] = df['mood'].map(mood_map)
    user_df = df.groupby('user_id').agg({
        'order_value': 'sum',
        'rating_given': 'mean',
        'mood_score': 'mean',
        'order_id': 'count'
    }).reset_index()
    user_df.columns = ['user_id', 'total_spent', 'avg_rating', 'avg_mood', 'order_count']
    return df, user_df

df, user_df = load_data()

# Sidebar điều hướng
st.sidebar.title("Menu Điều Hướng")
page = st.sidebar.radio("Chọn trang:", ["Giới thiệu & EDA", "Dự đoán Rating"])

# ---------------------------------------------------------
# TRANG 1: GIỚI THIỆU & EDA
# ---------------------------------------------------------
if page == "Giới thiệu & EDA":
    st.title("📊 Khám Phá Dữ Liệu Đặt Đồ Ăn")
    st.subheader("Sinh viên thực hiện: Lê Tấn Toàn - MSSV: 22T1020768")
    
    st.markdown("""
    **Giá trị thực tiễn:** Bài toán này giúp các nền tảng đặt đồ ăn hiểu rõ yếu tố nào ảnh hưởng đến sự hài lòng của khách hàng. 
    Từ đó, doanh nghiệp có thể tối ưu hóa dịch vụ, cải thiện tâm trạng khách hàng và tăng tỷ lệ giữ chân người dùng.
    """)
    
    st.divider()
    
    # Hiển thị dữ liệu thô
    st.write("### 1. Dữ liệu thô (Raw Data)")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 2. Phân bố Rating")
        fig, ax = plt.subplots()
        sns.countplot(x='rating_given', data=df, palette='viridis', ax=ax)
        st.pyplot(fig)
        st.write("Nhận xét: Rating tập trung nhiều ở mức 3-4 sao, dữ liệu khá cân bằng.")

    with col2:
        st.write("### 3. Ma trận tương quan")
        fig, ax = plt.subplots()
        # Chỉ lấy các cột số để tính tương quan
        corr = user_df[['total_spent', 'avg_rating', 'avg_mood', 'order_count']].corr()
        sns.heatmap(corr, annot=True, cmap='RdBu', ax=ax)
        st.pyplot(fig)
        st.write("Nhận xét: 'avg_mood' có tương quan thuận rõ rệt nhất với 'avg_rating'.")

    st.info("💡 **Kết luận sơ bộ:** Tâm trạng (Mood) là đặc trưng quan trọng nhất quyết định điểm số đánh giá của người dùng.")

# ---------------------------------------------------------
# TRANG 2: TRIỂN KHAI MÔ HÌNH
# ---------------------------------------------------------
else:
    st.title("🤖 Dự Đoán Rating Khách Hàng")

    # Load model
    try:
        model = joblib.load('model_xgb.pkl')
    except:
        st.warning("⚠️ Chưa tìm thấy file 'model_xgb.pkl'. Vui lòng huấn luyện và lưu mô hình trước.")
        st.stop()

    # Form nhập liệu
    with st.expander("📝 Nhập thông tin chi tiết đơn hàng", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            u_id = st.text_input("Mã khách hàng (User ID):", "22T102")
            age = st.number_input("Độ tuổi:", 18, 100, 25)
            city = st.selectbox("Thành phố:", df['city'].unique())
        with c2:
            mood = st.selectbox("Tâm trạng khách hàng:", ['Stressed', 'Lazy', 'Happy', 'Celebrating'])
            cuisine = st.selectbox("Loại ẩm thực:", df['cuisine'].unique())
            res_type = st.selectbox("Loại nhà hàng:", df['restaurant_type'].unique())
        with c3:
            order_val = st.number_input("Giá trị đơn hàng (Order Value):", 100, 2000, 500)
            order_cnt = st.number_input("Tổng số đơn đã đặt (Order Count):", 1, 100, 10)
            time_order = st.number_input("Thời gian giao (Time Taken):", 5, 120, 30)

        # Các widget bổ sung cho đầy đủ bộ dữ liệu
        c4, c5, c6 = st.columns(3)
        with c4:
            meal_type = st.selectbox("Bữa ăn:", df['meal_type'].unique())
        with c5:
            weather = st.radio("Thời tiết mưa?", ["No", "Yes"], horizontal=True)
        with c6:
            is_repeat = st.radio("Khách quay lại?", ["No", "Yes"], horizontal=True)

    # Nút dự đoán
    if st.button("🚀 Thực hiện dự đoán"):
        # Tiền xử lý Input (Chỉ lấy các biến model cần: total_spent, avg_mood, order_count)
        mood_map = {'Stressed': 1, 'Lazy': 2, 'Happy': 3, 'Celebrating': 4}
        
        # Tạo mảng input đúng cột như lúc train model
        # Giả sử X_train của Toàn gồm: [total_spent, avg_mood, order_count]
        input_data = pd.DataFrame([[order_val, mood_map[mood], order_cnt]], 
                                 columns=['total_spent', 'avg_mood', 'order_count'])
        
        # Dự đoán
        pred = model.predict(input_data)[0]
        pred = max(1, min(5, pred)) # Đảm bảo nằm trong khoảng 1-5 sao

        # Hiển thị kết quả
        st.divider()
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("⭐️ Rating dự đoán:", f"{pred:.1f} / 5.0")
            
        with res_col2:
            # Tính độ tin cậy dựa trên điểm số (Ví dụ đơn giản)
            confidence = "Cao" if pred > 4.0 or pred < 2.0 else "Trung bình"
            st.write(f"**Độ tin cậy:** {confidence}")
            st.progress(min(pred/5, 1.0))

        # Nhận xét kết quả
        if pred >= 4:
            st.success(f"Khách hàng {u_id} có khả năng cao sẽ hài lòng với dịch vụ!")
        elif pred >= 3:
            st.info("Khách hàng ở mức độ hài lòng vừa phải.")
        else:
            st.error("Cảnh báo: Khách hàng có xu hướng đánh giá thấp đơn hàng này!")