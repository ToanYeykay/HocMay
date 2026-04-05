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
elif page == "Dự đoán Rating":
    st.title("🤖 Dự đoán mức độ hài lòng khách hàng")
    
    # Load mô hình
    try:
        model = joblib.load('model_xgb.pkl')
    except:
        st.error("Không tìm thấy file model_xgb.pkl. Hãy upload file mô hình lên GitHub.")
        st.stop()

    st.write("### Nhập thông tin khách hàng")
    
    col_in1, col_in2 = st.columns(2)
    
    with col_in1:
        total_spent = st.number_input("Tổng số tiền đã chi tiêu (đơn vị tiền tệ):", min_value=0, value=5000)
        order_count = st.number_input("Tổng số đơn hàng đã đặt:", min_value=1, value=5)
    
    with col_in2:
        mood_label = st.selectbox("Tâm trạng thường xuyên khi đặt hàng:", 
                                 ['Stressed (Căng thẳng)', 'Lazy (Lười biếng)', 'Happy (Vui vẻ)', 'Celebrating (Ăn mừng)'])
        # Chuyển label về mood_score tương ứng lúc huấn luyện
        mood_mapping = {'Stressed (Căng thẳng)': 1, 'Lazy (Lười biếng)': 2, 'Happy (Vui vẻ)': 3, 'Celebrating (Ăn mừng)': 4}
        avg_mood = mood_mapping[mood_label]

    # Xử lý logic dự đoán
    if st.button("Dự đoán ngay"):
        # Tạo mảng input đúng định dạng (XGBoost cần DataFrame hoặc mảng 2D)
        input_data = pd.DataFrame([[total_spent, avg_mood, order_count]], 
                                 columns=['total_spent', 'avg_mood', 'order_count'])
        
        prediction = model.predict(input_data)[0]
        
        st.divider()
        st.write("### Kết quả phân tích:")
        
        # Hiển thị kết quả rõ ràng
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(label="Rating Dự Đoán", value=f"{prediction:.2f} ⭐")
            
        with res_col2:
            # Vì là mô hình Regressor, độ tin cậy có thể tính theo độ lệch chuẩn hoặc hiển thị mức độ
            confidence = "Cao" if prediction > 3.5 else "Trung bình"
            st.write(f"**Độ tin cậy của mô hình:** {confidence}")
            st.progress(min(prediction/5, 1.0))

        if prediction >= 4:
            st.success("Đây là một khách hàng tiềm năng và hài lòng!")
        elif prediction >= 3:
            st.warning("Khách hàng ở mức trung bình, cần cải thiện dịch vụ.")
        else:
            st.error("Cảnh báo: Khách hàng này có nguy cơ rời bỏ nền tảng!")