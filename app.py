import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hệ thống Quản lý & Dự báo Food Delivery", layout="wide")

# --- HÀM LOAD DỮ LIỆU & TIỀN XỬ LÝ ---
@st.cache_data
def load_and_prep_data():
    # Đọc dữ liệu gốc
    df = pd.read_csv('food_ordering_behavior_dataset.csv.csv')
    
    # Tạo mood_score để tính toán
    mood_map = {'Celebrating': 4, 'Happy': 3, 'Lazy': 2, 'Stressed': 1}
    df['mood_score'] = df['mood'].map(mood_map)
    
    # Tạo bảng user_df (Gộp theo User ID) để phục vụ Tab tra cứu khách cũ
    user_df = df.groupby('user_id').agg({
        'order_value': 'sum',
        'rating_given': 'mean',
        'mood_score': 'mean',
        'order_id': 'count'
    }).reset_index()
    user_df.columns = ['user_id', 'total_spent', 'avg_rating', 'avg_mood', 'order_count']
    
    return df, user_df

# Thử load dữ liệu
try:
    df, user_df = load_and_prep_data()
except Exception as e:
    st.error(f"Lỗi load dữ liệu: {e}. Đảm bảo file CSV nằm cùng thư mục.")
    st.stop()

# --- SIDEBAR ĐIỀU HƯỚNG ---
st.sidebar.title("📌 Menu Chức Năng")
page = st.sidebar.radio("Chọn nội dung hiển thị:", ["Trang 1: Giới thiệu & EDA", "Trang 2: Dự đoán & Phân tích"])

# ---------------------------------------------------------
# TRANG 1: GIỚI THIỆU & EDA
# ---------------------------------------------------------
if page == "Trang 1: Giới thiệu & EDA":
    st.title("📊 Khám Phá Dữ Liệu Hành Vi Đặt Đồ Ăn")
    
    # Section: Thông tin sinh viên
    st.success(f"""
    **Sinh viên thực hiện:** Lê Tấn Toàn  
    **MSSV:** 22T1020768  
    **Đề tài:** Dự báo sự hài lòng và tỉ lệ quay lại của khách hàng.
    """)

    st.markdown("""
    ### 🎯 Giá trị thực tiễn
    Hệ thống giúp doanh nghiệp nhận diện khách hàng VIP và khách hàng có nguy cơ rời bỏ (Churn). 
    Bằng cách dự báo Rating, quản lý có thể can thiệp kịp thời vào các đơn hàng có trải nghiệm kém.
    """)

    st.divider()

    # Hiển thị dữ liệu thô
    st.subheader("1. Trích xuất dữ liệu mẫu (Raw Data)")
    st.dataframe(df.head(15), use_container_width=True)

    # Trực quan hóa dữ liệu
    st.subheader("2. Phân tích đặc trưng quan trọng")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Phân phối mức độ hài lòng (Rating)**")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='rating_given', data=df, palette='viridis', ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.write("**Mối tương quan giữa các biến số hành vi**")
        fig2, ax2 = plt.subplots()
        # Chỉ lấy cột số để vẽ heatmap
        corr = user_df[['total_spent', 'avg_rating', 'avg_mood', 'order_count']].corr()
        sns.heatmap(corr, annot=True, cmap='RdBu', fmt=".2f", ax=ax2)
        st.pyplot(fig2)

    st.write("### 📝 Nhận xét dữ liệu")
    st.info("""
    - Dữ liệu khách hàng khá đa dạng về độ tuổi và khu vực địa lý.
    - **Tâm trạng (Mood)** có mối tương quan thuận mạnh nhất với Rating.
    - Đa số khách hàng tập trung ở mức 3-4 sao, cho thấy dịch vụ ở mức khá nhưng cần cải thiện để đạt mức 5 sao.
    """)

# ---------------------------------------------------------
# TRANG 2: TRIỂN KHAI MÔ HÌNH
# ---------------------------------------------------------
else:
    st.title("🤖 Hệ Thống Dự Báo Thông Minh")

    # Load 2 mô hình đã huấn luyện
    try:
        model_rating = joblib.load('model_xgb.pkl')
        model_repeat = joblib.load('model_repeat.pkl')
    except:
        st.warning("⚠️ Cảnh báo: Thiếu file 'model_xgb.pkl' hoặc 'model_repeat.pkl' trên server.")
        st.stop()

    # Chia Tab cho 2 bài toán
    tab1, tab2 = st.tabs(["⭐ Dự đoán Rating Đơn Mới", "📈 Dự đoán Tỉ lệ Quay lại (Khách cũ)"])

    # --- TAB 1: DỰ ĐOÁN RATING ---
    with tab1:
        st.subheader("Nhập thông tin đơn hàng hiện tại")
        with st.container():
            c1, c2, c3 = st.columns(3)
            with c1:
                u_id_new = st.text_input("User ID:", "TOAN_22T")
                order_val = st.number_input("Giá trị đơn hàng (VNĐ):", 100, 2000, 500)
            with c2:
                mood_new = st.selectbox("Tâm trạng khách:", ['Stressed', 'Lazy', 'Happy', 'Celebrating'])
                time_new = st.number_input("Thời gian giao (phút):", 5, 120, 30)
            with c3:
                city_new = st.selectbox("Thành phố:", df['city'].unique())
                weather_new = st.radio("Thời tiết mưa?", ["No", "Yes"], horizontal=True)

        if st.button("🚀 Dự đoán Rating"):
            mood_map = {'Stressed': 1, 'Lazy': 2, 'Happy': 3, 'Celebrating': 4}
            # Chuẩn bị input (Giả định order_count = 1 cho đơn mới)
            input_rating = pd.DataFrame([[order_val, mood_map[mood_new], 1]], 
                                       columns=['total_spent', 'avg_mood', 'order_count'])
            
            pred_r = model_rating.predict(input_rating)[0]
            pred_r = max(1, min(5, pred_r))

            st.divider()
            res_c1, res_c2 = st.columns(2)
            res_c1.metric("Rating dự báo", f"{pred_r:.1f} ⭐")
            
            # Sửa lỗi progress bằng ép kiểu float()
            progress_val = float(max(0.0, min(pred_r/5, 1.0)))
            res_c2.write(f"**Mức độ hài lòng:** {round(progress_val*100)}%")
            res_c2.progress(progress_val)
            
            if pred_r >= 4: st.success("Khách hàng này có khả năng cao sẽ hài lòng!")
            elif pred_r < 3: st.error("Cảnh báo: Đơn hàng cần được kiểm tra lại chất lượng!")

    # --- TAB 2: DỰ ĐOÁN % QUAY LẠI ---
    with tab2:
        st.subheader("🔍 Tra cứu & Dự báo khả năng giữ chân")
        search_id = st.number_input("Nhập User ID để tra cứu:", min_value=0, value=0)

        if search_id > 0:
            if search_id in user_df['user_id'].values:
                # Lấy dữ liệu lịch sử từ user_df
                u_data = user_df[user_df['user_id'] == search_id].iloc[0]
                
                st.write(f"#### Kết quả tra cứu khách hàng: `{search_id}`")
                st.write(f"- Tổng chi tiêu tích lũy: **{u_data['total_spent']:.0f}**")
                st.write(f"- Rating trung bình đã đánh giá: **{u_data['avg_rating']:.1f} ⭐**")
                st.write(f"- Tổng số đơn hàng đã đặt: **{u_data['order_count']:.0f}**")

                # Dự đoán % quay lại bằng model_repeat
                # Input: [total_spent, avg_rating, avg_mood]
                input_rep = pd.DataFrame([[u_data['total_spent'], u_data['avg_rating'], u_data['avg_mood']]], 
                                        columns=['total_spent', 'avg_rating', 'avg_mood'])
                
                # predict_proba trả về mảng [[prob_0, prob_1]]
                proba_return = model_repeat.predict_proba(input_rep)[0][1] * 100
                
                st.divider()
                st.write(f"### 📈 Xác suất khách hàng sẽ quay lại: **{proba_return:.1f}%**")
                
                # Sửa lỗi progress bằng ép kiểu float()
                st.progress(float(proba_return/100))
                
                if proba_return > 70:
                    st.balloons()
                    st.success("Đây là khách hàng trung thành (Loyal Customer)!")
                elif proba_return < 40:
                    st.warning("Khách hàng có nguy cơ rời bỏ cao (Churn Risk). Cần gửi mã giảm giá!")
            else:
                st.error("Không tìm thấy User ID này trong hệ thống dữ liệu hiện tại.")