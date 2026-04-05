import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Dự báo Food Delivery - Lê Tấn Toàn", layout="wide")

# --- HÀM LOAD DỮ LIỆU ---
@st.cache_data
def load_and_prep_data():
    # Đọc dữ liệu gốc
    df = pd.read_csv('food_ordering_behavior_dataset.csv')
    
    # Tiền xử lý nhanh cho EDA
    mood_map = {'Celebrating': 4, 'Happy': 3, 'Lazy': 2, 'Stressed': 1}
    df['mood_score'] = df['mood'].map(mood_map)
    
    # Bảng gộp theo User ID để tính toán tương quan
    user_df = df.groupby('user_id').agg({
        'order_value': 'sum',
        'rating_given': 'mean',
        'mood_score': 'mean',
        'order_id': 'count'
    }).reset_index()
    user_df.columns = ['user_id', 'total_spent', 'avg_rating', 'avg_mood', 'order_count']
    return df, user_df

try:
    df, user_df = load_and_prep_data()
except Exception as e:
    st.error("⚠️ Không tìm thấy file dữ liệu CSV. Vui lòng kiểm tra lại thư mục GitHub!")
    st.stop()

# --- SIDEBAR ĐIỀU HƯỚNG ---
st.sidebar.title("📌 Menu Chính")
page = st.sidebar.radio("Chuyển trang:", ["Trang 1: Giới thiệu & EDA", "Trang 2: Triển khai Mô hình"])

# ---------------------------------------------------------
# TRANG 1: GIỚI THIỆU & KHÁM PHÁ DỮ LIỆU (EDA)
# ---------------------------------------------------------
if page == "Trang 1: Giới thiệu & EDA":
    st.title("📊 Khám Phá Dữ Liệu Hành Vi Khách Hàng")
    
    # --- THÔNG TIN BẮT BUỘC ---
    st.info(f"""
    **Tên đề tài:** Dự báo mức độ hài lòng và tỉ lệ giữ chân khách hàng trên nền tảng Food Delivery  
    **Sinh viên thực hiện:** Lê Tấn Toàn  
    **MSSV:** 22T1020768
    """)

    st.markdown("""
    ### 🎯 Giá trị thực tiễn
    Bài toán giúp doanh nghiệp chủ động nhận diện trải nghiệm khách hàng ngay tại thời điểm đặt hàng. 
    Bằng cách dự báo **Rating** và **Tỉ lệ quay lại**, quản lý có thể đưa ra các quyết định Marketing hoặc hỗ trợ kịp thời để tối ưu hóa doanh thu và lòng trung thành của khách.
    """)

    st.divider()

    # --- NỘI DUNG KỸ THUẬT ---
    st.subheader("1. Hiển thị dữ liệu mẫu (Raw Data)")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("2. Biểu đồ phân tích trực quan")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Phân phối điểm Rating (Nhãn)**")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='rating_given', data=df, palette='magma', ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.write("**Ma trận tương quan giữa các đặc trưng quan trọng**")
        fig2, ax2 = plt.subplots()
        corr = user_df[['total_spent', 'avg_rating', 'avg_mood', 'order_count']].corr()
        sns.heatmap(corr, annot=True, cmap='RdBu', fmt=".2f", ax=ax2)
        st.pyplot(fig2)

    st.write("### 📝 Nhận xét về dữ liệu")
    st.write("""
    - **Phân phối:** Rating tập trung chủ yếu ở mức 3 và 4 sao. Dữ liệu có độ lệch nhẹ về phía các đánh giá tích cực.
    - **Đặc trưng:** Tâm trạng (`mood`) và Giá trị đơn hàng (`order_value`) là hai yếu tố có ảnh hưởng rõ rệt nhất đến kết quả dự báo.
    - **Tương quan:** Mối quan hệ giữa chi tiêu tích lũy và số lượng đơn hàng rất chặt chẽ, tạo cơ sở tốt để dự đoán lòng trung thành.
    """)

# ---------------------------------------------------------
# TRANG 2: TRIỂN KHAI MÔ HÌNH (DỰ BÁO TỔNG HỢP)
# ---------------------------------------------------------
else:
    st.title("🤖 Hệ Thống Dự Báo Thông Minh")

    # Load mô hình (.pkl)
    try:
        model_rating = joblib.load('model_xgb.pkl')
        model_repeat = joblib.load('model_repeat.pkl')
    except:
        st.warning("⚠️ Không tìm thấy file mô hình (.pkl). Vui lòng upload lên GitHub!")
        st.stop()

    st.subheader("📝 Nhập thông tin đơn hàng mới")
    
    # Thiết kế giao diện nhập liệu
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            u_id = st.text_input("User ID:", "TOAN_22T")
            order_val = st.number_input("Giá trị đơn hàng (VNĐ):", 50, 5000, 500)
        with c2:
            mood_label = st.selectbox("Tâm trạng khách hàng:", ['Stressed', 'Lazy', 'Happy', 'Celebrating'])
            time_order = st.number_input("Thời gian giao dự kiến (phút):", 5, 120, 30)
        with c3:
            city = st.selectbox("Thành phố giao hàng:", df['city'].unique())
            weather = st.radio("Thời tiết mưa?", ["No", "Yes"], horizontal=True)
            
        submit_btn = st.form_submit_button("🚀 Thực hiện phân tích")

    # Xử lý logic và Hiển thị kết quả
    if submit_btn:
        # Tiền xử lý
        mood_map = {'Stressed': 1, 'Lazy': 2, 'Happy': 3, 'Celebrating': 4}
        current_mood = mood_map[mood_label]

        # 1. Dự đoán Rating (XGBRegressor)
        # Input: [total_spent, avg_mood, order_count]
        input_r = pd.DataFrame([[order_val, current_mood, 1]], 
                              columns=['total_spent', 'avg_mood', 'order_count'])
        pred_rating = model_rating.predict(input_r)[0]
        pred_rating = max(1.0, min(5.0, pred_rating))

        # 2. Dự đoán % Quay lại (XGBClassifier) dựa trên Rating vừa dự báo
        # Input: [total_spent, avg_rating, avg_mood]
        input_rep = pd.DataFrame([[order_val, pred_rating, current_mood]], 
                                columns=['total_spent', 'avg_rating', 'avg_mood'])
        
        proba_raw = model_repeat.predict_proba(input_rep)[0][1]
        
        # Áp dụng công thức Baseline 50% (Dịch chuyển biên độ quanh mốc cân bằng)
        sensitivity = 0.8
        proba_final = 50 + (proba_raw - 0.5) * 100 * sensitivity
        proba_final = max(5.0, min(98.5, proba_final))

        # Hiển thị kết quả rõ ràng
        st.divider()
        st.subheader("📈 Kết quả dự báo từ hệ thống AI")
        
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.metric("⭐ Rating dự báo", f"{pred_rating:.1f} / 5.0")
            p_val_r = float(max(0.0, min(pred_rating/5, 1.0)))
            st.progress(p_val_r)
            if pred_rating >= 4: st.success("Dự báo: Khách hàng sẽ rất hài lòng!")
            elif pred_rating >= 3: st.warning("Dự báo: Trải nghiệm mức trung bình.")
            else: st.error("Dự báo: Rủi ro nhận đánh giá thấp!")

        with res_col2:
            st.metric("🔁 Xác suất quay lại", f"{proba_final:.1f}%")
            st.progress(float(proba_final/100))
            if proba_final > 50:
                st.info(f"Tỉ lệ giữ chân tích cực (+{(proba_final-50):.1f}%)")
            else:
                st.warning(f"Cần thêm ưu đãi để giữ chân (-{(50-proba_final):.1f}%)")