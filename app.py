import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import xgboost as xgb

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hệ thống Dự báo Food Delivery - Lê Tấn Toàn", layout="wide")

# --- HÀM LOAD DỮ LIỆU & TIỀN XỬ LÝ ---
@st.cache_data
def load_and_prep_data():
    # Đọc dữ liệu gốc
    df = pd.read_csv('food_ordering_behavior_dataset.csv')
    
    # Tạo mood_score (1-4)
    mood_map = {'Celebrating': 4, 'Happy': 3, 'Lazy': 2, 'Stressed': 1}
    df['mood_score'] = df['mood'].map(mood_map)
    
    # Tạo bảng user_df (Gộp theo User ID) để tra cứu lịch sử
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
    st.error(f"Lỗi: Không tìm thấy file dữ liệu CSV. Vui lòng kiểm tra lại!")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("Danh Mục")
page = st.sidebar.radio("Chuyển trang:", ["Trang 1: Giới thiệu & EDA", "Trang 2: Dự đoán Mô hình"])

# ---------------------------------------------------------
# TRANG 1: GIỚI THIỆU & EDA
# ---------------------------------------------------------
if page == "Trang 1: Giới thiệu & EDA":
    st.title("Khám Phá Dữ Liệu Food Ordering")
    
    # Thông tin sinh viên
    st.info("""
    **Họ tên SV:** Lê Tấn Toàn  
    **MSSV:** 22T1020768  
    **Đề tài:** Dự báo hành vi và mức độ hài lòng của khách hàng trên nền tảng đặt đồ ăn.
    """)

    st.markdown("""
    ### Giá trị thực tiễn
    Ứng dụng này giúp các đơn vị vận hành nhận diện sớm các đơn hàng có nguy cơ bị đánh giá thấp và dự báo khả năng giữ chân khách hàng (Retention). 
    Dữ liệu giúp tối ưu hóa trải nghiệm dựa trên tâm trạng và thói quen chi tiêu của người dùng.
    """)

    st.divider()

    # Hiển thị dữ liệu thô
    st.subheader("1. Dữ liệu mẫu (Raw Data)")
    st.dataframe(df.head(10), use_container_width=True)

    # Trực quan hóa
    st.subheader("2. Biểu đồ phân tích")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Phân bổ Rating trong hệ thống**")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='rating_given', data=df, palette='viridis', ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.write("**Ma trận tương quan (Correlation)**")
        fig2, ax2 = plt.subplots()
        corr = user_df[['total_spent', 'avg_rating', 'avg_mood', 'order_count']].corr()
        sns.heatmap(corr, annot=True, cmap='RdBu', fmt=".2f", ax=ax2)
        st.pyplot(fig2)

    st.write("### Nhận xét về dữ liệu")
    st.write("""
    - **Tính phân tán:** Rating tập trung chủ yếu ở mức 3 và 4 sao, cho thấy dịch vụ khá ổn định nhưng thiếu đột phá lên 5 sao.
    - **Yếu tố then chốt:** Tâm trạng (`mood`) và Thời gian giao hàng có tác động lớn nhất đến điểm số đánh giá.
    - **Đặc trưng quan trọng:** Khách hàng chi tiêu nhiều (`total_spent`) có xu hướng đặt hàng thường xuyên hơn, nhưng không tỷ lệ thuận hoàn toàn với mức độ hài lòng.
    """)

# ---------------------------------------------------------
# TRANG 2: TRIỂN KHAI MÔ HÌNH
# ---------------------------------------------------------
else:
    st.title("Trình Dự Báo Thông Minh")

    # Load Models
    try:
        model_rating = joblib.load('model_xgb.pkl')
        model_repeat = joblib.load('model_repeat.pkl')
    except:
        st.warning(" Không tìm thấy file .pkl. Hãy đảm bảo bạn đã upload model_xgb.pkl và model_repeat.pkl.")
        st.stop()

    tab1, tab2 = st.tabs(["Dự đoán Rating đơn mới", "Tỉ lệ khách quay lại"])

    # --- TAB 1: DỰ ĐOÁN RATING ---
    with tab1:
        st.subheader("Nhập thông tin đơn hàng để dự báo mức độ hài lòng")
        with st.expander("Điền thông tin chi tiết", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                u_id_input = st.text_input("Mã khách hàng:", "1001")
                order_val = st.number_input("Giá trị đơn:", 50, 5000, 500)
            with c2:
                mood_input = st.selectbox("Tâm trạng:", ['Celebrating', 'Stressed', 'Lazy', 'Happy'])
                time_input = st.number_input("Thời gian giao (phút):", 5, 150, 30)
            with c3:
                res_input = st.selectbox("Loại nhà hàng:", df['restaurant_type'].unique())
                weather_input = st.radio("Thời tiết mưa?", ["No", "Yes"], horizontal=True)

        if st.button("🚀 Dự đoán Rating"):
            mood_map = {'Stressed': 1, 'Lazy': 2, 'Happy': 3, 'Celebrating': 4}
            # Input cho XGBRegressor: [total_spent, avg_mood, order_count]
            # Giả định đơn mới nên order_count = 1
            input_r = pd.DataFrame([[order_val, mood_map[mood_input], 1]], 
                                  columns=['total_spent', 'avg_mood', 'order_count'])
            
            pred_rating = model_rating.predict(input_rating)[0] if 'input_rating' in locals() else model_rating.predict(input_r)[0]
            pred_rating = max(1.0, min(5.0, pred_rating))

            st.divider()
            r1, r2 = st.columns(2)
            r1.metric("Rating dự kiến", f"{pred_rating:.1f} ⭐")
            
            prog_val = float(max(0.0, min(pred_rating/5, 1.0)))
            r2.write(f"**Độ hài lòng:** {int(prog_val*100)}%")
            r2.progress(prog_val)
            
            if pred_rating >= 4: st.success("Khách hàng nhiều khả năng sẽ hài lòng!")
            elif pred_rating >= 3: st.warning("Trải nghiệm ở mức trung bình.")
            else: st.error("Cảnh báo: Đơn hàng có rủi ro nhận đánh giá thấp!")

    # --- TAB 2: TỈ LỆ QUAY LẠI (BASELINE 50%) ---
    with tab2:
        st.subheader("Dự báo khả năng giữ chân khách hàng cũ")
        search_id = st.number_input("Tra cứu User ID trong hệ thống:", min_value=0, value=0)

        if search_id > 0:
            if search_id in user_df['user_id'].values:
                # Lấy dữ liệu lịch sử
                user_record = user_df[user_df['user_id'] == search_id].iloc[0]
                
                st.write(f"**Thông tin khách hàng {search_id}:**")
                st.write(f"- Đã chi tiêu: {user_record['total_spent']:.0f} | - Đơn đã đặt: {user_record['order_count']:.0f} | - Rating trung bình: {user_record['avg_rating']:.1f} ⭐")

                # Dự đoán dùng mô hình Classifier
                input_rep = pd.DataFrame([[user_record['total_spent'], user_record['avg_rating'], user_record['avg_mood']]], 
                                        columns=['total_spent', 'avg_rating', 'avg_mood'])
                
                # Xác suất gốc từ XGBoost
                proba_raw = model_repeat.predict_proba(input_rep)[0][1]
                
                # LOGIC: Baseline 50% + (Xác suất - 0.5) * Sensitivity
                sensitivity = 0.8
                proba_final = 50 + (proba_raw - 0.5) * 100 * sensitivity
                proba_final = max(5.0, min(98.5, proba_final)) # Chặn biên cho đẹp

                st.divider()
                st.write(f"### Tỉ lệ khách hàng quay lại: **{proba_final:.1f}%**")
                st.progress(float(proba_final/100))
                
                if proba_final > 50:
                    st.success(f"Khách hàng có triển vọng trung thành cao hơn trung bình (+{(proba_final-50):.1f}%)")
                else:
                    st.warning(f"Cần thêm chương trình khuyến mãi để giữ chân khách này (-{(50-proba_final):.1f}%)")
            else:
                st.error("ID này không tồn tại trong lịch sử dữ liệu.")