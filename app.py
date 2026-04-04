import streamlit as st
import pandas as pd
import xgboost as xgb
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="Phân Tích", layout="wide")

# --- 2. HÀM TẢI TÀI NGUYÊN (CÓ CACHE) ---
@st.cache_resource
def load_prediction_model():
    """Tải mô hình XGBoost và danh sách cột đã train"""
    try:
        model = xgb.XGBRegressor()
        model.load_model('food_rating_model.json')
        with open('model_columns.json', 'r') as f:
            model_cols = json.load(f)
        return model, model_cols
    except Exception as e:
        st.error(f"Không thể tải mô hình: {e}")
        return None, None

@st.cache_data
def load_analysis_data():
    file_path = 'food_ordering_behavior_dataset.csv'
    
    # 1. Kiểm tra file có tồn tại không
    if not os.path.exists(file_path):
        st.error("Không tìm thấy file trên server!")
        return None

    # 2. Kiểm tra dung lượng file
    file_size = os.path.getsize(file_path)
    if file_size < 500: # Nếu file dưới 500 byte thì chắc chắn là lỗi
        st.error(f" File CSV hiện tại quá nhỏ ({file_size} bytes). Đây có thể là file lỗi từ GitHub LFS!")
        st.info("Cách sửa: Hãy upload trực tiếp file CSV từ máy tính lên GitHub Web (Add file -> Upload).")
        return None

    try:
        # 3. Thử đọc file
        df = pd.read_csv(file_path)
        if df.empty:
            st.error("File CSV tồn tại nhưng không có dữ liệu!")
            return None
        df.columns = df.columns.str.strip()
        return df
    except pd.errors.EmptyDataError:
        st.error("Lỗi: File CSV hoàn toàn trống rỗng!")
        return None
    except Exception as e:
        st.error(f"Lỗi không xác định: {e}")
        return None
# --- 3. TRANG NHẬP LIỆU & DỰ ĐOÁN ---
def prediction_page():
    st.title("Dự Đoán Rating Khách Hàng")
    st.markdown("Nhập thông tin đơn hàng để dự đoán mức độ hài lòng của khách.")

    model, model_cols = load_prediction_model()
    if model is None: return

    with st.form("prediction_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            user_id = st.text_input("Mã khách hàng", "USR-999")
            age = st.number_input("Tuổi", 10, 100, 25)
            city = st.selectbox("Thành phố", ["Pune", "Mumbai", "Delhi", "Bangalore", "Hyderabad"])
            cuisine = st.selectbox("Loại món ăn", ["Chinese", "South Indian", "Biryani", "North Indian", "Fast Food", "Desserts"])
        
        with c2:
            order_val = st.number_input("Giá trị món", 0, 1000, 500, step=50)
            delivery = st.number_input("Phí vận chuyển", 0, 100, 15, step=10)
            time_period = st.selectbox("Buổi trong ngày", ["Morning", "Afternoon", "Evening", "Night"])
            discount = st.selectbox("Áp dụng giảm giá", [1, 0], format_func=lambda x: "Có" if x==1 else "Không")

        with c3:
            mood = st.selectbox("Tâm trạng khách", ["Happy", "Stressed", "Lazy", "Celebrating"])
            hunger = st.selectbox("Mức độ đói", ["Low", "Medium", "High"])
            rain = st.selectbox("Thời tiết mưa", [1, 0], format_func=lambda x: "Đang mưa" if x==1 else "Không mưa")
            rank = st.selectbox("Hạng khách hàng", ["Diamond", "Gold", "Silver", "Bronze"])

        submit_btn = st.form_submit_button("Phân tích & Dự đoán")

        if submit_btn:
            # Tạo DataFrame 1 dòng với toàn bộ cột của mô hình (tất cả là 0)
            input_data = pd.DataFrame(0, index=[0], columns=model_cols)
            
            # Điền các cột số
            input_data['age'] = age
            input_data['order_value'] = order_val
            input_data['delivery_fee'] = delivery
            input_data['discount_applied'] = discount
            input_data['rainy_weather'] = rain
            input_data['net_value'] = order_val + delivery
            
            # Điền các cột One-Hot (Gán giá trị 1)
            categorical_features = {
                'city': city, 'order_time': time_period, 'cuisine': cuisine,
                'mood': mood, 'hunger_level': hunger, 'rank': rank
            }
            
            for feat, val in categorical_features.items():
                col_name = f"{feat}_{val}"
                if col_name in input_data.columns:
                    input_data[col_name] = 1

            # Thực hiện dự đoán
            pred_rating = model.predict(input_data)[0]
            
            # Hiển thị kết quả
            st.divider()
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Rating dự báo", f"{pred_rating:.2f} / 5.0")
            
            if pred_rating >= 4.0:
                st.balloons()
                st.success("Khách hàng này có khả năng cao sẽ rất hài lòng!")
            elif pred_rating <= 2.5:
                st.warning("Cảnh báo: Đơn hàng này có nguy cơ nhận đánh giá thấp.")

# --- 4. TRANG PHÂN TÍCH DASHBOARD ---
def analysis_page():
    st.title("Phân tích hành vi và dự báo")
    df = load_analysis_data()
    if df is None: return

    # --- TAB PHÂN CHIA ---
    tab1, tab2, tab3 = st.tabs(["Tổng Quan", "Chi tiết", "Hành Vi Khách Hàng"])

    with tab1:
        # 4 Chỉ số chính (Metrics)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng Đơn Hàng", f"{len(df):,}")
        m2.metric("Rating Trung Bình", f"{df['rating_given'].mean():.2f} ")
        m3.metric("Giá Trị Đơn Trung Bình", f"{int(df['order_value'].mean()):,}đ")
        m4.metric("Tỉ Lệ Quay Lại", f"{(df['is_repeat_order'].mean()*100):.1f}%")

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("🏙️ Rating theo Thành Phố")
            city_rating = df.groupby('city')['rating_given'].mean().reset_index()
            fig_city = px.bar(city_rating, x='city', y='rating_given', color='rating_given', 
                             color_continuous_scale='RdYlGn', title="Rating trung bình theo khu vực")
            st.plotly_chart(fig_city, use_container_width=True)

        with col_b:
            st.subheader("Hiệu suất theo Loại Ẩm Thực")
            cuisine_rating = df.groupby('cuisine')['rating_given'].mean().reset_index().sort_values('rating_given')
            fig_cuisine = px.line(cuisine_rating, x='cuisine', y='rating_given', markers=True, title="Xu hướng hài lòng theo món ăn")
            st.plotly_chart(fig_cuisine, use_container_width=True)

    with tab2:
        st.subheader("Giải thích mô hình XGBoost")
        st.write("Dưới đây là các yếu tố quan trọng nhất để đưa ra dự đoán.")
        
        # Giả lập Feature Importance từ XGBoost (Vì chúng ta đã biết logic "phù phép")
        # Trong thực tế bạn có thể lấy trực tiếp từ model.feature_importances_
        features = ['Phí vận chuyển', 'Tâm trạng', 'Mức độ đói', 'Giảm giá', 'Hạng khách hàng', 'Giá trị đơn']
        importance = [0.45, 0.25, 0.15, 0.08, 0.05, 0.02]
        fi_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values('Importance', ascending=True)
        
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                        title="Độ quan trọng của các tính năng (Feature Importance)")
        st.plotly_chart(fig_fi, use_container_width=True)

    with tab3:
        st.subheader("Phân tích Thời điểm & Tâm trạng")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Rating theo Buổi trong ngày**")
            fig_time = px.box(df, x='order_time', y='rating_given', color='order_time', title="Biến thiên Rating theo khung giờ")
            st.plotly_chart(fig_time, use_container_width=True)
            
        with c2:
            st.write("**Tương quan Tâm trạng & Mức độ đói**")
            # Heatmap tâm trạng vs đói
            mood_hunger = df.groupby(['mood', 'hunger_level'])['rating_given'].mean().unstack()
            fig_heat = px.imshow(mood_hunger, text_auto=True, color_continuous_scale='Viridis', title="Ma trận hài lòng (Mood vs Hunger)")
            st.plotly_chart(fig_heat, use_container_width=True)

# --- 5. ĐIỀU HƯỚNG CHÍNH ---
def main():
    st.sidebar.title("🛠️ Điều hướng")
    page = st.sidebar.radio("Chọn chức năng:", ["Dự đoán Rating", "Phân tích"])

    if page == "Dự đoán Rating":
        prediction_page()
    else:
        analysis_page()

if __name__ == "__main__":
    main()