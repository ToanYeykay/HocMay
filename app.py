import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Food Ordering Behavior Dashboard", layout="wide")

st.title("Phân Tích Hành Vi Đặt Đồ Ăn 🍔")

# Load dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("food_ordering_behavior_dataset.csv")
    return df

df = load_data()

# Sidebar lọc dữ liệu
city = st.sidebar.multiselect("Chọn thành phố:", options=df["city"].unique(), default=df["city"].unique())
df_selection = df[df["city"].isin(city)]

import streamlit as st
import pandas as pd

def main():
    st.title("🍽️ Hệ Thống Ghi Nhận Đơn Hàng")
    st.subheader("Nhập thông tin khách hàng và đơn hàng")

    with st.form("order_form"):
        # Chia cột để giao diện cân đối hơn
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Tên khách hàng", placeholder="Ví dụ: Nguyễn Văn A")
            age = st.number_input("Tuổi", min_value=1, max_value=100, value=25)
            city = st.text_input("Thành phố", placeholder="Ví dụ: Hà Nội, TP.HCM")
            cuisine = st.selectbox("Cuisine (Loại ẩm thực)", 
                                 ["Vietnamese", "Italian", "Japanese", "Korean", "Western", "Other"])
            meal_type = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner", "Snack"])

        with col2:
            order_value = st.number_input("Order Value (VNĐ)", min_value=0, step=1000)
            discount_applied = st.selectbox("Discount Applied", options=[1, 0], 
                                           format_func=lambda x: "Có (1)" if x == 1 else "Không (0)")
            
            # Các trường có giá trị mặc định theo yêu cầu
            mood = st.selectbox("Mood", ["Happy", "Neutral", "Sad", "Stressed"], index=0) # Mặc định Happy
            hunger_level = st.select_slider("Hunger Level", 
                                          options=["Low", "Medium", "High"], value="Medium") # Mặc định Medium
            company = st.selectbox("Company", ["Alone", "Friends", "Family", "Partner"], index=0) # Mặc định Alone

        # Nút xác nhận
        submitted = st.form_submit_button("Lưu thông tin đơn hàng")

        if submitted:
            if not name or not city:
                st.error("Vui lòng nhập đầy đủ Tên và Thành phố!")
            else:
                # Tạo một dictionary để lưu trữ hoặc xử lý tiếp
                new_data = {
                    "City": city,
                    "Name": name,
                    "Age": age,
                    "Cuisine": cuisine,
                    "Meal_Type": meal_type,
                    "Order_Value": order_value,
                    "Discount_Applied": discount_applied,
                    "Mood": mood,
                    "Hunger_Level": hunger_level,
                    "Company": company
                }
                
                st.success("Đã lưu thông tin thành công!")
                st.json(new_data) # Hiển thị dữ liệu vừa nhập để kiểm tra

if __name__ == "__main__":
    main()