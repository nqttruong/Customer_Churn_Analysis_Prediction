import streamlit as st
import joblib
import pandas as pd
import os

# Load mô hình đã train
model_path = r"D:\ml\Customer_Churn\saved_models\logreg_best_model.pkl"
if not os.path.exists(model_path):
    st.error("❌ Không tìm thấy file model. Vui lòng train và lưu model trước.")
    st.stop()

model = joblib.load(model_path)

st.title("🔮 Dự đoán khả năng khách hàng rời bỏ (Churn Prediction)")

# --- Form nhập liệu ---
st.subheader("📋 Thông tin khách hàng")

gender = st.selectbox("Giới tính", ["Male", "Female"])
senior_citizen = st.selectbox("Khách hàng cao tuổi", [0, 1])
partner = st.selectbox("Có vợ/chồng/đối tác", ["Yes", "No"])
dependents = st.selectbox("Có người phụ thuộc", ["Yes", "No"])
tenure = st.slider("Số tháng gắn bó (tenure)", 0, 72, 12)
phone_service = st.selectbox("Dịch vụ điện thoại", ["Yes", "No"])
multiple_lines = st.selectbox("Nhiều đường dây", ["Yes", "No"])
internet_service = st.selectbox("Dịch vụ Internet", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Bảo mật trực tuyến", ["Yes", "No"])
online_backup = st.selectbox("Sao lưu trực tuyến", ["Yes", "No"])
device_protection = st.selectbox("Bảo vệ thiết bị", ["Yes", "No"])
tech_support = st.selectbox("Hỗ trợ kỹ thuật", ["Yes", "No"])
streaming_tv = st.selectbox("Xem TV trực tuyến", ["Yes", "No"])
streaming_movies = st.selectbox("Xem phim trực tuyến", ["Yes", "No"])
contract = st.selectbox("Hợp đồng", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Hóa đơn điện tử", ["Yes", "No"])
payment_method = st.selectbox("Phương thức thanh toán", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Chi phí hàng tháng", min_value=0.0, value=50.0)
total_charges = st.number_input("Tổng chi phí", min_value=0.0, value=600.0)

# --- Gom dữ liệu vào DataFrame ---
input_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}])

# --- Đảm bảo đủ cột như lúc training ---
expected_cols = model.named_steps["preprocessor"].feature_names_in_
for col in expected_cols:
    if col not in input_data.columns:
        input_data[col] = 0  # Giá trị mặc định nếu thiếu

# --- Nút dự đoán ---
if st.button("Dự đoán"):
    prob = model.predict_proba(input_data)[0][1]
    churn_label = "❌ Có khả năng rời bỏ" if prob > 0.5 else "✅ Ở lại"
    
    st.write(f"🔎 Xác suất churn: **{prob*100:.2f}%**")
    st.write(f"📌 Kết quả: {churn_label}")
    
    # Gợi ý hành động
    if prob > 0.5:
        st.warning("💡 Gợi ý: Cung cấp khuyến mãi hoặc ưu đãi dịch vụ để giữ chân khách hàng.")
    else:
        st.success("👍 Khách hàng có khả năng ở lại, duy trì chăm sóc tốt.")
