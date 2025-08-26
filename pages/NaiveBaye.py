from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import streamlit as st
import numpy as np

# โหลดข้อมูลจากไฟล์ Medicaldataset_converted.csv
try:
    df = pd.read_csv("./data/Medicaldataset_converted.csv")
except FileNotFoundError:
    st.error("ไม่พบไฟล์ 'Medicaldataset_converted.csv' กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์ data/ และชื่อไฟล์ถูกต้อง")
    st.stop()

# กำหนดคอลัมน์เป้าหมาย
target_column = 'Result'
# ดึงรายชื่อคอลัมน์ที่เป็น feature ทั้งหมดโดยอัตโนมัติ
features = [col for col in df.columns if col != target_column]

# กำหนด X (features) และ y (target)
X = df[features]
y = df[target_column]

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและฝึกโมเดล Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)

st.subheader("กรุณาป้อนข้อมูลเพื่อพยากรณ์")

# สร้าง dictionary สำหรับแปลงชื่อคอลัมน์เป็นภาษาไทย
feature_labels = {
    'Age': 'อายุ',
    'Gender': 'เพศ',
    'Heart rate': 'อัตราการเต้นของหัวใจ',
    'Systolic blood pressure': 'ความดันโลหิตตัวบน (ซิสโตลิก)',
    'Diastolic blood pressure': 'ความดันโลหิตตัวล่าง (ไดแอสโตลิก)',
    'Blood sugar': 'น้ำตาลในเลือด',
    'CK-MB': 'ครีเอตีนไคเนส เอ็มบี (เอนไซม์บ่งบอกกล้ามเนื้อหัวใจ)',
    'Troponin': 'โทรโปนิน (โปรตีนบ่งบอกความเสียหายของกล้ามเนื้อหัวใจ)'
}

# สร้างช่องรับข้อมูล (input fields) สำหรับแต่ละ feature โดยอัตโนมัติ
user_input = {}
for feature in features:
    label_text = feature_labels.get(feature, feature)
    if feature == 'Gender':
        gender_options = {'ชาย': 0, 'หญิง': 1}
        selected_gender = st.selectbox(
            f'กรุณาเลือกข้อมูล: {label_text}',
            options=list(gender_options.keys())
        )
        user_input[feature] = gender_options[selected_gender]
    else:
        user_input[feature] = st.number_input(f'กรุณาป้อนค่าสำหรับ: {label_text}', value=0.0)

if st.button("พยากรณ์"):
    # สร้างรายการค่าที่ผู้ใช้ป้อน เพื่อส่งให้โมเดลทำนาย
    x_input = np.array([[user_input[feature] for feature in features]])
    y_predict2 = clf.predict(x_input)
    st.write("### ผลการพยากรณ์:")
    st.write(y_predict2)
    st.button("ไม่พยากรณ์")
else:
    st.button("ไม่พยากรณ์")
