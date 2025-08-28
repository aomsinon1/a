from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import streamlit as st
import numpy as np

# --- Set Page Config ---
st.set_page_config(
    page_title="การทำนายโรคหัวใจด้วย Naive Bayes",
    page_icon="❤️",
    layout="wide"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
        color: #333;
    }
    .st-emotion-cache-1c7y2n2 {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .st-emotion-cache-16ajdlt {
        background-color: #e6f7ff;
        border-left: 5px solid #1890ff;
        color: #1890ff;
    }
    h1 {
        color: #e83e8c;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    h2, h3, h4 {
        color: #004085;
    }
    .stButton>button {
        background-color: #1890ff;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .stButton>button:hover {
        background-color: #40a9ff;
    }
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.title("❤️ การทำนายโรคหัวใจวายด้วย Naive Bayes")
st.markdown("""
แอปพลิเคชันนี้ใช้โมเดล **Naive Bayes** เพื่อทำนายความเสี่ยงโรคหัวใจวายจากข้อมูลสุขภาพของคุณ
""")
st.markdown("---")

# --- Load Data Section ---
try:
    df = pd.read_csv("./data/Medicaldataset_converted.csv")
except FileNotFoundError:
    st.error("❌ **ไม่พบไฟล์ 'Medicaldataset_converted.csv'** กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์ data/ และชื่อไฟล์ถูกต้อง")
    st.stop()

# --- Model Training ---
target_column = 'Result'
features = [col for col in df.columns if col != target_column]
X = df[features]
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = GaussianNB()
clf.fit(X_train, y_train)

# --- Prediction Section ---
st.subheader("🔮 กรุณาป้อนข้อมูลเพื่อพยากรณ์")
st.info("กรุณาป้อนข้อมูลของคุณในช่องด้านล่างเพื่อรับการทำนาย")

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
cols = st.columns(2)
for i, feature in enumerate(features):
    with cols[i % 2]:
        label_text = feature_labels.get(feature, feature)
        if feature == 'Gender':
            gender_options = {'ชาย': 0, 'หญิง': 1}
            selected_gender = st.selectbox(
                f'กรุณาเลือกข้อมูล: {label_text}',
                options=list(gender_options.keys())
            )
            user_input[feature] = gender_options[selected_gender]
        else:
            # แก้ไขส่วนนี้เพื่อเอาทศนิยมออก
            user_input[feature] = st.number_input(
                f'กรุณาป้อนค่าสำหรับ: **{label_text}**', 
                value=0, 
                step=1,
                key=f"input_{feature}"
            )

# Prediction button and result display
st.markdown("---")
if st.button("🌟 พยากรณ์", type="primary"):
    # สร้างรายการค่าที่ผู้ใช้ป้อน เพื่อส่งให้โมเดลทำนาย
    x_input = np.array([[user_input[feature] for feature in features]])
    y_predict = clf.predict(x_input)
    st.subheader("✅ ผลการพยากรณ์:")
    
    if y_predict[0] == 1:
        st.error('⚠️ **คุณมีความเสี่ยงที่จะเป็นโรคหัวใจวาย**')
        st.markdown("ขอแนะนำให้ปรึกษาแพทย์ผู้เชี่ยวชาญเพื่อยืนยันผลและรับคำแนะนำที่ถูกต้อง")
    else:
        st.success('🟢 **คุณไม่มีความเสี่ยงที่จะเป็นโรคหัวใจวาย**')
        st.markdown("อย่างไรก็ตาม การดูแลสุขภาพอย่างสม่ำเสมอเป็นสิ่งสำคัญ")
    st.button("ไม่พยากรณ์")
else:
    st.button("ไม่พยากรณ์")
