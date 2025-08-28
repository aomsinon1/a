import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="การทำนายโรคหัวใจวายDtree", page_icon="❤️")

# --- Header Section ---
st.title("🩺 การพยากรณ์โรคหัวใจวายด้วย Decision Tree")
st.markdown("---")
st.markdown("แอปพลิเคชันนี้ใช้โมเดล **Decision Tree** ในการวิเคราะห์ข้อมูลและทำนายความเสี่ยงของภาวะหัวใจวาย")

# --- Data Loading Section ---
st.subheader("📊 ข้อมูลที่ใช้ในการฝึกโมเดล")

try:
    df = pd.read_csv('data/Medicaldataset_converted.csv')
    st.info("✅ โหลดข้อมูลจากไฟล์ **'Medicaldataset_converted.csv'** เรียบร้อยแล้ว")
    st.write("ตัวอย่าง 10 แถวแรกของชุดข้อมูล:")
    st.dataframe(df.head(10))

    # --- Data Preparation ---
    target_column = 'Result'
    features = [col for col in df.columns if col != target_column]
    X = df[features]
    y = df[target_column]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=200)

    # --- Model Training ---
    with st.spinner('กำลังฝึกโมเดล Decision Tree...'):
        ModelDtree = DecisionTreeClassifier()
        dtree = ModelDtree.fit(x_train, y_train)
    st.success("✨ ฝึกโมเดลสำเร็จ!")

    # --- User Input Section ---
    st.subheader("✍️ ป้อนข้อมูลเพื่อพยากรณ์")
    st.markdown("---")

    col1, col2 = st.columns(2)
    
    feature_labels = {
        'Age': 'อายุ', 'Gender': 'เพศ', 'Heart rate': 'อัตราการเต้นของหัวใจ', 
        'Systolic blood pressure': 'ความดันโลหิตตัวบน (ซิสโตลิก)', 
        'Diastolic blood pressure': 'ความดันโลหิตตัวล่าง (ไดแอสโตลิก)', 
        'Blood sugar': 'น้ำตาลในเลือด', 'CK-MB': 'ครีเอตีนไคเนส เอ็มบี', 
        'Troponin': 'โทรโปนิน'
    }

    user_input = {}
    with col1:
        user_input['Age'] = st.number_input(f'ป้อนค่าสำหรับ: {feature_labels["Age"]}', min_value=0, max_value=120, value=45)
        user_input['Heart rate'] = st.number_input(f'ป้อนค่าสำหรับ: {feature_labels["Heart rate"]}', min_value=0, value=75)
        user_input['Systolic blood pressure'] = st.number_input(f'ป้อนค่าสำหรับ: {feature_labels["Systolic blood pressure"]}', min_value=0, value=120)
        user_input['Diastolic blood pressure'] = st.number_input(f'ป้อนค่าสำหรับ: {feature_labels["Diastolic blood pressure"]}', min_value=0, value=80)
    
    with col2:
        gender_options = {'ชาย': 0, 'หญิง': 1}
        selected_gender = st.selectbox(f'ป้อนค่าสำหรับ: {feature_labels["Gender"]}', options=list(gender_options.keys()))
        user_input['Gender'] = gender_options[selected_gender]
        user_input['Blood sugar'] = st.number_input(f'ป้อนค่าสำหรับ: {feature_labels["Blood sugar"]}', min_value=0.0, value=90.0)
        user_input['CK-MB'] = st.number_input(f'ป้อนค่าสำหรับ: {feature_labels["CK-MB"]}', min_value=0.0, value=0.0)
        user_input['Troponin'] = st.number_input(f'ป้อนค่าสำหรับ: {feature_labels["Troponin"]}', min_value=0.0, value=0.0)


    if st.button("พยากรณ์ผล", type="primary"):
        x_input = [[user_input[feature] for feature in features]]
        y_predict2 = dtree.predict(x_input)
        
        st.write("---")
        st.subheader("### 💡 ผลการพยากรณ์:")
        if y_predict2[0] == 1:
            st.error("⚠️ **มีความเสี่ยงสูงที่จะเกิดโรคหัวใจวาย**")
        else:
            st.success("🟢 **ความเสี่ยงในการเกิดโรคหัวใจวายอยู่ในระดับต่ำ**")

    # --- Model Performance & Visualization ---
    st.markdown("---")
    st.subheader("📈 ประสิทธิภาพของโมเดล")
    y_predict = dtree.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    st.metric(label="ความแม่นยำของโมเดล (Accuracy Score)", value=f"{score * 100:.2f} %")

    st.subheader("🌳 แผนผัง Decision Tree")
    fig, ax = plt.subplots(figsize=(20, 15))
    tree.plot_tree(dtree, feature_names=features, class_names=['Low Risk', 'High Risk'], ax=ax, filled=True, rounded=True, fontsize=10)
    st.pyplot(fig)

except FileNotFoundError:
    st.error("❌ **ไม่พบไฟล์ 'Medicaldataset_converted.csv'** กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์ `data/` และชื่อไฟล์ถูกต้อง")
except KeyError as e:
    st.error(f"❌ **เกิดข้อผิดพลาด: ไม่พบคอลัมน์ '{e}'** ในไฟล์ CSV ของคุณ กรุณาตรวจสอบว่าชื่อคอลัมน์ในโค้ดตรงกับในไฟล์ข้อมูล")
except Exception as e:
    st.error(f"❌ **เกิดข้อผิดพลาด**: {e}")