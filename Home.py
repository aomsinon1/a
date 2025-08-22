import streamlit as st

st.page_link("Home.py", label="หน้าแรก", icon="🏠")
st.page_link("pages/DTree.py", label="การทำนายโรคหัวใจวาย", icon="1️⃣")
st.page_link("pages/NaiveBaye.py", label="การทำนายข้อมูลด้วยเทคนิค Naive Baye", icon="2️⃣", disabled=False)
st.page_link("http://www.google.com", label="Google", icon="🌎")