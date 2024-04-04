import streamlit as st
import pandas as pd
import joblib
import numpy as np


# Title
st.title("3-month functional outcome prediction of SSSI patients")

st.subheader("Prediction(obtain after inputting parameters)")

#st.text("The implementation of this 3-month functional outcome prediction is NOT intended for use supporting or informing clinical decision-making.It is ONLY to be used for academic research, peer review and validation purposes, and it must NOT be used with data or information relating to any individual.")
st.markdown("The implementation of this 3-month functional outcome prediction is NOT intended for use support ing or informing clinical decision-making.It is ONLY to be used for academic research, peer review and validation purposes,, and it must NOT be used with data or information relating to any individual.")
# Input bar 1
st.header("Input Parameters")
Age = st.number_input("Enter Age(years)",18,)
DB = st.number_input("Enter Diabetes(0=no,1=yes)",0,1)
NIHSS = st.number_input("Enter NIHSS(scores)",0,42)
SHR = st.number_input("Enter SHR")
pSSSI = st.number_input("Enter pSSSI(0=dSSSI,1=pSSSI)",0,1)
# Dropdown input
#eyes = st.selectbox("Select Eye Colour", ("Blue", "Brown"))

# If button is pressed
if st.button("Submit"):
    # Unpickle classifier
    #clf = joblib.load("D:/Material/new/ML compare Yxiu/clfSSSIcatboost.pkl")
    clf = joblib.load('other data/clfSSSIcatboost.pkl')
    import pickle
    with open('other data/dfXstandardization_params.pkl', 'rb') as f:
            loaded_dfXstandardization_params = pickle.load(f)
    # Store inputs into dataframe
    X = pd.DataFrame([[Age, DB, NIHSS,SHR,pSSSI]],
                     columns=["Age", "DB", "NIHSS","SHR","pSSSI"])
    #X = X.replace(["Brown", "Blue"], [1, 0])
    standardized_X_new = X.copy()
    for column in X.columns:
        if column in loaded_dfXstandardization_params:
            mean = loaded_dfXstandardization_params[column]['mean']
            std = loaded_dfXstandardization_params[column]['std']
            # 应用标准化
            standardized_X_new[column] = (X[column] - mean) / std
    # Get prediction
    # 假设模型能够直接返回预测概率
    predicted_probabilities = clf.predict_proba(standardized_X_new)
    predicted_classes = predicted_probabilities.argmax(axis=1)
    st.markdown(f"This predicted probabilities of three month functional outcome")
    predicted_probabilities
    # 使用列表推导式替换所有
    if predicted_classes == 0:
        output = "favourable functional outcome"  # 如果结果等于目标值，则输出指定的值
    else:
        output = "unfavourable functional outcome"  # 如果结果不等于目标值，则输出其他值

    print(output)  # 打印输出值
    st.markdown(f"This prediction of three month outcome is {output}")

