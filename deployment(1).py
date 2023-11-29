import pandas as pd
import streamlit as st
from pickle import load
import numpy as np

st.set_page_config(
    page_title="Bankruptcy Prevention App",
    page_icon="📈",
    layout="wide",
    menu_items={
        'Get Help': 'http://www.quickmeme.com/img/54/547621773e22705fcfa0e73bc86c76a05d4c0b33040fcb048375dfe9167d8ffc.jpg',
        'Report a bug': "https://w7.pngwing.com/pngs/839/902/png-transparent-ladybird-ladybird-bug-miscellaneous-presentation-insects-thumbnail.png",
        'About': "This is a Bankruptcy Prevention App. Very Easy to use!"
    }
)

# Title and customization
st.title('Company Bankruptcy Prediction')
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Input Features
st.sidebar.header('User Input Parameters')

ind_risk = st.sidebar.selectbox('Industrial Risk', ['0', '0.5', '1'], key='ind_risk')
man_risk = st.sidebar.selectbox('Management Risk', ['0', '0.5', '1'], key='man_risk')
fin_flex = st.sidebar.selectbox('Financial Flexibility', ['0', '0.5', '1'], key='fin_flex')
cred = st.sidebar.selectbox('Credibility', ['0', '0.5', '1'], key='cred')
comp = st.sidebar.selectbox('Competitiveness', ['0', '0.5', '1'], key='comp')
op_risk = st.sidebar.selectbox('Operating Risk', ['0', '0.5', '1'], key='op_risk')

# Feature explanation
# Feature Explanation
st.markdown("### Feature Explanation")
st.write("Industrial Risk: Industry sector risk (0=low, 0.5=medium, 1=high).")
st.write("Management Risk: Management-related risk (0=low, 0.5=medium, 1=high).")
st.write("Financial Flexibility: Financial stability (0=low, 0.5=medium, 1=high).")
st.write("Credibility: Trustworthiness (0=low, 0.5=medium, 1=high).")
st.write("Competitiveness: Competitive strength (0=low, 0.5=medium, 1=high).")
st.write("Operating Risk: Operational risk (0=low, 0.5=medium, 1=high).")


# Create the data dictionary
data = {
    'industrial_risk': ind_risk,
    'management_risk': man_risk,
    'financial_flex': fin_flex,
    'credibility': cred,
    'competitiveness': comp,
    'operating_risk': op_risk
}

# Create a DataFrame from the data
df = pd.DataFrame(data, index=[0])

# Display the User Input Parameters
st.subheader('User Input Parameters')
st.write(df)

# Load the Logistic Regression model
loaded_model = load(open('bankruptcy_prevention.sav', 'rb'))

# Make Prediction 
if st.button('Make Prediction'):
    with st.spinner('Predicting...'):
        # Get the prediction result
        result = loaded_model.predict(df)[0]

        if result == 0:
            st.error('The company **will go bankrupt**')
        elif result == 1:
            st.success('The company **will not go bankrupt**')

# Probability 
if st.checkbox('Show Probability'):
    st.subheader('Model Probability:')
    
    # Get the probability of the prediction
    probability = loaded_model.predict_proba(df)

    # Create a DataFrame to display the probability
    prob_df = pd.DataFrame({'Probability for Bankrupt (0)': probability[0][0], 'Probability for Not Bankrupt (1)': probability[0][1]}, index=['Probability'])
    
    st.table(prob_df)

