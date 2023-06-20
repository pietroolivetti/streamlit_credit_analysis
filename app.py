import streamlit as st

import numpy as np
import pandas as pd

import joblib
# CSS = """
# .css-ffhzg2 {
#     background = rgb(38, 62, 111);
# }
# """

# if st.checkbox('Inject CSS'):
#     st.write(f'<style>{CSS}</style>', unsafe_allow_html = True)

linear_model = joblib.load('models/linear_model.joblib')
class_preproc = joblib.load('models/logit_preproc.joblib')
class_model = joblib.load('models/logit_xgboost_model.joblib')

def predict(loan_amnt, annual_inc, tot_cur_bal, term, grade, emp_length, home_ownership, purpose):

    final_df = pd.DataFrame({'loan_amnt': float(loan_amnt), 'annual_inc': float(annual_inc), 'tot_cur_bal': float(tot_cur_bal), 'term': term, 'grade': grade, 'emp_length': emp_length, 'home_ownership': home_ownership, 'purpose': purpose}, index=[0])

    int_rate = float(linear_model.predict(final_df)[0])
    final_df['int_rate'] = int_rate

    final_df = class_preproc.transform(final_df)

    loan_result = int(class_model.predict(final_df[:1])[0])

    loan_prob = float(class_model.predict_proba(final_df[:1])[0][0])

    return {'int_rate': int_rate
            , 'loan_result': loan_result
            , 'loan_prob': loan_prob}

st.markdown(
"""# Credit Approval Simulation
## Check your loan conditions based on your personal information.

The goal of this project is to provide information to lenders and users that are searching for a loan.

We provide a prediction on whether your loan application would be successful and an estimation of your probable interest rate.
"""
)

# Tem como limitar valores mínimos e máximos?
# loan_amnt
amount = st.number_input('How much do you wish to borrow?')
st.write("The amount you want is ", amount)

# annual_inc
income = st.number_input('What is your approximate annual income?')
st.write("Your annual income is ", income)

# tot_cur_bal
balance = st.number_input('Provide an estimation of how much money you have as assets and at your disposal.')
st.write("Your approximate balance and assets are ", balance)


# Categorical variables
df = pd.read_csv('treated_df.csv')

# term
possible_terms = np.sort(df['term'].unique())
term = st.selectbox('Select the term of your loan:', possible_terms)
st.write('The term select is ', term)

# grade
possible_grades = np.sort(df['grade'].unique())
grade = st.selectbox('Select your credit grade: ', possible_grades)
st.write('Your credit grade is ', grade)

# emp_length
possible_emp = np.sort(df['emp_length'].unique())
emp_length = st.selectbox('What is your employment status: ', possible_emp)
st.write('Your employment status is ', emp_length)

# home_onwership
possible_home = np.sort(df['home_ownership'].unique())
home_ownership = st.selectbox('What is your home ownership status: ', possible_home)
st.write("You've selected ", home_ownership)

# purpose
possible_purpose = np.sort(df['purpose'].unique())
purpose = st.selectbox('What will this loan be destined for: ', possible_purpose)
st.write("You've selected ", purpose)

# Getting features from user
dict_to_api = {
    'loan_amnt': amount,
    'annual_inc': income,
    'tot_cur_bal': balance,
    'term': term,
    'grade': grade,
    'emp_length': emp_length,
    'home_ownership': home_ownership,
    'purpose': purpose
}

# Sending to API
st.write(dict_to_api)
st.write("Confirm your information then press the button below to get your results.")

if st.button('Make My Prediction'):
    print("Processando seus dados")

    st.write('Here are your results:')
    # Call da api
    res = predict(**dict_to_api)
    if res['loan_result'] == 0:
        st.markdown(f"""
                 # You have a high chance of having your loan approved. The probability of approval is {round(res['loan_prob'], 2)*100}%.
                 ## Your interest rate would be close to {round(res['int_rate'], 2)}
                 """)
    else: st.markdown(f"""
                 # You have a low chance of having your loan approved. The probability of approval is {round(res['loan_prob'], 2)*100}%.
                 ## Your interest rate would be close to {round(res['int_rate'], 2)}%
                 """)

else:
    st.write('Waiting further information.')
