import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.sidebar.title('Check Your Data:')

linear_model = joblib.load('models/linear_model.joblib')
class_preproc = joblib.load('models/logit_preproc.joblib')
class_model = joblib.load('models/logit_xgboost_model.joblib')

# Categorical variables
df = pd.read_csv('treated_df.csv')


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
    
tab1, tab2 = st.tabs(['Informations', 'Result'])

with tab1:
    st.markdown(
    """# Credit Approval Simulation
    ## Check your loan conditions based on your personal information.

    The goal of this project is to provide information to lenders and users that are searching for a loan.

    We provide a prediction on whether your loan application would be successful and an estimation of your probable interest rate.
    """
    )

    # Tem como limitar valores mínimos e máximos?
    # loan_amnt
    st.markdown('### How much do you wish to borrow?')
    st.write('The amount of money you would like to borrow is weighed against your information, such as salary, credit history and credit score.')
    #amount = st.number_input('Enter the amount you wish to borrow: ', step=5000)
    amount = float(st.text_input('Enter the amount you wish to borrow: ', placeholder='0', value='0'))
    if amount != '':
        with st.sidebar:
            st.subheader(f':white_check_mark: Loan amount: {amount}')

    st.write("The amount you want is ", round(amount, 2))

    # annual_inc
    st.markdown('### What is your approximate annual income?')
    st.write('If your loan request is too high compared to your salary, your chances of being approved are lowered.')
    #income = st.number_input('Enter your annual income.')
    income = float(st.text_input('Enter your annual income.', placeholder='0', value='0'))
    st.write("Your annual income is ", int(income))
    if income != '':
        with st.sidebar:
            st.subheader(f':white_check_mark: Income: {income}')
            
    # tot_cur_bal
    st.markdown('### What is your approximate current balance?')
    st.write('Enter the value of any assets you have that can count as collateral, as well as your bank balance.')
    #balance = st.number_input("Enter an estimate of your assets: ")
    balance = float(st.text_input("Enter an estimate of your assets: ", placeholder='0', value='0'))
    st.write("Your approximate balance and assets are ", balance)
    if balance != '':
        with st.sidebar:
            st.subheader(f':white_check_mark: Balance: {balance}')




    # term
    st.markdown("### What is the desired timeframe for your payment?")
    possible_terms = np.sort(df['term'].unique())
    term = st.selectbox('Select the term of your loan:', possible_terms)
    st.write('The selected term is ', term)
    if term != '':
        with st.sidebar:
            st.subheader(f':white_check_mark: Term: {term}')

    # grade
    grade_url = 'https://www.myfico.com/fico-credit-score-estimator/estimator'
    possible_grades = np.sort(df['grade'].unique())


    st.markdown('### Tell us your credit score: ')
    st.write("If you don't know your FICO score, you can estimate it [here](%s). The credit grades are defined according to these intervals:" % grade_url)

    st.write('A - 720 or above')
    st.write('B - between 680 and 719')
    st.write('C - between 630 and 679')
    st.write('D - between 550 and 629')
    st.write('E - between 470 and 549')
    st.write('F - between 400 and 469')
    st.write('G - below 400')
    grade = st.selectbox('Select your credit grade: ', possible_grades)
    st.write('Your credit grade is ', grade)
    if grade != '':
        with st.sidebar:
            st.subheader(f':white_check_mark: Grade: {grade}')

    # emp_length
    possible_emp = np.sort(df['emp_length'].unique())
    st.markdown("### How long have you been employed for?")
    st.write("The longer and more stable your job is, the better your chances.")
    emp_length = st.selectbox('Enter your employment status: ', possible_emp)
    st.write('Your employment status is ', emp_length)
    if emp_length != '':
        with st.sidebar:
            st.subheader(f':white_check_mark: Grade: {emp_length}')

    # home_onwership

    possible_home = np.sort(df['home_ownership'].unique()) #[0], df['home_ownership'].unique()[1], df['home_ownership'].unique()[2], df['home_ownership'].unique()[3] 

    st.markdown("### What is your home ownership status?")
    st.write("A house is an asset that can be used as collateral, and is usually given great importance in lending contracts.")
    home_ownership = st.selectbox('Select your home ownership status: ', possible_home, key=str)
    st.write("You've selected ", home_ownership)
    if home_ownership != '':
        with st.sidebar:
            st.subheader(f':white_check_mark: Home Ownership: {home_ownership}')

    # purpose
    possible_purpose = np.sort(df['purpose'].unique())
    st.markdown("### Finally, tell us what is the purpose of your loan.")
    st.write("The risks linked to certain activities are higher than others. Lenders usually consider that too!")
    purpose = st.selectbox('Select the purpose that best fits your needs: ', possible_purpose)
    st.write("You've selected ", purpose)
    if home_ownership != '':
        with st.sidebar:
            st.subheader(f':white_check_mark: Purpose: {purpose}')

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
    st.markdown("## Confirm the information you've supplied:")
    st.write(dict_to_api)
    st.write("Confirm your information then press the button below to get your results.")

if st.button('Make My Prediction'):
    with tab2:
        print("Processando seus dados")

        st.write('Here are your results:')

        res = predict(**dict_to_api)
        if res['loan_prob'] >= 0.75:
            st.markdown(f"""
                    ## You have a high chance of having your loan approved. The probability of approval is {int(round(res['loan_prob'],2)*100)}%.
                    ### Your interest rate would be close to {int(round(res['int_rate']))}%.
                    """)
        elif res['loan_prob'] < 0.75 and res['loan_prob'] >= 0.50:
            st.markdown(f"""
                    ## Your loan will probably be approved. The probability of approval is {int(round(res['loan_prob'],2)*100)}%.
                    ### Your interest rate would be close to {int(round(res['int_rate']))}%.
                    """)
        elif res['loan_prob'] < 0.50 and res['loan_prob'] >= 0.25:
            st.markdown(f"""
                    ## You have a low chance of getting your loan approved. The probability of approval is {int(round(res['loan_prob'],2)*100)}%.
                    ### Your interest rate would be close to {int(round(res['int_rate']))}%.
                    """)
        else: st.markdown(f"""
                    ## It is not likely that your loan application will be approved. The probability of approval is {int(round(res['loan_prob'],2)*100)}%.
                    ### Your interest rate would be close to {int(round(res['int_rate']))}%.
                    """)

    # else:
    #     st.write('Waiting further information.')
        


    # Using object notation


    # Using "with" notation

        
        


        
        

