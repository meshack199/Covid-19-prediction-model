from datetime import datetime, time
from itertools import count
import joblib
from numpy import datetime_as_string 
import streamlit as st
classifier=joblib.load('dt.pkl')
# defining a function to generate predictions app
def prediction_generator(cough, fever, sore_throat, shortness_of_breath, head_ache, age_60_and_above, test_indication):
   prediction=classifier.predict([[cough, fever, sore_throat, shortness_of_breath, head_ache, age_60_and_above, test_indication]])
   if prediction == 0:
      output="negative"
   elif prediction == 1:
      output="other"
   else:
      output ="positive"
   return output

def main():
   st.title("Covid 19 Prediction App (pick one; i.e 1 for yes and 0 for No")
   
   count_iter = count()  # This will be the iterator you need

   cough = float(st.selectbox("Do you experience cough", ('0', '1'), key=f"cough_{next(count_iter)}"))
   fever = float(st.selectbox("Do you experience fever", ('0', '1'), key=f"fever_{next(count_iter)}"))
   sore_throat = float(st.selectbox("Do you have sore throat", ('0', '1'), key=f"sore_throat_{next(count_iter)}"))
   shortness_of_breath = float(st.selectbox("Do you feel shortness of breath", ('0', '1'), key=f"shortness_of_breath_{next(count_iter)}"))
   head_ache = float(st.selectbox("Do you feel headache", ('0', '1'), key=f"head_ache_{next(count_iter)}"))
   age_60_and_above = float(st.selectbox("Are you 60 years and above?", ("0", "1"), key=f"age_60_and_above_{next(count_iter)}"))
   test_indication = float(st.selectbox("Have you been in contact with confirmed or from abroad recently?", ("0", "1", "2"), key=f"test_indication_{next(count_iter)}"))

   result=" "
   if st.button("predict"):
      result=prediction_generator(cough, fever, sore_throat, shortness_of_breath, head_ache, age_60_and_above, test_indication)
      st.success(f'Covid-19 result is {result}')

if __name__== "__main__":
   main()


