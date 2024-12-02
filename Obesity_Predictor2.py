import streamlit as st
import numpy as np
import joblib

st.title('OBESITY PREDICTOR')

#Load Saved Model & preprocessing tools

model = joblib.load('ProjectModel.pkl')
fam_hist_with_overweight_encode = joblib.load('FamHist.pkl')
HighCaloricFoodIntake_encode = joblib.load('HighCalories.pkl')
FoodbtwnMeals_encode = joblib.load('FoodbtwnMeals.pkl')
Smoker_encode = joblib.load('Smoker.pkl')
AlcoholIntake_encode = joblib.load('AlcoholConsumption.pkl')
MeansOfTransport_encode = joblib.load('MeansOfTransport.pkl')

# Collect user input

meals_per_day = st.number_input('Meals per day', min_value=1, max_value=10, value=3)

physical_activities = st.number_input('Physical Activities (hours per week)', min_value=0, max_value=20, value=0)

fam_hist_with_Overweight = st.selectbox('Family History of Overweight', options=['yes', 'no'])
Fam_hist_with_Overweight_encoded = fam_hist_with_overweight_encode.transform([fam_hist_with_Overweight])[0]

HighCaloricFoodIntake = st.selectbox('High Food Calories Intake', options=['yes', 'no'])
HighCaloricFoodIntake_encoded = HighCaloricFoodIntake_encode.transform([HighCaloricFoodIntake])[0]

Food_between_meals = st.selectbox('Food Between Meals', options=['Sometimes', 'Frequently', 'Always', 'no'])
Food_between_meals_encoded = FoodbtwnMeals_encode.transform([Food_between_meals])[0]

Smoker = st.selectbox('Smoker', options=['yes', 'no'])
Smoker_encoded = Smoker_encode.transform([Smoker])[0]

Alcohol = st.selectbox('Alcohol Consumption', options=['no', 'Sometimes', 'Frequently'])
Alcohol_encoded = AlcoholIntake_encode.transform([Alcohol])[0]

Means_of_Transport = st.selectbox('Means of Transport', options=['Walking', 'Public_Transportation', 'Automobile', 'Motorbike', 'Bike'])
Means_of_transport_encoded = MeansOfTransport_encode.transform([Means_of_Transport])[0]


# Prepare input data for prediction
input_data = np.array([[meals_per_day, physical_activities, Fam_hist_with_Overweight_encoded, HighCaloricFoodIntake_encoded, Food_between_meals_encoded,
                        Smoker_encoded, Alcohol_encoded, Means_of_transport_encoded]])


# Predict obesity level when the user clicks the button
if st.button('Predict Obesity Level'):
    prediction = model.predict(input_data)
 
    prob = model.predict_proba(input_data)
    
    Obesity_Level = {0:'Insufficient_Weight', 1:'Normal_Weight', 2:'Obesity_Type_I', 3:'Obesity_Type_II',
                    4:'Obesity_Type_III', 5:'Overweight_Level_I', 6:'Overweight_Level_II'}

    confidence = np.max(prob * 100)
    
    predicted_level = Obesity_Level[prediction[0]]
    st.write(f'Predicted Obesity Level: {predicted_level}')
    st.write(f'The confidence level is {confidence} percent')



