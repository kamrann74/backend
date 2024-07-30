from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
model_path = r'C:\Users\User\OneDrive\Desktop\PROJECT 8 SEM\Project 8 - Copy\project kamran\my-app\Backend\best_model_pipeline.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

#Creating a dictionary
label_mapping ={
    0: 'Insuffiecient weight',
    1: ' Normal weight',
    2: 'Obesity level 1',
    3: 'Obesity_level 2',
    4: 'Obesity level 3',
    5: 'Overweight level 1',
    6:'Overweight level 2'

}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        print(f"Received data: {data}")

        # Convert JSON data to DataFrame
        df = pd.DataFrame([data])
        print("DataFrame before renaming columns:")
        print(df)

        # Rename columns to match model expectations
        df.rename(columns={
            'gender': 'Gender',
            'age': 'Age',
            'height': 'Height',
            'weight': 'Weight',
            'family_history_with_overweight': 'family_history_with_overweight',
            'FAVC': 'Freq_consump_of_high-caloric_food',
            'FCVC': 'Freq_of_veg_consump',
            'NCP': 'No._of_main_meals',
            'CAEC': 'Consump_of_food_btw_meals',
            'SMOKE': 'SMOKE',
            'CH2O': 'Daily_water_intake(L)',
            'FAF': 'Phy_activity_freq',
            'TUE': 'Time_using_tech_devices',
            'CALC': 'Alcohol_consumption',
            'MTRANS': 'Transportation_used'
        }, inplace=True)

        print("DataFrame after renaming columns:")
        print(df)

        # Ensure data types are correct
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['Height'] = pd.to_numeric(df['Height'], errors='coerce')
        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')

        print("DataFrame with correct data types:")
        print(df)

        # Check if all required columns are present
        required_columns = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'Freq_consump_of_high-caloric_food', 'Freq_of_veg_consump', 'No._of_main_meals', 'Consump_of_food_btw_meals', 'SMOKE', 'Daily_water_intake(L)', 'Phy_activity_freq', 'Time_using_tech_devices', 'Alcohol_consumption', 'Transportation_used']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

        # Ensure the DataFrame has the correct column order and missing columns filled with NaNs
        df = df.reindex(columns=required_columns, fill_value=np.nan)
        print("DataFrame after reindexing:")
        print(df)

        # Make prediction
        prediction = model.predict(df)[0]
        print(f"Prediction: {prediction}")
        
    
        prediction_label = label_mapping.get(prediction, "Unknown")
        return jsonify({'prediction': prediction_label})

    except Exception as e:
        print(f"Exception occurred: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
