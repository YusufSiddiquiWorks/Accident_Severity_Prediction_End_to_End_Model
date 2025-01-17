from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
application = Flask(__name__)
app = application

# Load the model and scaler
model = pickle.load(open('XBGC_Model_Accident_Severity_Prediction.pkl', 'rb'))
scaler = pickle.load(open('Scalar_Model_Accident_Severity_Prediction.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            Number_of_Casualties = float(request.form['Number_of_Casualties'])
            Number_of_Vehicles = float(request.form['Number_of_Vehicles'])
    

            # One-hot encoding for Light
            Light_Conditions = request.form['Light_Conditions']
            Darkness_Lights_Unknown = 1 if Light_Conditions == 'Darkness_Lights_Unknown' else 0
            Darkness_Lights_off = 1 if Light_Conditions == 'Darkness_Lights_off' else 0
            Darkness_Lights_on = 1 if Light_Conditions == 'Darkness_Lights_on' else 0
            Darkness_no_Lights = 1 if Light_Conditions == 'Darkness_no_Lights' else 0
            Daylight = 1 if Light_Conditions == 'Daylight' else 0

            # One-hot encoding for Road Surface
            Road_Surface_Conditions = request.form['Road_Surface_Conditions']
            Dry = 1 if Road_Surface_Conditions == 'Dry' else 0
            Flood = 1 if Road_Surface_Conditions == 'Flood' else 0
            Frost = 1 if Road_Surface_Conditions == 'Frost' else 0
            Snow = 1 if Road_Surface_Conditions == 'Snow' else 0
            Wet = 1 if Road_Surface_Conditions == 'Wet' else 0
            
            # One-hot encoding for Road_Type
            Road_Type = request.form['Road_Type']
            Dual_Carriageway = 1 if Road_Type == 'Dual carriageway' else 0
            Oneway = 1 if Road_Type == 'One way street' else 0
            Roundabout = 1 if Road_Type == 'Roundabout' else 0
            Single_Carriageway = 1 if Road_Type == 'Single carriageway' else 0
            Sliproad = 1 if Road_Type == 'Slip road' else 0
          

            # One-hot encoding for Weather Condition
            weather = request.form['Weather_Conditions']
            Weather_Condition_Fine_Highwind = 1 if weather == 'Fine + high winds' else 0
            Weather_Condition_Fine_Nowind = 1 if weather == 'Fine no high winds' else 0
            Weather_Condition_Fog = 1 if weather == 'Fog or mist' else 0
            Weather_Condition_other = 1 if weather == 'Other' else 0
            Weather_Condition_RainHighwind = 1 if weather == 'Raining + high winds' else 0
            Weather_Condition_RainNoWind = 1 if weather == 'Raining no high winds' else 0
            Weather_Condition_Snow_HighWind = 1 if weather == 'Snowing + high winds' else 0
            Weather_Condition_Snow_nowind = 1 if weather == 'Snowing no high winds' else 0
            
             # One-hot encoding for Area
            Urban_or_Rural_Area = request.form['Urban_or_Rural_Area']
            Rural = 1 if Urban_or_Rural_Area == 'Rural' else 0
            Urban = 1 if Urban_or_Rural_Area == 'Urban' else 0
            
            
             # One-hot encoding for Vehicle Type
            Vehicle_Type = request.form['Vehicle_Type']
            Argricultural_Vehicle = 1 if Vehicle_Type == 'Agricultural vehicle' else 0
            Bus = 1 if Vehicle_Type == 'Bus' else 0
            Car = 1 if Vehicle_Type == 'Car' else 0
            Goods_Carrier = 1 if Vehicle_Type == 'Goods Carrier' else 0
            Motorcycle = 1 if Vehicle_Type == 'MotorCycle' else 0
            Other = 1 if Vehicle_Type == 'Other vehicle' else 0
            Pedal_Cycle = 1 if Vehicle_Type == 'Pedal cycle' else 0
            Horse = 1 if Vehicle_Type == 'Ridden horse' else 0

            # Prepare input data for scaling
            input_data = np.array([Number_of_Casualties,Number_of_Vehicles,Darkness_Lights_off,Darkness_Lights_on,Darkness_Lights_Unknown,Darkness_no_Lights,Daylight,Dry,Flood,Frost,Snow,Wet,Dual_Carriageway,Oneway,Roundabout,Single_Carriageway,Sliproad,Weather_Condition_Fine_Highwind,Weather_Condition_Fine_Nowind,Weather_Condition_Fog,Weather_Condition_other,Weather_Condition_RainHighwind,Weather_Condition_RainNoWind,Weather_Condition_Snow_HighWind,Weather_Condition_Snow_nowind,Rural,Urban,Argricultural_Vehicle,Bus,Car,Goods_Carrier,Motorcycle,Other,Pedal_Cycle,Horse]).reshape(1, -1) 

            # Scale the input data using the loaded scaler
            

            # Prediction
            output_labels =  ['Slight', 'Serious', 'Fatal']
            prediction = model.predict(input_data)
            result = output_labels[prediction[0]]
            print(result)
            
            return render_template('prediction.html', prediction=result)
        except ValueError:
            return render_template('index.html', error_message="Please select at least one option for each group.")
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')