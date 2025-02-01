from flask import Flask, request, render_template
import pandas as pd
import pickle
import joblib

# Load the trained model and other preprocessing objects
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
print("model type:")
print(type(model))  # Check the type of the loaded model

feature_names = joblib.load('feature_names.joblib')
onehot_encoder = joblib.load('onehot_encoder.pkl')
label_encoder = joblib.load('label_encoder.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/pred')
def predict():
    return render_template("index.html")

@app.route('/out', methods=["POST"])
def output():
    print("abcs")
    # Get form data
    gender = request.form['gender']
  
    time_spend = request.form['Time_spend']
    feel = request.form['feel']
    felt_left = request.form['felt_left']
    avoid = request.form['avoid']
    affect = request.form['affect']
    amount = request.form['amount']
    pressure = request.form['pressure']
    experience = request.form['experience']
    worry = request.form['worry']
    distracted = request.form['distracted']
    others = request.form['others']
    share = request.form['share']
    anxious = request.form['Anxious']
    follow = request.form['Follow']
    interact = request.form['interact']
    impact = request.form['impact']

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[gender, time_spend, feel, felt_left, avoid, affect, amount, pressure, experience, worry, distracted, others, share, anxious, follow, interact, impact]], 
                              columns=['gender', 'Time_spend', 'feel', 'felt_left', 'avoid', 'affect', 'amount', 'pressure', 'experience', 'worry', 'distracted', 'others', 'share', 'Anxious', 'Follow', 'interact', 'impact'])

    # One-hot encode the categorical columns
    categorical_cols = input_data.select_dtypes(include=['object']).columns
    encoded_data = pd.DataFrame(onehot_encoder.transform(input_data[categorical_cols]), columns=onehot_encoder.get_feature_names_out(categorical_cols))
    
    # Combine the encoded columns with the original DataFrame
    input_data = input_data.drop(columns=categorical_cols)
    input_data = pd.concat([input_data, encoded_data], axis=1)

    # Ensure the input data has the same columns as the training data
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    # Predict using the model
    if hasattr(model, 'predict'):
        pred = model.predict(input_data)
        pred = label_encoder.inverse_transform(pred)[0]
        if pred == 'high':
            return render_template("output.html", y="This person requires mental health treatment.")
        else:
            return render_template("output.html", y="This person doesn't require mental health treatment.")
    else:
        return render_template("output.html", y="The model object does not have a predict method")

if __name__ == '__main__':
    app.run(debug=True)
