# Main entry point for our project
from flask import Flask, render_template, request
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

#Flask app - starting point of our api
app = Flask(__name__)

@app.route('/') #homepage
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST']) # predict route
def predict():
    if request.method == 'POST':
        try:
            # Get the input text from form
            text = request.form.get('message')
            
            # Create CustomData object
            data = CustomData(text=text)
            
            # Make prediction
            pred_pipeline = PredictPipeline()
            prediction = pred_pipeline.predict(text)
            
            # Convert prediction to numeric format for the template
            result = 1 if prediction == 1 else 0
            
            return render_template('index.html', 
                                result=result)  # Match the template's expected variable
            
        except Exception as e:
            error_message = str(e)
            return render_template('index.html',
                                error=f"Error occurred: {error_message}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True) # localhost ip address = 0.0.0.0:5000