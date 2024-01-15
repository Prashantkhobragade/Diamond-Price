from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData


app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/predict", methods = ["GET", "POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("form.html")
    

if __name__ =="__main__":
    app.run(host="0.0.0.0", port=8000)