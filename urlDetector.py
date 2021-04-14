from flask import Flask, redirect, url_for , render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from load_model import logit_predict




app = Flask(__name__)


@app.route("/")
def main():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def dumb():
    url = request.form["url"]
    url=[url]
    answer = logit_predict(url)
    prediction = answer[0]
    return render_template("predict.html",data=prediction)

if __name__ == "__main__":
    app.run(debug=True) 







