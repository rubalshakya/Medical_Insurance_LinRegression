from flask import Flask,request,jsonify,render_template
from project_app.utils import MedicalInsurance

app = Flask(__name__)

@app.route("/",methods=["post","get"])
def base():
    return render_template("home.html")


@app.route("/predict",methods=["post"])
def predictCharges():
    data = dict(request.form)
    medical_Insurance = MedicalInsurance(data)

    charges = medical_Insurance.predict_InsuranceCharges()
    return render_template("result.html", result = charges)



app.run(debug=True)


