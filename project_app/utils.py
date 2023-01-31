import config
import pickle
import json
import numpy as np

class MedicalInsurance:

    def __init__(self,data):
        self.data = data


    def load_model(self):
        with open (config.model_path,"rb") as f:
            self.model = pickle.load(f)
        with open (config.project_data_path, "r") as f:
            self.project_data = json.load(f)

    def predict_InsuranceCharges(self):
        self.load_model()

        sex_index = self.project_data["feature"].index("sex_" + self.data["sex"])
        smoker_index = self.project_data["feature"].index("smoker_" + self.data["smoker"])
        region_index = self.project_data["feature"].index("region_" + self.data["region"])
        test_array = np.zeros(len(self.project_data["feature"]))
        test_array[0] = int(self.data["age"])
        test_array[1] = float(self.data["bmi"])
        test_array[2] = float(self.data["children"])
        test_array[sex_index] = 1
        test_array[smoker_index] = 1
        test_array[region_index] = 1

        charges = np.around(self.model.predict([test_array])[0], 2)
        return charges




        

