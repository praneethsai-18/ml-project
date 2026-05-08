import os
import sys
import pandas as pd
import pickle


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:

            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            print("Checking files")

            print(os.path.exists(model_path))
            print(os.path.exists(preprocessor_path))

            with open(model_path, "rb") as file:
                model = pickle.load(file)

            with open(preprocessor_path, "rb") as file:
                preprocessor = pickle.load(file)

            print("Model Loaded")

            print(features.columns)

            data_scaled = preprocessor.transform(features)

            print("Data transformed")

            preds = model.predict(data_scaled)

            print(preds)

            return preds

        except Exception as e:

            print("ERROR :", e)

            raise e


class CustomData:

    def __init__(
        self,
        gender,
        race_ethnicity,
        parental_level_of_education,
        lunch,
        test_preparation_course,
        reading_score,
        writing_score
    ):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):

        custom_data_input_dict = {

            "gender": [self.gender],

            "race/ethnicity": [self.race_ethnicity],

            "parental level of education":
                [self.parental_level_of_education],

            "lunch": [self.lunch],

            "test preparation course":
                [self.test_preparation_course],

            "reading score": [self.reading_score],

            "writing score": [self.writing_score]
        }

        return pd.DataFrame(custom_data_input_dict)