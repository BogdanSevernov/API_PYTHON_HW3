import unittest
import requests
import pandas as pd

DF_train_cl = pd.read_csv('train.csv')
DF_test_cl = pd.read_csv('test.csv')
headers = {'Content-Type': 'application/json'}

class ApiUniTest(unittest.TestCase):

    def test_get_MLModelsInfo(self):
        respons = requests.get('http://127.0.0.1:8080/api/ml_models_info')
        self.assertEqual(respons.status_code, 200)

    def test_post_ModelTrain(self):
        json_1 = {'data': DF_train_cl.to_json(), 'args': {'max_d': 3}, 'target': 'Survived', 'model_name': 'my_model'}
        respons = requests.post('http://127.0.0.1:8080/api/ml_models/train/1', json=json_1)
        self.assertEqual(respons.status_code, 200)
        self.assertEqual(type(respons.json()), dict)

    def test_post_ModelTest(self):
        json_2 = {'data': DF_test_cl.to_json(), 'model_name': 'my_model'}
        respons = requests.post('http://127.0.0.1:8080/api/ml_models/test/1', json=json_2)
        self.assertEqual(respons.status_code, 200)
        self.assertEqual(type(respons.json()), dict)

    def test_post_MLflow(self):
        json_2 = {'data': DF_test_cl.to_json(), 'model_name': 'my_model'}
        respons = requests.post('http://127.0.0.1:8080/api/ml_models/mlflow_test/1/1', json=json_2)
        self.assertEqual(respons.status_code, 200)
        self.assertEqual(type(respons.json()), dict)

    def test_all_in_one(self):
        json_1 = {'data': DF_train_cl.to_json(), 'args': {'max_d': 3}, 'target': 'Survived', 'model_name': 'my_model'}
        json_2 = {'data': DF_test_cl.to_json(), 'model_name': 'my_model'}
        respons = requests.get('http://127.0.0.1:8080/api/ml_models_info')
        self.assertEqual(respons.status_code, 200)
        respons = requests.post('http://127.0.0.1:8080/api/ml_models/train/1', json=json_1)
        self.assertEqual(respons.status_code, 200)
        self.assertEqual(type(respons.json()), dict)
        respons = requests.post('http://127.0.0.1:8080/api/ml_models/test/1', json=json_2)
        self.assertEqual(respons.status_code, 200)
        self.assertEqual(type(respons.json()), dict)
        respons = requests.post('http://127.0.0.1:8080/api/ml_models/mlflow_test/1/1', json=json_2)
        self.assertEqual(respons.status_code, 200)
        self.assertEqual(type(respons.json()), dict)

if __name__ == '__main__':
    unittest.main()
