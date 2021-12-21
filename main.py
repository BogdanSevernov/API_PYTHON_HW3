import pandas as pd
from flask import Flask
from flask_restx import Api, Resource
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn import metrics as metric
import joblib
import sqlite3
import os
import logging
from prometheus_flask_exporter import PrometheusMetrics
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import mlflow.pyfunc
from flask_mail import Mail, Message

app = Flask(__name__)
api = Api(app)
metrics = PrometheusMetrics(app)

#параметры для отправки сообщения на mail
mail_settings = {
    'MAIL_SERVER': 'smtp.gmail.com',
    'MAIL_PORT': 465,
    'MAIL_USE_SSL': True,
    'MAIL_USERNAME': 'bsevernov@gmail.com',
    'MAIL_PASSWORD': os.environ['Password']
}
app.config.update(mail_settings)
mail = Mail(app)

# Настройка логирования
fh = logging.FileHandler("logs.log", mode='w')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

class MLModels():
    def __init__(self):
        self.ml_models = {'models': [{'id': 1, 'task': 'classification', 'model_name': 'tree'},
                                     {'id': 2, 'task': 'regression', 'model_name': 'linear_regression'}]}
        self.model_directory = 'model'


    def data_preprocessing(self, data):
        """Этот метод нужен для предварительной обработки данных"""
        logger = logging.getLogger("data preprocessing")
        logger.setLevel(logging.INFO)
        logger.addHandler(fh)
        logger.info("data preprocessing started")
        data = data
        # работа с null значениями
        for col in data.columns:
            # Количество пустых значений
            temp_null_count = data[data[col].isnull()].shape[0]
            dt = str(data[col].dtype)
            # Ищем поля с типом 'float64' или 'int64'
            if temp_null_count > 0 and (dt == 'float64' or dt == 'int64'):
                temp_data = data[[col]]
                imp_num = SimpleImputer(strategy='mean')
                data_num_imp = imp_num.fit_transform(temp_data)
                data[col] = data_num_imp
            # Ищем поля с типом 'object'
            elif temp_null_count > 0 and (dt == 'object'):
                temp_data = data[[col]]
                imp_num = SimpleImputer(strategy='most_frequent')
                data_num_imp = imp_num.fit_transform(temp_data)
                data[col] = data_num_imp

        # работа с категориальными признаками
        le = LabelEncoder()
        for col in data.columns:
            dt = str(data[col].dtype)
            if dt == 'object':
                cat_enc_le = le.fit_transform(data[col])
                data[col] = cat_enc_le
        return data

    def train_test_split_fun(self, data, target, id):
        """Метод для разделения данных на тренеровочные и тестовые. Он нужен чтобы выводить метрики качества"""
        logger = logging.getLogger("data preprocessing")
        logger.setLevel(logging.INFO)
        logger.addHandler(fh)
        if target not in data.columns:
            api.abort(404, 'target={} not in data columns'.format(target))
        data = self.data_preprocessing(data)
        X = data.drop(columns=[str(target)])
        y = data[str(target)]
        if int(id) == 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=777, stratify=y)
        elif int(id) == 2:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=777)
        logger.info("data preprocessing finished")
        return X_train, X_test, y_train, y_test

    def train_models(self, data, id, target, model_name, **params):
        """Метод для обучения моделей"""
        logger = logging.getLogger("model train")
        logger.setLevel(logging.INFO)
        logger.addHandler(fh)
        data = data
        X_train, X_test, y_train, y_test = self.train_test_split_fun(data, target, id)
        logger.info("train started")
        if int(id) == 1:
            mlflow.set_tracking_uri('http://localhost:5001/')
            mlflow.set_experiment('classification2')
            with mlflow.start_run(run_name='classification_models'):
                clf = DecisionTreeClassifier(max_depth=int(params['max_d']), random_state=0)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = round(metric.accuracy_score(y_test, y_pred), 2)
                mlflow.log_params(clf.get_params())
                mlflow.log_metrics({'acc': acc})
                signature = infer_signature(X_test, clf.predict(X_test))
                mlflow.sklearn.log_model(clf, 'ml_model', signature=signature, registered_model_name=model_name)

            model_file_name = '{}/{}_id1.pkl'.format(self.model_directory, model_name)
            joblib.dump(clf, model_file_name)
            model_info = {'acc': acc, 'model_name': model_name}
            logger.info('train finished')
            return model_info
        elif int(id) == 2:
            mlflow.set_tracking_uri('http://localhost:5001/')
            mlflow.set_experiment('LinearRegression')
            with mlflow.start_run(run_name='regression'):
                lin_model = LinearRegression()
                lin_model.fit(X_train, y_train)
                y_pred = lin_model.predict(X_test)
                mse = round(metric.mean_squared_error(y_test, y_pred))
                mlflow.log_params(lin_model.get_params())
                mlflow.log_metrics({'mse': mse})
                signature = infer_signature(X_test, lin_model.predict(X_test))
                mlflow.sklearn.log_model(lin_model, 'ml_model', signature=signature, registered_model_name=model_name)

            model_file_name = '{}/{}_id2.pkl'.format(self.model_directory, model_name)
            joblib.dump(lin_model, model_file_name)
            model_info = {'mse': mse, 'model_name': model_name}
            logger.info('train finished')
            return model_info

    def predict(self, id, model_name, data):
        """Метод для прогнозирования"""
        logger = logging.getLogger("model predict")
        logger.setLevel(logging.INFO)
        logger.addHandler(fh)
        logger.info("predict started")
        if int(id) == 1:
            model_path = '{}/{}_id1.pkl'.format(self.model_directory, model_name)
            # если файл не существует, то выводится ошибка
            if os.path.exists(model_path):
                clf = joblib.load(model_path)
                y_pred = clf.predict(data)
                logger.info("predict finished")
                return {'prediction': list(map(int, y_pred))}
            else:
                api.abort(404, 'ml_model {} doesnt exist'.format(model_name))
        elif int(id) == 2:
            model_path = '{}/{}_id2.pkl'.format(self.model_directory, model_name)
            # если файл не существует, то выводится ошибка
            if os.path.exists(model_path):
                lin_model = joblib.load(model_path)
                y_pred = lin_model.predict(data)
                logger.info("predict finished")
                return {'prediction': list(map(float, y_pred))}
            else:
                api.abort(404, 'ml_model {} doesnt exist'.format(model_name))
        else:
            logger.error('ml_model {} doesnt exist'.format(model_name))

    def predict_mlflow(self, model_name, model_number, id, data):
        mlflow.set_tracking_uri('http://localhost:5001/')
        mlflow.set_experiment('classification2')
        data = self.data_preprocessing(data)
        model_path = 'models:/' + model_name + '/{}'.format(model_number)
        pred_model = mlflow.pyfunc.load_model(model_uri=model_path)
        y_pred = pred_model.predict(data)
        if int(id) == 1:
            return {'prediction': list(map(int, y_pred))}
        if int(id) == 2:
            return {'prediction': list(map(float, y_pred))}


    def delete(self, del_model_path):
        """Метод для удаления существующих моделей"""
        logger = logging.getLogger("model delete")
        logger.setLevel(logging.INFO)
        logger.addHandler(fh)
        os.remove(del_model_path)
        logger.info("model deleted")

#Класс для работы с БД
class Api_db():
    #метод для создания БД(pi_db.db) и таблицы(api_table)
    def create_db_table(self):
        logger = logging.getLogger("api db")
        logger.setLevel(logging.INFO)
        logger.addHandler(fh)

        sqlite_connection = sqlite3.connect('api_db.db')
        if sqlite_connection:
            cursor = sqlite_connection.cursor()
            logger.info('connected with db')

        else:
            logger.error('error connection')
            api.abort(404, 'error connection')

        sqlite_create_table_query = '''CREATE TABLE IF NOT EXISTS api_table (model_id INTEGER NOT NULL,
                                                               model_name TEXT NOT NULL,
                                                               model_metric REAL NOT NULL);'''
        cursor.execute(sqlite_create_table_query)
        sqlite_connection.commit()
        cursor.close()
        sqlite_connection.close()
    # метод для чтения из БД
    def get(self, model_id):
        # логирование
        logger = logging.getLogger("api db")
        logger.setLevel(logging.INFO)
        logger.addHandler(fh)

        sqlite_connection = sqlite3.connect('api_db.db', timeout=20)
        cursor = sqlite_connection.cursor()
        sqlite_select_query = """SELECT * from api_table where  model_id = {}; """.format(model_id)
        cursor.execute(sqlite_select_query)
        row = cursor.fetchone()
        cursor.close()
        sqlite_connection.close()
        logger.info('read from table')
        return {'model_id': row[0], 'model_name': row[1], 'model_metric': row[2]}

    # метод для добавления информации в БД
    def post(self, model_id, model_name, model_metric):
        # логирование
        logger = logging.getLogger("api db")
        logger.setLevel(logging.INFO)
        logger.addHandler(fh)

        sqlite_connection = sqlite3.connect('api_db.db')
        cursor = sqlite_connection.cursor()
        cursor.execute("INSERT INTO api_table (model_id, model_name, model_metric) VALUES(?, ?, ?);",
                       (model_id, model_name, model_metric))
        sqlite_connection.commit()
        cursor.close()
        sqlite_connection.close()
        logger.info('info added to table')
    # метод для удаления данных из БД
    def delete(self, model_id):
        # логирование
        logger = logging.getLogger("api db")
        logger.setLevel(logging.INFO)
        logger.addHandler(fh)

        sqlite_connection = sqlite3.connect('api_db.db', timeout=20)
        cursor = sqlite_connection.cursor()
        sqlite_select_query = """DELETE from api_table where  model_id = {}; """.format(model_id)
        cursor.execute(sqlite_select_query)
        cursor.close()
        sqlite_connection.close()
        logger.info('info deleted from table')


ml = MLModels()
db = Api_db()
db.create_db_table()

@api.route('/api/ml_models_info')
class MLModelsInfo(Resource):
    @metrics.counter('cnt_gets_MLModelsInfo', 'Number of gets', labels={'status': lambda resp: resp.status_code})
    def get(self):
        models = ml.ml_models
        return models

@api.route('/api/ml_models/train/<int:id>')
class ModelTrain(Resource):
    @metrics.counter('cnt_post_ModelTrain', 'Number of posts', labels={'status': lambda resp: resp.status_code})
    def post(self, id):
        data = pd.read_json(api.payload['data'])
        target = api.payload['target']
        args = api.payload['args']
        model_name = api.payload['model_name']
        train_info = ml.train_models(data, id, str(target), model_name, **dict(args))
        if id == 1:
            db.post(id, train_info['model_name'], train_info['acc'])
        else:
            db.post(id, train_info['model_name'], train_info['mse'])
        return train_info

    @metrics.counter('cnt_delete_ModelTrain', 'Number of delete', labels={'status': lambda resp: resp.status_code})
    def delete(self, id):
        model_name = api.payload['model_name']
        model_del = '{}/{}_id{}.pkl'.format(ml.model_directory, model_name, id)
        ml.delete(model_del)

@api.route('/api/ml_models/test/<int:id>')
class ModelTest(Resource):
    @metrics.counter('cnt_post_ModelTest', 'Number of posts', labels={'status': lambda resp: resp.status_code})
    def post(self, id):
        DF_test = pd.read_json(api.payload['data'])
        model_name = api.payload['model_name']
        DF_test = ml.data_preprocessing(DF_test)
        X_test = DF_test.to_numpy()
        pred = ml.predict(id, model_name, X_test)
        return pred

# Ручка для тестирования ml моделей на mlflow
@api.route('/api/ml_models/mlflow_test/<int:model_number>/<int:id>')
class Mlflowtest(Resource):
    @metrics.counter('cnt_post_Mlflowtest', 'Number of posts', labels={'status': lambda resp: resp.status_code})
    def post(self, model_number, id):
        DF_test = pd.read_json(api.payload['data'])
        model_name = api.payload['model_name']
        pred = ml.predict_mlflow(model_name, model_number, id, DF_test)
        return pred

@api.route('/api/read/<int:id>')
class ReadDb(Resource):
    @metrics.counter('cnt_gets_ReadDb', 'Number of gets', labels={'status': lambda resp: resp.status_code})
    def get(self, id):
        return db.get(id)

@api.route('/api/delete/<int:id>')
class UpdateDB(Resource):
    @metrics.counter('cnt_delete_UpdateDB', 'Number of deletes', labels={'status': lambda resp: resp.status_code})
    def delete(self, id):
        db.delete(id)

# Ручка для отправки сообщения на почту
@api.route('/mail')
class Mail(Resource):
    def post(self):
        address = api.payload['address']
        msg = Message('Thank you for choosing our service', sender='bsevernov@gmail.com', recipients=['bvsevernov@edu.hse.ru'])
        msg.html = '<h1>EMAIL</h1><p>Text</p>'
        mail.send(msg)
        return '', 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

