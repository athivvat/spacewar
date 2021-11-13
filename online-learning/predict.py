import json
import logging
import pickle
import microgear.client as microgear

from river import preprocessing
from river import metrics

# NETPIE Environment
appid = 'datastream'
gearkey = 'qY0dhxc3TAswzeC'
gearsecret = 'eNInuhdaicInPOJl0KfPrBJfS'
user_data_topic = '/401Chathai_data'
user_pred_topic = '/401Chathai_pred'

microgear.create(gearkey, gearsecret, appid, {'debugmode': True})

user = None
feature = None
y_pred = None
model = None
metric = None

scaler = preprocessing.StandardScaler()
metric = metrics.cluster.Cohesion()

with open('./model.pkl', 'rb') as f:
    model = pickle.load(f)

type_label = {0: 'Casual Achiever', 1: 'Casual Killer', 2: 'Hardcore Killer', 3: 'Hardcore Achiever' }


def callback_connect():
    logging.info("Now I am connected with netpie")


def disconnect():
    logging.info("disconnected")


def callback_message(topic, message):
    import ast
    global user
    global feature
    global model
    global metric

    try:
        if topic == f"/{appid}{user_data_topic}" and message:
            res = json.loads(ast.literal_eval(message).decode('utf-8'))
            user = res['user']
            feature = res['feature']

            x = scaler.learn_one(feature).transform_one(feature)
            model = model.learn_one(x)
            y_pred = model.predict_one(x)
            metric = metric.update(x, y_pred, model.centers)

            player_type = type_label[y_pred]
            data = dict(user=user, type=player_type)
            microgear.publish(user_pred_topic, json.dumps(data))

            logging.info(f'User: {user} | Predict: {y_pred} | Type: {player_type} | Metric: {metric}')

    except Exception:
        pass


def callback_error(msg):
    print("error", msg)


microgear.setalias("tanapong")
microgear.on_connect = callback_connect
microgear.on_message = callback_message
microgear.on_error = callback_error
microgear.on_disconnect = disconnect
microgear.subscribe(user_data_topic)
microgear.connect(True)