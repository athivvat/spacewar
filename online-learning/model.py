import json
import logging
import microgear.client as microgear
import time

from river import cluster
from river import preprocessing
from river import stream
from river import metrics

# NETPIE Environment
appid = 'datastream'
gearkey = 'qY0dhxc3TAswzeC'
gearsecret = 'eNInuhdaicInPOJl0KfPrBJfS'
user_score_topic = '/401Chathai_data'

microgear.create(gearkey, gearsecret, appid, {'debugmode': True})

feature = None
y_pred = None


def callback_connect():
    logging.info("Now I am connected with netpie")


def disconnect():
    logging.info("disconnected")


def callback_message(topic, message):
    import ast
    global feature

    try:
        if topic == f"/{appid}{user_score_topic}" and message:
            feature = json.loads(ast.literal_eval(message).decode('utf-8'))
    except Exception:
        pass
    # logging.info(message)
    #logging.info(print(f'{feature} is assigned to cluster {y_pred}'))

def callback_error(msg):
    print("error", msg)


microgear.setalias("tanapong")
microgear.on_connect = callback_connect
microgear.on_message = callback_message
microgear.on_error = callback_error
microgear.on_disconnect = disconnect
microgear.subscribe(user_score_topic)
microgear.connect(False)

k_means = cluster.KMeans(n_clusters=3, halflife=0.4, sigma=3, seed=0)
metric = metrics.cluster.Silhouette()
scaler = preprocessing.StandardScaler()

metrics = []
while True:
    if feature is not None:
        x = scaler.learn_one(feature).transform_one(feature)
        k_means = k_means.learn_one(x)
        y_pred = k_means.predict_one(x)
        metric = metric.update(x, y_pred, k_means.centers)

        print(f'Assigned to cluster {y_pred} == Silhouette: {metric}')


