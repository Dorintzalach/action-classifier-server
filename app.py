from flask import Flask, request, Response
from keras.models import model_from_json
import pandas as pd
import numpy as np


global loaded_model_accel_x
global loaded_model_accel_y
global loaded_model_accel_z
global loaded_model_gyro_x
global loaded_model_gyro_y
global loaded_model_gyro_z
global loaded_model_magn_x
global loaded_model_magn_y
global loaded_model_magn_z

lables = ['game', 'pocket', 'rest', 'stairs', 'shaking', 'texting', 'walking']
lables_accel = ['game', 'rest', 'pocket', 'stairs', 'shaking', 'texting', 'walking']

########### add model path
path = 'C:/Users/dorin/PycharmProjects/action-classfier/Models 28.4 with var and max'
app = Flask(__name__)


@app.before_request
def load_model_func():
    global loaded_model_accel_x
    global loaded_model_accel_y
    global loaded_model_accel_z
    global loaded_model_gyro_x
    global loaded_model_gyro_y
    global loaded_model_gyro_z
    global loaded_model_magn_x
    global loaded_model_magn_y
    global loaded_model_magn_z
    loaded_model_accel_x = load_model('/accel_x.json', '/accel_x.h5')
    loaded_model_accel_y = load_model('/accel_y.json', '/accel_y.h5')
    loaded_model_accel_z = load_model('/accel_z.json', '/accel_z.h5')
    loaded_model_gyro_x = load_model('/gyro_x.json', '/gyro_x.h5')
    loaded_model_gyro_y = load_model('/gyro_y.json', '/gyro_y.h5')
    loaded_model_gyro_z = load_model('/gyro_z.json', '/gyro_z.h5')
    loaded_model_magn_x = load_model('/magn_x.json', '/magn_x.h5')
    loaded_model_magn_y = load_model('/magn_y.json', '/magn_y.h5')
    loaded_model_magn_z = load_model('/magn_z.json', '/magn_z.h5')


def load_model(model_name, weights_name):
    path_model = path + model_name
    path_weights = path + weights_name
    json_file = open(path_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path_weights)
    print("Loaded model from disk")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model


@app.route('/', methods=['GET', 'POST'])
def main():
    return 'SDI action classifier is in the House!'


@app.route('/predict', methods=['POST'])
def predict():
    try:
        global loaded_model_accel_x
        global loaded_model_accel_y
        global loaded_model_accel_z
        global loaded_model_gyro_x
        global loaded_model_gyro_y
        global loaded_model_gyro_z
        global loaded_model_magn_x
        global loaded_model_magn_y
        global loaded_model_magn_z
        data = request.get_json()
        gyro = data['gyro']
        accel = data['accel']
        magn = data['magn']
        df_gyro = get_data_as_df(gyro)
        df_accel = get_data_as_df(accel)
        df_magn = get_data_as_df(magn)
        accel_x_ds = df_accel[['x']]
        accel_y_ds = df_accel[['y']]
        accel_z_ds = df_accel[['z']]

        gyro_x_ds = df_gyro[['x']]
        gyro_y_ds = df_gyro[['y']]
        gyro_z_ds = df_gyro[['z']]

        magn_x_ds = df_magn[['x']]
        magn_y_ds = df_magn[['y']]
        magn_z_ds = df_magn[['z']]

        accel_x_ds = accel_x_ds.reset_index()
        accel_y_ds = accel_y_ds.reset_index()
        accel_z_ds = accel_z_ds.reset_index()

        gyro_x_ds = gyro_x_ds.reset_index()
        gyro_y_ds = gyro_y_ds.reset_index()
        gyro_z_ds = gyro_z_ds.reset_index()

        magn_x_ds = magn_x_ds.reset_index()
        magn_y_ds = magn_y_ds.reset_index()
        magn_z_ds = magn_z_ds.reset_index()

        num_of_samples = 12

        final_accel_x_df = reshape_df(accel_x_ds, num_of_samples, 'x')
        final_accel_y_df = reshape_df(accel_y_ds, num_of_samples, 'y')
        final_accel_z_df = reshape_df(accel_z_ds, num_of_samples, 'z')

        final_gyro_x_df = reshape_df(gyro_x_ds, num_of_samples, 'x')
        final_gyro_y_df = reshape_df(gyro_y_ds, num_of_samples, 'y')
        final_gyro_z_df = reshape_df(gyro_z_ds, num_of_samples, 'z')

        final_magn_x_df = reshape_df(magn_x_ds, num_of_samples, 'x')
        final_magn_y_df = reshape_df(magn_y_ds, num_of_samples, 'y')
        final_magn_z_df = reshape_df(magn_z_ds, num_of_samples, 'z')


        accel_x_pred = loaded_model_accel_x.predict_classes(final_accel_x_df)
        accel_y_pred = loaded_model_accel_y.predict_classes(final_accel_y_df)
        accel_z_pred = loaded_model_accel_z.predict_classes(final_accel_z_df)

        gyro_x_pred = loaded_model_gyro_x.predict_classes(final_gyro_x_df)
        gyro_y_pred = loaded_model_gyro_y.predict_classes(final_gyro_y_df)
        gyro_z_pred = loaded_model_gyro_z.predict_classes(final_gyro_z_df)

        magn_x_pred = loaded_model_magn_x.predict_classes(final_magn_x_df)
        magn_y_pred = loaded_model_magn_y.predict_classes(final_magn_y_df)
        magn_z_pred = loaded_model_magn_z.predict_classes(final_magn_z_df)

        voices = {}
        for i in range(0, 7):
            voices[i] = 0

        if(accel_x_pred[0]==1):
            accel_x_pred[0]=2
        elif (accel_x_pred[0]==2):
            accel_x_pred[0] = 1

        if(accel_y_pred[0]==1):
            accel_y_pred[0]=2
        elif (accel_y_pred[0]==2):
            accel_y_pred[0] = 1

        if(accel_z_pred[0]==1):
            accel_z_pred[0]=2
        elif (accel_z_pred[0]==2):
            accel_z_pred[0] = 1

        voices[gyro_x_pred[0]] += 1
        voices[gyro_y_pred[0]] += 1
        voices[gyro_z_pred[0]] += 1

        voices[accel_x_pred[0]] += 2
        voices[accel_y_pred[0]] += 3
        voices[accel_z_pred[0]] += 2

        voices[magn_x_pred[0]] += 1
        voices[magn_y_pred[0]] += 1
        voices[magn_z_pred[0]] += 1

        keymax = max(voices, key=voices.get)
        resp = Response(lables[keymax], status=200, mimetype='application/json')
    except Exception as e:
        resp = Response(e, status=500, mimetype='application/json')
    return resp


def get_data_as_df(array_to_load):
    after_split = array_to_load.split('}')
    df = pd.DataFrame()
    for i in after_split:
        i = i.replace('[', '')
        if (i[0] == ','):
            i = i[1: len(i)]
        i = i.replace(']', '')
        if ('' == i):
            print(i)
        else:
            i = i + '}'
            j_rec = eval(i)
            df = df.append(j_rec, ignore_index=True)
    return df


def resample_df(df, ms):
    for index, row in df.iterrows():
        print(row['t'].astype('datetime64[ms]'))
    df['t'] = df['t'].astype('datetime64[ns]')
    df = df.set_index(['t'])
    resample = df.resample(ms)
    five_hz_mean = resample.mean()
    five_hz_mean = five_hz_mean.dropna()
    print(five_hz_mean.shape)
    return five_hz_mean


def reshape_df(df, n_samples, axis):
    final_df = pd.DataFrame()
    transpose = pd.DataFrame()
    x = df[[axis]]
    if (x.shape[0] >= n_samples):
        y = int(x.shape[0] / n_samples)
        x = x[:y * n_samples]
        transpose = pd.DataFrame(np.reshape(x.values, (y, n_samples)))
    transpose['var'] = transpose[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]].var(axis=1)
    transpose['max'] = transpose[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]].max(axis=1)
    final_df = transpose[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 'var', 'max']]
    return final_df


def get_action(sensor_type, classification):
    if (sensor_type == 'accel'):
        return lables_accel[classification]
    else:
        return lables[classification]


if __name__ == "__main__":
    app.run()
