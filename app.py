from flask import Flask, jsonify, request, Response
from keras.models import model_from_json
from keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
import lzstring
import json
import numpy as np

loaded_model_accel_x = None
loaded_model_accel_y = None
loaded_model_accel_z = None
loaded_model_gyro_x = None
loaded_model_gyro_y = None
loaded_model_gyro_z = None
loaded_model_magn_x = None
loaded_model_magn_y = None
loaded_model_magn_z = None

########### add model path
path = 'model path here'
app = Flask(__name__)


@app.before_request
def load_model_func():
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
    # path_model = path + model_name
    # path_weights = path + weights_name
    # json_file = open(path_model, 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights(path_weights)
    print("Loaded model from disk")
    # return loaded_model


@app.route('/', methods=['GET', 'POST'])
def main():
    return 'SDI action classifier is in the House!'


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        gyro = data['gyro']
        accel = data['accel']
        magn = data['magn']
        df_gyro = get_data_as_df(gyro)
        df_accel = get_data_as_df(accel)
        df_magn = get_data_as_df(magn)
        ms_resample = '2'
        df_gyro_resampled = resample_df(df_gyro, ms_resample)
        df_accel_resampled = resample_df(df_accel, ms_resample)
        df_magn_resampled = resample_df(df_magn, ms_resample)
        accel_x_ds = df_accel_resampled[['x']]
        accel_y_ds = df_accel_resampled[['y']]
        accel_z_ds = df_accel_resampled[['z']]

        gyro_x_ds = df_gyro_resampled[['x']]
        gyro_y_ds = df_gyro_resampled[['y']]
        gyro_z_ds = df_gyro_resampled[['z']]

        magn_x_ds = df_magn_resampled[['x']]
        magn_y_ds = df_magn_resampled[['y']]
        magn_z_ds = df_magn_resampled[['z']]

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

        final_accel_x_df = reshape_df(accel_x_ds, num_of_samples)
        final_accel_y_df = reshape_df(accel_y_ds, num_of_samples)
        final_accel_z_df = reshape_df(accel_z_ds, num_of_samples)

        final_gyro_x_df = reshape_df(gyro_x_ds, num_of_samples)
        final_gyro_y_df = reshape_df(gyro_y_ds, num_of_samples)
        final_gyro_z_df = reshape_df(gyro_z_ds, num_of_samples)

        final_magn_x_df = reshape_df(magn_x_ds, num_of_samples)
        final_magn_y_df = reshape_df(magn_y_ds, num_of_samples)
        final_magn_z_df = reshape_df(magn_z_ds, num_of_samples)

        resp = Response("Updated", status=200, mimetype='application/json')
    except Exception as e:
        resp = Response("something went wrong", status=500, mimetype='application/json')
    return resp
    # unzipped_gyro = unzip(gyro)
    # unzipped_accel = unzip(accel)
    # unzipped_magn = unzip(magn)
    # query_df = pd.DataFrame(json_)
    # query = pd.get_dummies(query_df)
    # prediction = classifier.predict(query)
    # return jsonify({'prediction': list(prediction)})


def unzip(together):
    x = lzstring.LZString
    data = x.decompressFromBase64(together)
    j = json.loads(data)
    return j


def get_data_as_df(array_to_load):
    after_split = array_to_load.split('}')
    df = pd.DataFrame()
    for i in after_split:
        i = i.replace('[','')
        if(i[0]==','):
            i = i[1: len(i)]
        i = i.replace(']','')
        if('' == i):
            print(i)
        else:
            i = i + '}'
            j_rec = eval(i)
            df = df.append(j_rec, ignore_index=True)
    return df

def resample_df(df, ms):
    df = df.set_index(['t'])
    resample = df.resample(ms)
    five_hz_mean = resample.mean()
    five_hz_mean = five_hz_mean.dropna()
    return  five_hz_mean

def reshape_df(df, n_samples):
    final_df = pd.DataFrame()
    if (df.shape[0] >= n_samples):
        y = int(df.shape[0] / n_samples)
        x = df[:y * n_samples]
        transpose = pd.DataFrame(np.reshape(x.values, (y, n_samples)))
    transpose['var'] = transpose[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]].var(axis=1)
    transpose['max'] = transpose[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]].max(axis=1)
    final_df = transpose[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 'var', 'max']]
    return  final_df

if __name__ == "__main__":
    app.run()
