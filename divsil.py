#!/usr/bin/env python
# Resources:
# https://ai.stackexchange.com/questions/2008/how-can-neural-networks-deal-with-varying-input-sizes
#
from sklearn.model_selection import train_test_split
from keras.utils import normalize
import tensorflow as tf
import os.path as op
import numpy as np
import json
import os
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['GOTO_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.eviron['openmp'] = 'True'
import code

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
IN_ORD_MATRIX_PATH = 'in_ord_matrix.npy' # (31,)
OUT_ORD_MATRIX_PATH = 'out_ord_matrix.npy' # (11,)
MODEL_PATH = 'model.h5'
config = tf.ConfigProto(device_count={"CPU": 6})

array2pal = lambda array: ''.join([chr(ord) for ord in array])
pal2array = lambda word: [ord(chr) for chr in word]

if not op.exists(OUT_ORD_MATRIX_PATH) or not op.exists(IN_ORD_MATRIX_PATH):

    palavras_path = '/home/nexor/prog/python/portal-da-lingua-portuguesa/palavras-divisao-silabica.json'
    palavras = json.load(open(palavras_path, 'rb'))
    max_len_pal = max_qtd_hifen = 0

    for char in palavras:
        palavras_q_comecam_com_char = palavras[char]
        for palavra,dividida in palavras_q_comecam_com_char.items():
            if max_len_pal < len(palavra): max_len_pal = len(palavra)
            qtd_hifen = dividida.count('-')
            if max_qtd_hifen < qtd_hifen:
                max_qtd_hifen = qtd_hifen

    in_ord_matrix = []
    out_ord_matrix = []
    for c,palavras_q_comecam_com_char in palavras.items():
        for palavra,dividida in palavras_q_comecam_com_char.items():
            palavra_tam_fixo = ('{:<%d}' % max_len_pal).format(palavra)
            in_ord_matrix += [pal2array(palavra_tam_fixo)]

            ixs_hifen = [i for i, a in enumerate(dividida) if a == '-']
            if len(ixs_hifen) < max_qtd_hifen:
                for i in range(max_qtd_hifen - len(ixs_hifen)):
                    ixs_hifen += [-1]
            out_ord_matrix += [ixs_hifen]


    in_ord_matrix = np.array(in_ord_matrix)
    out_ord_matrix = np.array(out_ord_matrix)

    np.save(open(IN_ORD_MATRIX_PATH, 'wb'), in_ord_matrix)
    np.save(open(OUT_ORD_MATRIX_PATH, 'wb'), out_ord_matrix)

else:
    X = in_ord_matrix = np.load(open(IN_ORD_MATRIX_PATH, 'rb'))
    Y = out_ord_matrix = np.load(open(OUT_ORD_MATRIX_PATH, 'rb'))

    X = normalize(X, axis=-1, order=2)
    Y = normalize(Y, axis=-1, order=2)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    def keras_approach(X, Y):
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=16)) as sess:
            from keras.backend import tensorflow_backend as K
            K.set_session(sess)
            if op.exists(MODEL_PATH):
                from keras.models import load_model
                model = load_model(MODEL_PATH)
            else:
                from keras.models import Sequential
                from keras.layers import Dense
                model = Sequential()
                model.add(Dense(32, input_dim=31))
                model.add(Dense(20))
                model.add(Dense(20))
                model.add(Dense(11))
                from keras.optimizers import SGD
                opt = SGD(lr=0.0001)
                model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

            from keras.callbacks import ModelCheckpoint
            checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
            model.fit(X,Y, epochs=1000, batch_size=25, callbacks=[checkpointer])
            scores = model.evaluate(X,Y,verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            model.save(MODEL_PATH)

    def keras_predict(palavra):



    def autosklearn_approach(X,Y):
        import sklearn
        from autosklearn.regression import AutoSklearnRegressor
        automl = AutoSklearnRegressor(
                time_left_for_this_task=120,
                per_run_time_limit=30,
                tmp_folder='/tmp/autosklearn_regression_example_tmp',
                output_folder='/tmp/autosklearn_regression_example_out',
            )
        automl.fit(X, Y)
        print(automl.show_models())
        predictions = automl.predict(X_test)
        print("R2 score:", sklearn.metrics.r2_score(y_test, predictions))


#     autosklearn_approach(X_train,Y_train)
    keras_approach(X_train, Y_train)

