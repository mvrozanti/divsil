#!/usr/bin/env python
# Resources:
# https://ai.stackexchange.com/questions/2008/how-can-neural-networks-deal-with-varying-input-sizes
#
from sklearn.model_selection import train_test_split
from keras.models import load_model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.utils import normalize
from keras.optimizers import SGD
from keras.layers import Dense
import tensorflow as tf
import os.path as op
import numpy as np
import argparse
import json
import code
import sys
import os
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['GOTO_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['openmp'] = 'True'


np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
IN_ORD_MATRIX_PATH = 'in_ord_matriz.npy' # (31,)
OUT_ORD_MATRIX_PATH = 'out_ord_matriz.npy' # (11,)
config = tf.ConfigProto(device_count={"CPU": 6})

array2pal = lambda array: ''.join([chr(ord) for ord in array])
pal2array = lambda word: [ord(chr) for chr in word]
pad_palavra = lambda palavra, max_len_pal: ('{:<%d}' % max_len_pal).format(palavra)

def json2matriz():
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

    in_ord_matriz = []
    out_ord_matriz = []
    for c,palavras_q_comecam_com_char in palavras.items():
        for palavra,dividida in palavras_q_comecam_com_char.items():
            palavra_tam_fixo = pad_palavra(palavra, max_len_pal)
            in_ord_matriz += [pal2array(palavra_tam_fixo)]

            ixs_hifen = [i for i, a in enumerate(dividida) if a == '-']
            if len(ixs_hifen) < max_qtd_hifen:
                for i in range(max_qtd_hifen - len(ixs_hifen)):
                    ixs_hifen += [-1]
            out_ord_matriz += [ixs_hifen]


    in_ord_matriz = np.array(in_ord_matriz)
    out_ord_matriz = np.array(out_ord_matriz)

    np.save(open(IN_ORD_MATRIX_PATH, 'wb'), in_ord_matriz)
    np.save(open(OUT_ORD_MATRIX_PATH, 'wb'), out_ord_matriz)

def interagir_com_matrizes(model_path, palavra=None):
    X = in_ord_matriz = np.load(open(IN_ORD_MATRIX_PATH, 'rb'))
    Y = out_ord_matriz = np.load(open(OUT_ORD_MATRIX_PATH, 'rb'))

    X = normalize(X, axis=-1, order=2)
    Y = normalize(Y, axis=-1, order=2)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    def keras_approach(X, Y):
        config = tf.ConfigProto(intra_op_parallelism_threads=4,\
        inter_op_parallelism_threads=4, allow_soft_placement=True,\
        device_count = {'CPU' : 1, 'GPU': 0})
        session = tf.Session(config=config)
        from keras import backend as K
        K.set_session(session)
        if op.exists(model_path):
            model = load_model(model_path)
        else:
            model = Sequential()
            model.add(Dense(32, input_dim=31))
            model.add(Dense(50))
            model.add(Dense(11))
            opt = SGD(lr=0.0001)
            model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

        checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
        model.fit(X,Y, epochs=100, batch_size=25, callbacks=[checkpointer])
        scores = model.evaluate(X,Y,verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        model.save(model_path)

    def keras_predict(palavra):
        if op.exists(model_path):
            model = load_model(model_path)
            X = np.array([pal2array(pad_palavra(palavra, 31))])
            prediction = model.predict(X)
            code.interact(local=globals().update(locals()) or globals())
        else:
            print('Modelo não encontrado.') or sys.exit(1)

    if palavra:
        keras_predict(palavra)
    else:
        keras_approach(X_train, Y_train)


def main():
    parser = argparse.ArgumentParser(prog='divsil', description='divisor de sílabas genérico')
    actions = parser.add_mutually_exclusive_group(required=False)
    actions.add_argument('-t', '--treinar', metavar='<NOME_DO_MODELO>.h5', default='model.h5', help='treinar modelo (default=./modelo.h5')
    actions.add_argument('-p', '--palavra', metavar='palavra',                                 help='teste manual de uma única palavra')
    parser.add_argument( '-e', '--epochs',  metavar='N', default=50,                           help='treinar por N epochs (default=50)')
    args = parser.parse_args()

    if not op.exists(OUT_ORD_MATRIX_PATH) or not op.exists(IN_ORD_MATRIX_PATH):
        json2matriz()
    else:
        interagir_com_matrizes(args.treinar, args.palavra)

main()
