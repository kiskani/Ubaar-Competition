import warnings 
import time 
import datetime
warnings.filterwarnings('ignore')
import sys
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import tensorflow as tf 

start_time = time.time()

def mean_absolute_precision_error(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

test  = pd.read_pickle('dataFrames/test_OneHotEncoding.pkl')
train = pd.read_pickle('dataFrames/train_OneHotEncoding.pkl')

continuous_cols = ['destinationLatitude', 'destinationLongitude', 'distanceKM', 'sourceLatitude',
                   'sourceLongitude', 'taxiDurationMin', 'weight', 'source', 'destination',
                   'y_gboost', 'y_xgb', 'y_bag', 'y_knn', 'y_dec', 'y_lgb' ]
categorical_cols = train.columns.drop(continuous_cols + ['ID', 'price']).tolist()

NOM = train[categorical_cols].shape[1]
renaming_dict = dict(zip(train[categorical_cols].columns, [str(x) for x in list(range(NOM)) ]))

train_renamed = train[categorical_cols].rename(columns=renaming_dict)
test_renamed  = test[categorical_cols].rename(columns=renaming_dict)

for column in continuous_cols:
    train_renamed[column] = train[column]
    test_renamed[column] = test[column]

test_renamed['ID']   = test['ID']
train_renamed['ID'] = train['ID']
test_renamed['price'] = test['price']
train_renamed['price'] = train['price']

X_train, X_val = train_test_split(train_renamed, test_size=0.2, random_state=42)

BATCH_SIZE           = int(sys.argv[1])
HIDDEN_LAYER_1_SIZE  = int(sys.argv[2]) 
HIDDEN_LAYER_2_SIZE  = int(sys.argv[3])
HIDDEN_LAYER_3_SIZE  = int(sys.argv[4])
lr                   = float(sys.argv[5])
TRAIN_EPOCHS         = int(sys.argv[7])

if sys.argv[6]=='False':
    USE_ALL_FEATURES = False
elif sys.argv[6]=='True':
    USE_ALL_FEATURES = True

print("BATCH_SIZE          =", BATCH_SIZE, "\nHIDDEN_LAYER_1_SIZE =", HIDDEN_LAYER_1_SIZE, "\nHIDDEN_LAYER_2_SIZE =",
        HIDDEN_LAYER_2_SIZE, "\nHIDDEN_LAYER_3_SIZE =", HIDDEN_LAYER_3_SIZE, "\nlr                  =" ,lr,
        "\nUSE_ALL_FEATURES    =", USE_ALL_FEATURES, "\nTRAIN_EPOCHS        =", TRAIN_EPOCHS)

print(datetime.datetime.now())

def make_model(features, labels, mode, params, config):
    input_layer = tf.feature_column.input_layer(features=features, feature_columns=feature_columns)
    global_step = tf.train.get_or_create_global_step()
    x = tf.layers.dense(inputs=input_layer, units=HIDDEN_LAYER_1_SIZE, activation=tf.nn.relu, name="first_layer")
    x = tf.layers.dropout(inputs=x,name="first_dropout")
    x = tf.layers.dense(inputs=x, units=HIDDEN_LAYER_2_SIZE, activation=tf.nn.relu, name="second_layer")
    x = tf.layers.dense(inputs=x, units=HIDDEN_LAYER_3_SIZE, activation=tf.nn.relu, name="third_layer")
    predictions = tf.contrib.layers.fully_connected(inputs=x, num_outputs=1)
    if mode == tf.estimator.ModeKeys.PREDICT :
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    elif mode == tf.estimator.ModeKeys.EVAL:
        loss  = tf.reduce_mean(tf.abs(tf.divide(predictions-labels,labels)))
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss)
    else:
        loss  = tf.reduce_mean(tf.abs(tf.divide(predictions-labels,labels)))
        tf.summary.scalar("Loss", loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op)

feature_columns = set()

if USE_ALL_FEATURES:
    for col in categorical_cols:
        col_feat = tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(renaming_dict[col], 2),2)
        feature_columns.add(col_feat)

for cont in continuous_cols:
    col_feat = tf.feature_column.numeric_column(cont)
    feature_columns.add(col_feat)

def input_fn(df, pred = False, use_all_features = USE_ALL_FEATURES):
        
    useful_fueatures = list()
    
    if use_all_features:
        for col in categorical_cols:
            useful_fueatures.append(np.array(df[renaming_dict[col]].values, dtype=np.int32))

    for cont in continuous_cols:
        useful_fueatures.append(np.array(df[cont].values, dtype=np.float32))    
    
    if pred: 
        train_number = 1
        batch_number = 1
    else:
        useful_fueatures.append(np.array(df["price"].values, dtype=np.float32))
        train_number = TRAIN_EPOCHS
        batch_number = BATCH_SIZE
        
    A = tf.train.slice_input_producer(
        tensor_list=useful_fueatures,
        num_epochs=train_number,
        shuffle= not pred,
        capacity=BATCH_SIZE * 5
    )

    dataset_dict = dict()
    
    if use_all_features:
        for i in range(len(A)):
            if i < len(categorical_cols):
                dataset_dict[renaming_dict[categorical_cols[i]]] = A[i]
            elif i < len(categorical_cols) + len(continuous_cols):
                dataset_dict[continuous_cols[i-len(categorical_cols)]] = A[i]
    else:
        for i in range(len(A)):
            if i < len(continuous_cols):
                dataset_dict[continuous_cols[i]] = A[i]
            
    if not pred:
        dataset_dict['labels'] = A[-1]
            
    batch_dict = tf.train.batch(
        dataset_dict,
        batch_number,
   )

    if pred == False:
        batch_labels = batch_dict.pop('labels')
        return batch_dict, tf.reshape(batch_labels, [-1, 1]) 
    else:
        return batch_dict 

hparams = tf.contrib.training.HParams(learning_rate=lr)
rconfig = tf.estimator.RunConfig(log_step_count_steps = 10)
estimator_val = tf.estimator.Estimator(model_fn=make_model, params=hparams, config = rconfig)
estimator_val.train(input_fn=lambda: input_fn(X_train), steps=TRAIN_EPOCHS)

predictions_val   = list(estimator_val.predict(input_fn = lambda: input_fn(X_val, pred=True)))
y_preds_val       = [int(x) for x in predictions_val]
val_score         = mean_absolute_precision_error(y_preds_val, X_val.price)
print( '%.2f' % float((time.time() - start_time)/60 ) +" mins, score= ", '%.2f' % val_score)
