from __future__ import annotations
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROWS,COLS,PLANES,ACTIONS=5,9,7,1080

def residual(x, filters, name):
    y=layers.Conv2D(filters,3,padding='same',use_bias=False,name=name+'_c1')(x)
    y=layers.BatchNormalization(name=name+'_bn1')(y); y=layers.ReLU()(y)
    y=layers.Conv2D(filters,3,padding='same',use_bias=False,name=name+'_c2')(y)
    y=layers.BatchNormalization(name=name+'_bn2')(y)
    y=layers.Add()([x,y]); return layers.ReLU()(y)

def build_model(filters=128, blocks=12):
    inp=keras.Input((ROWS,COLS,PLANES),name='board')
    x=layers.Conv2D(filters,3,padding='same',use_bias=False)(inp)
    x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    for i in range(blocks): x=residual(x,filters,f'res{i:02d}')
    p=layers.Conv2D(32,1,padding='same',activation='relu',use_bias=False,name='policy_conv')(x)
    p=layers.Flatten()(p); p=layers.Dense(ACTIONS,name='policy')(p)
    v=layers.Conv2D(32,1,padding='same',activation='relu',use_bias=False,name='value_conv')(x)
    v=layers.Flatten()(v); v=layers.Dense(128,activation='relu')(v); v=layers.Dense(1,activation='tanh',name='value')(v)
    return keras.Model(inp,[p,v],name='FanoronaStrongResNet')
