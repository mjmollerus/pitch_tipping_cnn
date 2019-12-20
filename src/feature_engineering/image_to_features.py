import sys
import cv2
import numpy as np
import keras
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from sklearn.model_selection import train_test_split
import pickle
from keras import models
from keras import layers
from keras import optimizers

def move_rename(pitcher_name):
    pitch_types = os.listdir(f'../data/{pitcher_name}/image/')
    os.mkdir(f'../data/{pitcher_name}/image/combined')
    for pitch in pitch_types:
        vid_files = os.listdir(f'../data/{pitcher_name}/image/{pitch}/')[1:]
        for vid in vid_files:
            os.rename(f'../data/{pitcher_name}/image/{pitch}/{vid}',f'../data/{pitcher_name}/image/combined/{pitch}_{vid}')

def create_df(pitcher_name,pitch_types):
    
    img_list = sorted(os.listdir(f'../data/{pitcher_name}/image/combined'))[1:]
        
    df = pd.DataFrame(img_list).rename(columns={0:'file_name'})
    
    for pitch in pitch_types:
        df[f'{pitch}'] = np.where(df.file_name.str.contains(pitch), f'{pitch}', f'not_{pitch}')
    
    return df

def split_df(df):
    train_df,test_val_df = train_test_split(df, test_size = .4)
    test_df, val_df = train_test_split(test_val_df, test_size = .5)
    
    return train_df,val_df,test_df

def extract_train_features(pitcher,sample_count,train_df,pitch_type):
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')
    batch_size = 20

    train_generator = train_datagen.flow_from_dataframe(
        dataframe = train_df,
        directory = f'../data/{pitcher}/image/combined/',
        x_col = 'file_name',
        y_col = f'{pitch_type}',
        class_mode = 'binary',
        batch_size = batch_size,
        shuffle = False
    #     target_size=(1280,720)
        )
    
    features = np.zeros(shape=(sample_count, 8, 8, 512))
    labels = np.zeros(shape=(sample_count))
    
    i = 0
    for inputs_batch, labels_batch in train_generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    
    features = np.reshape(features, (sample_count, 8*8*512))
    
    return features, labels

def extract_val_test_features(pitcher,sample_count,df,pitch_type):
    
    train_datagen = ImageDataGenerator(rescale=1./255)
    batch_size = 20

    train_generator = train_datagen.flow_from_dataframe(
        dataframe = df,
        directory = f'../data/{pitcher}/image/combined/',
        x_col = 'file_name',
        y_col = f'{pitch_type}',
        class_mode = 'binary',
        batch_size = batch_size,
        shuffle = False
    #     target_size=(1280,720)
        )
    
    features = np.zeros(shape=(sample_count, 8, 8, 512))
    labels = np.zeros(shape=(sample_count))
    
    i = 0
    for inputs_batch, labels_batch in train_generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    
    features = np.reshape(features, (sample_count, 8*8*512))

    return features, labels

if __name__ == '__main__':
    move_rename(pitcher_name)
    df = create_df(pitcher_name,pitch_types):
    train_df,val_df,test_df = split_df(df)

    conv_base = vgg16.VGG16(weights='imagenet',include_top=False)
    for pitch in pitch_types:
        train_features,train_labels = extract_train_features(pitcher,train_df.shape[0],train_df,pitch)
        val_features,val_labels = extract_val_test_features(pitcher,val_df.shape[0],val_df,pitch)
        test_features,test_labels = extract_val_test_features(pitcher,test_df.shape[0],test_df,pitch)

        pickle.dump(train_features,open('../data/{pitcher_name}/pickles/{pitch}_train_features.pickle','wb'))
        pickle.dump(train_labels,open('../data/{pitcher_name}/pickles/{pitch}_train_labels.pickle','wb'))
        pickle.dump(val_features,open('../data/{pitcher_name}/pickles/{pitch}_val_features.pickle','wb'))
        pickle.dump(val_labels,open('../data/{pitcher_name}/pickles/{pitch}_val_labels.pickle','wb'))
        pickle.dump(test_features,open('../data/{pitcher_name}/pickles/{pitch}_test_features.pickle','wb'))
        pickle.dump(test_labels,open('../data/{pitcher_name}/pickles/{pitch}_test_labels.pickle','wb'))





