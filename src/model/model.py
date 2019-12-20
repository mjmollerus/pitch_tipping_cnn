import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from keras import models
from keras import layers
from keras import optimizers
from collections import defaultdict

def load_features(pitcher_name,pitch_types):
	data_dict = defaultdict()
	for pitch in pitch_types:
		data_dict[f'{pitch}_train_features'] = pickle.load(open(f'../data/{pitcher_name}/pickles/{pitch}_train_features.pickle','rb'))
		data_dict[f'{pitch}_train_labels'] = pickle.load(open(f'../data/{pitcher_name}/pickles/{pitch}_train_labels.pickle','rb'))
		data_dict[f'{pitch}_val_features'] = pickle.load(open(f'../data/{pitcher_name}/pickles/{pitch}_val_features.pickle','rb'))
		data_dict[f'{pitch}_val_labels'] = pickle.load(open(f'../data/{pitcher_name}/pickles/{pitch}_val_labels.pickle','rb'))
		data_dict[f'{pitch}_test_features'] = pickle.load(open(f'../data/{pitcher_name}/pickles/{pitch}_test_features.pickle','rb'))
		data_dict[f'{pitch}_test_labels'] = pickle.load(open(f'../data/{pitcher_name}/pickles/{pitch}_test_labels.pickle','rb'))

	return data_dict

def train_model(pitcher_name,pitch):
	model = models.Sequential()
	model.add(layers.Dense(256, activation='relu', input_dim=8 * 8 * 512))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(128, activation='relu', input_dim=8 * 8 * 512))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(1, activation='sigmoid'))

	model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
	              loss='binary_crossentropy',
	              metrics=['acc'])

	model.fit(data_dict[f'{pitch}_train_features'], data_dict[f'{pitch}_train_labels'],
                    epochs=30,
                    batch_size=50,
                    validation_data=(data_dict[f'{pitch}_val_features'], data_dict[f'{pitch}_val_labels']))

	model.save(f'../models/{pitcher_name}/{pitch}_model.h5')

if __name__ == '__main__':
	data_dict = load_features(pitcher_name,pitch_types)
	for pitch in pitch_types:
		print(pitch)
		train_model(pitcher_name,pitch)

