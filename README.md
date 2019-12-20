## Detecting Pitch Tipping

In baseball, a pitcher is said to be 'tipping' their pitches if they accidentally telegraph the type of pitch they are about to throw with their body language. Occasionally these tells are so obvious humans pick up on it with the naked eye. This repo contains a system of models that identify more subtle pitch tipping and help humans pick up on it.

The data comes from MLB's Statcast website, which has video every pitch from the past few seasons tagged by pitcher, pitch type, and other factors. I use Selenium and BeautifulSoup to extract the videos for the pitchers I want to analyze, then use openCV to capture a frame of the pitcher in their set position and to filter out unsuable data.

I then train a neural network with a VGG16 convolutional base for each of the pitcher's pitches. One model tries to pick out a fastball against all other pitches, another a curveball against all other pitches, and so on.

Most model iterations don't pick out their target with any great accuracy, which shouldn't be surprising; most pitchers aren't tipping their pitches all the time. However, frequently they do succeed at differentiating pitches. Scouts can then examine these cases to find what is triggering the neural network and see if that body language is significant enough for hitters to use.

# Data

# Notebooks

The notebooks contain several examples of how to use this system of models, from processing the data to fine-tuning the model to visually identifying usual information from the output and the images.

# Code

The src folder contains an template of the pipeline. The modeling step requires fine-tuning the hyperparameters for the densely connected layers for each pitch model, but the hyperparameters as currently set generally produce a very good result.

# Deck

The deck contains a presentation given for Metis's Fall 2019 Career Night.
