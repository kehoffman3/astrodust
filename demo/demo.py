import astrodust
import pickle
import os
import sys

# Load a set of sample parameters
path = os.path.join(sys.path[0], 'rf_input_params.pkl')
file = open(path, 'rb')
params = pickle.load(file)

# Create our model class. It will download the necessary pre-trained models
model = astrodust.DustModel()

# Show the prediction
print(model.predict(**params))