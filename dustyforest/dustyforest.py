import numpy as np
from joblib import load
import xgboost as xgb
from tqdm import *
import requests
import sklearn
import os
import math
import warnings


class DustModel:

    URL_QUALITY_MODEL = "https://dataverse.lib.virginia.edu/api/access/datafile/35742"
    URL_RF_MODEL = "https://dataverse.lib.virginia.edu/api/access/datafile/35746"

    FILENAME_QUALITY_MODEL = "models/prediction-quality.model"
    FILENAME_RF_MODEL = "models/rf-model-large.joblib"

    
    def __init__(self):
        """ Loads a random forest model for predicting dust densities and XGBoost model for predicting quality of the predicted dust densities."""
        self.quality_model = xgb.XGBClassifier()


        model_dir = 'models'
        # Check if our model folder exists, create it if not
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # First try to load the quality model from disk
        try:
            self.quality_model.load_model(self.FILENAME_QUALITY_MODEL)
        except xgb.core.XGBoostError:
            # Try to download the quality model if the user does not have it
            try:
                self._download_file(self.URL_QUALITY_MODEL, self.FILENAME_QUALITY_MODEL)
                self.quality_model.load_model(self.FILENAME_QUALITY_MODEL)
            except FileNotFoundError as e:
                e.strerror = "The model file must exist in the model directory."
                raise e
        # Try to load the random forest model from disk
        try:
            self.model = load(self.FILENAME_RF_MODEL)
        except:
            try:
                self._download_file(self.URL_RF_MODEL, self.FILENAME_RF_MODEL)
                self.model = load(self.FILENAME_RF_MODEL)
            except FileNotFoundError as e:
                e.strerror = "The model file must exist in the model directory."
                raise e

    def _download_file(self, url, name):
        """ Downloads a given file and writes to disk. Also displays a progress bar. From: https://stackoverflow.com/a/56796119"""
        # From https://towardsdatascience.com/how-to-download-files-using-python-part-2-19b95be4cdb5
        r = requests.get(url, stream=True, allow_redirects=True)
        total_size = int(r.headers.get('content-length'))
        initial_pos = 0
        with open(name,'wb') as f: 
            with tqdm(total=total_size, unit='B', unit_scale=True,  desc=name,initial=initial_pos, ascii=True) as pbar:
                for ch in r.iter_content(chunk_size=(1024 * 128)): # 128KB chunk size                    
                    if ch:
                        f.write(ch) 
                        pbar.update(len(ch))

    def predict(self, r, mstar, alpha, d2g, sigma, tgas, t, delta_t, input_bins):
        """
        Makes a prediction from the given set of input parameters. It will also print a warning if the predicted quality is not good.
        
        r (float): distance from central star
        mstar (float): mass of star in the system
        alpha (float): turblance
        d2g (float): dust to gas ratio
        sigma (float): the surface density of the gas in the model (in g/cm^2)
        tgas (float): temperature of the gas
        t (int): absolute time in seconds
        delta_t (int): time in seconds in the future to predict for
        input_bins (array length 171): 171 length array of dust densities

        Returns (array length 171): 171 length array of the predicted dust densities
        """


        input_params = [r, mstar, alpha, d2g, sigma, tgas]
        input_bins_sum = np.sum(input_bins)
        #Normalize the input bins and zero out normalized bins less than 10^-30\n",
        normalized_input_bins = []
        for i in range(len(input_bins)):
            new_input_bin = np.sum(input_bins[i]) / input_bins_sum
            if new_input_bin < 1e-30:
                new_input_bin = 0
            normalized_input_bins.append(new_input_bin)

        # Input features
        X = np.concatenate([input_params,normalized_input_bins,[t, delta_t]])

        X_formatted = X.reshape(1, -1)
        prediction = self.model.predict(X_formatted)
        
        # Unnormalize the prediction by multiplying each bin by the input mass
        prediction *= input_bins_sum

        # Check the quality of the prediction
        prediction_quality = self.quality_model.predict(X_formatted)
        if prediction_quality == 0:
            warnings.warn("For the given input paramters, the resulting prediction is most likely not accurate.")
        return prediction.flatten()