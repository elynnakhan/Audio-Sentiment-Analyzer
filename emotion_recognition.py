from data_extractor import load_data
from utils import extract_feature, AVAILABLE_EMOTIONS
from create_csv import write_emodb_csv, write_tess_ravdess_csv, write_custom_csv

from sklearn.metrics import accuracy_score, make_scorer, fbeta_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as pl
from time import time
from utils import get_audio_config
import numpy as np
import tqdm
import os
import random
import pandas as pd


class EmotionRecognizer:
    """A class for training, testing and predicting emotions based on
    speech's features that are extracted and fed into `sklearn` or `keras` model"""
    def __init__(self, model, **kwargs):
        """
        Params:
            model (sklearn model): the model used to detect emotions.
            emotions (list): list of emotions to be used. Note that these emotions must be available in
                RAVDESS_TESS & EMODB Datasets, available nine emotions are the following:
                    'neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps' ( pleasant surprised ), 'boredom'.
                Default is ["sad", "neutral", "happy"].
            tess_ravdess (bool): whether to use TESS & RAVDESS Speech datasets, default is True
            emodb (bool): whether to use EMO-DB Speech dataset, default is True,
            custom_db (bool): whether to use custom Speech dataset that is located in `data/train-custom`
                and `data/test-custom`, default is True
            tess_ravdess_name (str): the name of the output CSV file for TESS&RAVDESS dataset, default is "tess_ravdess.csv"
            emodb_name (str): the name of the output CSV file for EMO-DB dataset, default is "emodb.csv"
            custom_db_name (str): the name of the output CSV file for the custom dataset, default is "custom.csv"
            features (list): list of speech features to use, default is ["mfcc", "chroma", "mel"]
                (i.e MFCC, Chroma and MEL spectrogram )
            classification (bool): whether to use classification or regression, default is True
            balance (bool): whether to balance the dataset ( both training and testing ), default is True
            verbose (bool/int): whether to print messages on certain tasks, default is 1
        Note that when `tess_ravdess`, `emodb` and `custom_db` are set to `False`, `tess_ravdess` will be set to True
        automatically.
        """
        # model
        self.model = model
        # emotions
        self.emotions = kwargs.get("emotions", ["sad", "neutral", "happy"])
        # make sure that there are only available emotions
        self._verify_emotions()
        # audio config
        self.features = kwargs.get("features", ["mfcc", "chroma", "mel"])
        self.audio_config = get_audio_config(self.features)
        # datasets
        self.tess_ravdess = kwargs.get("tess_ravdess", True)
        self.emodb = kwargs.get("emodb", True)
        self.custom_db = kwargs.get("custom_db", True)

        if not self.tess_ravdess and not self.emodb and not self.custom_db:
            self.tess_ravdess = True
    
        self.classification = kwargs.get("classification", True)
        self.balance = kwargs.get("balance", True)
        self.override_csv = kwargs.get("override_csv", True)
        self.verbose = kwargs.get("verbose", 1)

        self.tess_ravdess_name = kwargs.get("tess_ravdess_name", "tess_ravdess.csv")
        self.emodb_name = kwargs.get("emodb_name", "emodb.csv")
        self.custom_db_name = kwargs.get("custom_db_name", "custom.csv")

        self.verbose = kwargs.get("verbose", 1)

        # set metadata path file names
        self._set_metadata_filenames()
        # write csv's anyway
        self.write_csv()

        # boolean attributes
        self.data_loaded = False
        self.model_trained = False

    def _set_metadata_filenames(self):
        """
        Protected method to get all CSV (metadata) filenames into two instance attributes:
        - `self.train_desc_files` for training CSVs
        - `self.test_desc_files` for testing CSVs
        """
        train_desc_files, test_desc_files = [], []
        if self.tess_ravdess:
            train_desc_files.append(f"train_{self.tess_ravdess_name}")
            test_desc_files.append(f"test_{self.tess_ravdess_name}")
        if self.emodb:
            train_desc_files.append(f"train_{self.emodb_name}")
            test_desc_files.append(f"test_{self.emodb_name}")
        if self.custom_db:
            train_desc_files.append(f"train_{self.custom_db_name}")
            test_desc_files.append(f"test_{self.custom_db_name}")

        # set them to be object attributes
        self.train_desc_files = train_desc_files
        self.test_desc_files  = test_desc_files

    def _verify_emotions(self):
        """
        This method makes sure that emotions passed in parameters are valid.
        """
        for emotion in self.emotions:
            assert emotion in AVAILABLE_EMOTIONS, "Emotion not recognized."


    def write_csv(self):
        """
        Write available CSV files in `self.train_desc_files` and `self.test_desc_files`
        determined by `self._set_metadata_filenames()` method.
        """
        for train_csv_file, test_csv_file in zip(self.train_desc_files, self.test_desc_files):
            # not safe approach
            if os.path.isfile(train_csv_file) and os.path.isfile(test_csv_file):
                # file already exists, just skip writing csv files
                if not self.override_csv:
                    continue
            if self.emodb_name in train_csv_file:
                write_emodb_csv(self.emotions, train_name=train_csv_file, test_name=test_csv_file, verbose=self.verbose)
                if self.verbose:
                    print("[+] Writed EMO-DB CSV File")
            elif self.tess_ravdess_name in train_csv_file:
                write_tess_ravdess_csv(self.emotions, train_name=train_csv_file, test_name=test_csv_file, verbose=self.verbose)
                if self.verbose:
                    print("[+] Writed TESS & RAVDESS DB CSV File")
            elif self.custom_db_name in train_csv_file:
                write_custom_csv(emotions=self.emotions, train_name=train_csv_file, test_name=test_csv_file, verbose=self.verbose)
                if self.verbose:
                    print("[+] Writed Custom DB CSV File")

    def load_data(self):
        """
        Loads and extracts features from the audio files for the db's specified
        """
        if not self.data_loaded:
            result = load_data(self.train_desc_files, self.test_desc_files, self.audio_config, self.classification,
                                emotions=self.emotions, balance=self.balance)
            self.X_train = result['X_train']
            self.X_test = result['X_test']
            self.y_train = result['y_train']
            self.y_test = result['y_test']
            self.train_audio_paths = result['train_audio_paths']
            self.test_audio_paths = result['test_audio_paths']
            self.balance = result["balance"]
            if self.verbose:
                print("[+] Data loaded")
            self.data_loaded = True

    def train(self, verbose=1):
        """
        Train the model, if data isn't loaded, it 'll be loaded automatically
        """
        if not self.data_loaded:
            # if data isn't loaded yet, load it then
            self.load_data()
        if not self.model_trained:
            self.model.fit(X=self.X_train, y=self.y_train)
            self.model_trained = True
            if verbose:
                print("[+] Model trained")

    def predict(self, audio_path):
        """
        given an `audio_path`, this method extracts the features
        and predicts the emotion
        """
        feature = extract_feature(audio_path, **self.audio_config).reshape(1, -1)
        return self.model.predict(feature)[0]

    def predict_proba(self, audio_path):
        """
        Predicts the probability of each emotion.
        """
        if self.classification:
            feature = extract_feature(audio_path, **self.audio_config).reshape(1, -1)
            proba = self.model.predict_proba(feature)[0]
            result = {}
            for emotion, prob in zip(self.model.classes_, proba):
                result[emotion] = prob
            return result
        else:
            raise NotImplementedError("Probability prediction doesn't make sense for regression")

   

    


