import json
import os
import glob
class ExperimentLoader():

    def __init__(self,path2experiment):
        self.path2experiment = path2experiment
        self.path2modelweights = path2experiment + '/model_weights/'

    def getsettings(self):
        # read json settings
        with open(self.path2experiment + '/settings.json') as json_file:
            self.settings = json.load(json_file)
        return self.settings

    def get_lowest_mse_train_weights(self):
        os.chdir(self.path2modelweights)
        for file in glob.glob("*mse*train*"):
            print(file)
        return self.path2modelweights + file
