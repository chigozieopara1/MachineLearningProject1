import json
import numpy as np
import pandas as pd
import joblib
import operator

# Load the model and config file
model = joblib.load("concrete_strength_pipeline.joblib")
with open("app_config.json") as f:
    config = json.load(f)


    # exact columns used in training the model
FEATURES = config['feature_order']  

print(FEATURES)
