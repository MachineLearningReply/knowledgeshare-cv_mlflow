# knowledgeshare-cv_mlflow

## Installation
Use python 3.11. 

Optionally create venv and install dependencies
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Download the dataset by executing 
```
python3 download_data.py
```

## Training
to start the mlflow server, do:
```
mlflow ui
```
then you can access the local server at the provided link (if local then http://127.0.0.1:5000)

Similarly, for the tensorboard UI, do:
```
tensorboard --logdir=tensorboard_logs --port=6006
````
and you shall be able to access the ui at http://localhost:6006/


to start the training do:
```
python src/train.py
```

## Inference
to start the inference, do:
```
python src/predict.py
```
