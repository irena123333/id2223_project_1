# id2223_project The AQI prediction  of Beijing for next week

## The link of our space in hugging face:
https://huggingface.co/spaces/irena/aqi_prediction

## Objective: 
Build a model to predict the air prediction index of Beijing in the next 7 days

## Data sources:
1. We collect the historical aqi data of Beijing from World Air Quality Index.
2. We collect the historical and future weather data of Beijing from Free Weather API | Visual Crossing.

## The structure of our project:
To realize the desired functions of our project, we divided our project into 3 parts, the feature pipeline, training pipeline and inference pipeline.
### 1.Feature pipeline:
First, we create features from weather data from 2014-1-1 to 2023-1-6, and then upload the weather features to Hopsworks to create a weather feature group.
Second, we use historical aqi data from 2014-1-1 to 2023-1-6 to create an aqi feature group in hopsworks.

### 2.Training pipeline:
Before training, we combine the above two feature groups and then apply different transformation functions to different features to create the training data.
We decide to use weather features to predict the aqi of the same day, so the label is aqi.
Then we split out a test dataset from the created training data. 
Then we fit the training data into XGB regressor. The approach we use to tune the hyperparameters of the model is cross validation and grid search. Then, we evaluate the model accuracy on test data using rmse evaluation metric. The rmse of our model is 27.5. Finally, we upload our model to hopsworks.

### 3. Inference pipeline:
We get the today’s and next 6 days’ weather data from Free Weather API | Visual Crossing. Then we download the model from hopsworks to predict the aqi of Beijing for today and next 6 days. Finally we create a Streamlit user interface to live update the aqi predictions of Beijing in next week.





