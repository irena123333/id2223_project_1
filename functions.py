import datetime
import requests
import os
import joblib
import pandas as pd
from dateutil import parser
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np


def data_encoder_0(X):
    X.drop(columns=['date'], inplace=True)
    category_cols = ['conditions', 'aqi']
    col_names = []
    col_names = X.columns.values
    for col_name in col_names:
        if col_name not in category_cols:
            X[col_name] = StandardScaler().fit_transform(X[[col_name]])
    X['conditions'] = OrdinalEncoder().fit_transform(X[['conditions']])
    return X


def decode_features(df, feature_view):
    """Decodes features in the input DataFrame using corresponding Hopsworks Feature Store transformation functions"""
    df_res = df.copy()

    import inspect

    td_transformation_functions = feature_view._batch_scoring_server._transformation_functions

    res = {}
    for feature_name in td_transformation_functions:
        if feature_name in df_res.columns:
            td_transformation_function = td_transformation_functions[feature_name]
            sig, foobar_locals = inspect.signature(td_transformation_function.transformation_fn), locals()
            param_dict = dict(
                [(param.name, param.default) for param in sig.parameters.values() if param.default != inspect._empty])
            if td_transformation_function.name == "min_max_scaler":
                df_res[feature_name] = df_res[feature_name].map(
                    lambda x: x * (param_dict["max_value"] - param_dict["min_value"]) + param_dict["min_value"])

            elif td_transformation_function.name == "standard_scaler":
                df_res[feature_name] = df_res[feature_name].map(
                    lambda x: x * param_dict['std_dev'] + param_dict["mean"])
            elif td_transformation_function.name == "label_encoder":
                dictionary = param_dict['value_to_index']
                dictionary_ = {v: k for k, v in dictionary.items()}
                df_res[feature_name] = df_res[feature_name].map(
                    lambda x: dictionary_[x])
    return df_res


def get_model(project, model_name, evaluation_metric, sort_metrics_by):
    """Retrieve desired model or download it from the Hopsworks Model Registry.
    In second case, it will be physically downloaded to this directory"""
    TARGET_FILE = "model.pkl"
    list_of_files = [os.path.join(dirpath, filename) for dirpath, _, filenames \
                     in os.walk('.') for filename in filenames if filename == TARGET_FILE]

    if list_of_files:
        model_path = list_of_files[0]
        model = joblib.load(model_path)
    else:
        if not os.path.exists(TARGET_FILE):
            mr = project.get_model_registry()
            # get best model based on custom metrics
            model = mr.get_best_model(model_name,
                                      evaluation_metric,
                                      sort_metrics_by)
            model_dir = model.download()
            model = joblib.load(model_dir + "/model.pkl")

    return model


def get_air_json(city_name, AIR_QUALITY_API_KEY):
    return requests.get(f'https://api.waqi.info/feed/{city_name}/?token={AIR_QUALITY_API_KEY}').json()['data']


def get_air_quality_data(city_name):
    AIR_QUALITY_API_KEY = "d2e42997817782ad3e6c1a25b96b71f5a392be39"
    json = get_air_json(city_name, AIR_QUALITY_API_KEY)
    iaqi = json['iaqi']
    forecast = json['forecast']['daily']
    return [
        json['time']['s'][:10],  # Date
        max(int(forecast['pm25'][0]['avg']), int(forecast['pm10'][0]['avg'])),  # AQI
        forecast['pm10'][0]['avg'],
        forecast['pm25'][0]['avg'],
    ]


def get_weather_data_weekly(city: str, start_date: datetime) -> pd.DataFrame:
    WEATHER_API_KEY = "SH95C5HXZ3Y24J388UHAB7XB5"
    end_date = f"{start_date + datetime.timedelta(days=6):%Y-%m-%d}"
    answer = requests.get(
        f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/{start_date}/{end_date}?unitGroup=metric&include=days&key={WEATHER_API_KEY}&contentType=json').json()
    weather_data = answer['days']
    final_df = pd.DataFrame()

    for i in range(7):
        data = weather_data[i]

        list_of_data = [
            data['datetime'], data['tempmax'], data['tempmin'], data['temp'], data['feelslikemax'],
            data['feelslikemin'], data['feelslike'], data['dew'], data['humidity'], data['precip'], data['precipprob'],
            data['precipcover'],
            data['snow'], data['snowdepth'], data['windspeed'], data['winddir'], data['cloudcover'],
            data['visibility'], data['solarradiation'], data['solarenergy'], data['uvindex'], data['conditions']
        ]

        weather_df = get_weather_df(list_of_data)
        final_df = pd.concat([final_df, weather_df])
    return final_df


def get_air_quality_df(data):
    col_names = [

        'date',
        'aqi',
        'pm10_avg',
        'pm25_avg',
    ]

    new_data = pd.DataFrame(
        data,
        columns=col_names
    )
    new_data.date = new_data.date.apply(timestamp_2_time_1)

    return new_data


def get_weather_json(city, date, WEATHER_API_KEY):
    return requests.get(
        f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city.lower()}/{date}?unitGroup=metric&include=days&key={WEATHER_API_KEY}&contentType=json').json()


def get_weather_data(city_name, date):
    WEATHER_API_KEY = "SH95C5HXZ3Y24J388UHAB7XB5"
    json = get_weather_json(city_name, date, WEATHER_API_KEY)
    data = json['days'][0]

    return [

        data['datetime'],
        data['tempmax'],
        data['tempmin'],
        data['temp'],
        data['feelslikemax'],
        data['feelslikemin'],
        data['feelslike'],
        data['dew'],
        data['humidity'],
        data['precip'],
        data['precipprob'],
        data['precipcover'],
        data['snow'],
        data['snowdepth'],
        data['windspeed'],
        data['winddir'],
        data['cloudcover'],
        data['visibility'],
        data['solarradiation'],
        data['solarenergy'],
        data['uvindex'],
        data['conditions']
    ]


def get_weather_df(data):
    col_names = [

        'date',
        'tempmax',
        'tempmin',
        'temp',
        'feelslikemax',
        'feelslikemin',
        'feelslike',
        'dew',
        'humidity',
        'precip',
        'precipprob',
        'precipcover',
        'snow',
        'snowdepth',
        'windspeed',
        'winddir',
        'cloudcover',
        'visibility',
        'solarradiation',
        'solarenergy',
        'uvindex',
        'conditions'
    ]

    new_data = pd.DataFrame(
        data
    ).T
    new_data.columns = col_names
    new_data.date = new_data.date.apply(timestamp_2_time_1)
    for col in col_names:
        if col not in ['date', 'conditions']:
            new_data[col] = pd.to_numeric(new_data[col])

    return new_data


def timestamp_2_time(x):
    dt_obj = parser.parse(str(x))
    dt_obj = dt_obj.timestamp() * 1000
    return int(dt_obj)


def timestamp_2_time_1(x):
    dt_obj = parser.parse(str(x))
    dt_obj = dt_obj.timestamp() * 1000
    return int(dt_obj)


def encoder_range(temps):
    boundary_list = np.array([0, 50, 100, 150, 200, 300])
    redf = np.logical_not(temps <= boundary_list)
    hift = np.concatenate((np.roll(redf, -1)[:, :-1], np.full((temps.shape[0], 1), False)), axis=1)
    cat = np.nonzero(np.not_equal(redf, hift))
    air_pollution_level = ['Good', 'Moderate', 'Unhealthy for sensitive Groups', 'Unhealthy', 'Very Unhealthy',
                           'Hazardous']
    level = [air_pollution_level[el] for el in cat[1]]
    return level