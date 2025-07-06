import pickle as pk


linearRegressionModel = pk.load(open("./models/LinearRegression.model","rb"))
DecisionTreeRegressorModel = pk.load(open("./models/DecisionTreeRegressor.model","rb"))
RandomForestRegressorModel = pk.load(open("./models/RandomForestRegressor.model","rb"))

pollutants_mean = {
    "Noi":37.389323,
    "SO2i":13.799566,
    "RSPMi":70.109842,
    "SPMi":168.861038,
    "PM2_5i":48.406153
}

def AQI_Range(x):
    if x<=50:
        return "Good"
    elif x>50 and x<=100:
        return "Moderate"
    elif x>100 and x<=200:
        return "Poor"
    elif x>200 and x<=300:
        return "Unhealthy"
    elif x>300 and x<=400:
        return "Very unhealthy"
    elif x>400:
        return "Hazardous"