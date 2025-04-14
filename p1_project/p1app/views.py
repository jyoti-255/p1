import os
import pickle
import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import login, logout, authenticate
from django.contrib import messages
from django.conf import settings

# ---------------------- User Authentication Views ----------------------

def uhome(request):
    return render(request, "home.html")

def ubase(request):
    return render(request, "base.html")

def usignup(request):
    if request.user.is_authenticated:
        return redirect("ubase")

    if request.GET:
        un = request.GET.get("un", "").strip()
        pw1 = request.GET.get("pw1", "").strip()
        pw2 = request.GET.get("pw2", "").strip()

        if not un or not pw1 or not pw2:
            msg = "All fields are required."
            return render(request, "signup.html", {"msg": msg})

        if pw1 != pw2:
            msg = "Passwords did not match."
            return render(request, "signup.html", {"msg": msg})

        if User.objects.filter(username=un).exists():
            msg = f"{un} is already registered."
            return render(request, "signup.html", {"msg": msg})

        user = User.objects.create_user(username=un, password=pw1)
        login(request, user)
        return redirect("ubase")

    return render(request, "signup.html")

def ulogin(request):
    if request.user.is_authenticated:
        return redirect("ubase")

    if request.GET:
        username = request.GET.get("username", "").strip()
        password = request.GET.get("password", "").strip()

        if not username or not password:
            messages.error(request, "Username and password are required.")
            return render(request, "login.html")

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("ubase")
        else:
            messages.error(request, "Invalid username or password.")

    return render(request, "login.html")

def ulogout(request):
    logout(request)
    return render(request, "logout.html")

def dashboard(request):
    if not request.user.is_authenticated:
        return redirect("ulogin")
    return render(request, "dashboard.html", {"user": request.user})

# ---------------------- Crop Prediction ----------------------

def load_pickle_file(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"⚠️ File not found: {file_path}")
        return None

model_path1 = os.path.join(settings.BASE_DIR, 'p1app', 'models1', 'crop_prediction_model.pkl')
scaler_path1 = os.path.join(settings.BASE_DIR, 'p1app', 'models1', 'scaler.pkl')
label_encoders_path1 = os.path.join(settings.BASE_DIR, 'p1app', 'models1', 'label_encoders.pkl')

model = load_pickle_file(model_path1)
scaler1 = load_pickle_file(scaler_path1)
label_encoders = load_pickle_file(label_encoders_path1)

def crop(request):
    prediction_result = ""
    error_message = ""

    if request.GET:
        try:
            rainfall = float(request.GET.get("rainfall"))
            temperature = float(request.GET.get("temperature"))
            humidity = float(request.GET.get("humidity"))
            soil_type = request.GET.get("soil_type")
            irrigation_type = request.GET.get("irrigation_type")
            soil_pH = float(request.GET.get("soil_pH"))

            if soil_type not in label_encoders["Soil Type"].classes_ or irrigation_type not in label_encoders["Irrigation Type"].classes_:
                error_message = "Invalid soil type or irrigation type"
                return render(request, "crop.html", {"error": error_message})

            soil_type_encoded = label_encoders["Soil Type"].transform([soil_type])[0]
            irrigation_type_encoded = label_encoders["Irrigation Type"].transform([irrigation_type])[0]

            input_data = np.array([[rainfall, temperature, humidity, soil_type_encoded, irrigation_type_encoded, soil_pH]])

            if input_data.shape[1] != scaler1.n_features_in_:
                error_message = f"Feature mismatch! Model expects {scaler1.n_features_in_} features, but received {input_data.shape[1]}."
                return render(request, "crop.html", {"error": error_message})

            input_scaled = scaler1.transform(input_data)
            prediction = model.predict(input_scaled)
            crop_predicted = label_encoders["Crop Type"].inverse_transform(prediction)[0]

            prediction_result = f"Predicted Crop: {crop_predicted}"

        except Exception as e:
            error_message = f"Error: {str(e)}"
            return render(request, "crop.html", {"error": error_message})

    return render(request, "crop.html", {"predicted_crop": prediction_result, "error": error_message})

# ---------------------- Water Prediction ----------------------

model_path = os.path.join(settings.BASE_DIR, 'p1app', 'models', 'final_release_model.pkl')
scaler_path = os.path.join(settings.BASE_DIR, 'p1app', 'models', 'scaler(1).pkl')
stack_model_path = os.path.join(settings.BASE_DIR, 'p1app', 'models', 'stack_models.pkl')
inflow_model_path = os.path.join(settings.BASE_DIR, 'p1app', 'models', 'inflow_model.pkl')

stack_models = joblib.load(stack_model_path)
inflow_model = joblib.load(inflow_model_path)
final_release_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

month_mapping = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

def prediction(request):
    error = None
    predicted_bmc_tmc = None
    predicted_irrigation = None
    selected_month = None
    start_water_level = None
    start_gross_storage = None

    if request.GET:
        try:
            month_num = int(request.GET.get("month"))
            start_water_level = float(request.GET.get("start_water_level"))
            start_gross_storage = float(request.GET.get("start_gross_storage"))

            selected_month = month_mapping[month_num]
            month_name = f"Month_{selected_month}"

            input_data = pd.DataFrame([[start_water_level, start_gross_storage]],
                                      columns=["Reservoir water level on start of month (m)",
                                               "Effective gross storage on start on month (Mcum)"])

            all_months = [f"Month_{abbr}" for abbr in month_mapping.values()]
            month_df = pd.DataFrame([[1 if col == month_name else 0 for col in all_months]], columns=all_months)

            input_data = pd.concat([input_data, month_df], axis=1)

            missing_cols = set(scaler.feature_names_in_) - set(input_data.columns)
            for col in missing_cols:
                input_data[col] = 0

            input_data = input_data[scaler.feature_names_in_]
            input_scaled = scaler.transform(input_data)

            predicted_values = [model.predict(input_scaled)[0] for model in stack_models]
            predicted_inflow = inflow_model.predict(input_scaled)[0]
            predicted_release = final_release_model.predict(input_scaled)[0]

            if month_num in [6, 7, 8, 9]:
                predicted_release[1] = 0

            predicted_bmc_tmc = round(predicted_release[0], 2)
            predicted_irrigation = round(predicted_release[1], 2)

        except Exception as e:
            error = str(e)

    return render(request, "prediction.html", {
        "month_mapping": month_mapping,
        "error": error,
        "predicted_bmc_tmc": predicted_bmc_tmc,
        "predicted_irrigation": predicted_irrigation,
        "month": selected_month,
        "start_water_level": start_water_level,
        "start_gross_storage": start_gross_storage
    })
