from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import onnxruntime as ort
import joblib
import pandas as pd
import numpy as np
import os
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

age_scaler = joblib.load("scalar/age.save")
mm_position_scaler = joblib.load("scalar/mm_position.save")
position_scaler = joblib.load("scalar/position.save")
ratings_scaler = joblib.load("scalar/ratings.save")
value_mm_scaler = joblib.load("scalar/value_mm.save")

model_session = ort.InferenceSession("model/model.onnx")

def euro_to_dollar(val):
    return 1.17 * float(val)

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def submit_form(
    request: Request,
    name: str = Form(...),
    age: int = Form(...),
    overall_rating: int = Form(...),
    shooting: int = Form(...),
    dribbling: int = Form(...),
    defending: int = Form(...),
    pace: int = Form(...),
    passing: int = Form(...),
    position: str = Form(...),
    currency: str = Form(...)
):
    try:
        data_dict = {
            'Age': age,
            'Overall Rating': overall_rating,
            'Shooting': shooting,
            "Dribbling": dribbling,
            "Defending": defending,
            "Pace": pace,
            "Passing": passing,
            "Position": position
        }
        df = pd.DataFrame([data_dict])

        df['Age'] = age_scaler.transform(np.array(df['Age']).reshape(-1, 1))
        df['Overall Rating'] = value_mm_scaler.transform(np.array(df['Overall Rating']).reshape(-1, 1))
        df['Shooting'] = value_mm_scaler.transform(np.array(df['Shooting']).reshape(-1, 1))
        df['Dribbling'] = value_mm_scaler.transform(np.array(df['Dribbling']).reshape(-1, 1))
        df['Defending'] = value_mm_scaler.transform(np.array(df['Defending']).reshape(-1, 1))
        df['Pace'] = value_mm_scaler.transform(np.array(df['Pace']).reshape(-1, 1))
        df['Passing'] = value_mm_scaler.transform(np.array(df['Passing']).reshape(-1, 1))
        df["Position"] = mm_position_scaler.transform(
            np.array(position_scaler.transform(np.array(df["Position"]).reshape(-1, 1))).reshape(-1, 1)
        )

        input_name = model_session.get_inputs()[0].name
        output = model_session.run(None, {input_name: df.values.astype(np.float32)})[0]

        euro_value = value_mm_scaler.inverse_transform(output)
        dollar_value = euro_to_dollar(euro_value)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "name_output": name,
            "estimated_value": f'{dollar_value:.2f}' if currency == "Dollar" else f'{euro_value.item():.2f}',
            "error": None
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e),
            "estimated_value": None
        })

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
