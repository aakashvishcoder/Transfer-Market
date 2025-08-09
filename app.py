from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
import re

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class NeuralNetwork(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.l1 = nn.Linear(input_size,50)
        self.l2 = nn.Linear(50,25)
        self.l3 = nn.Linear(25,1)
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        return self.l3(out)

model = NeuralNetwork(8)
model.load_state_dict(torch.load("model/model_state.pth"))
model.eval()

age_scaler = joblib.load("scalar/age.save")
mm_position_scaler = joblib.load("scalar/mm_position.save")
position_scaler = joblib.load("scalar/position.save")
ratings_scaler = joblib.load("scalar/ratings.save")
value_mm_scaler = joblib.load("scalar/value_mm.save")

def euro_to_dollar(val):
    return 1.17 * float(val)

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request":request})

@app.post('/',response_class=HTMLResponse)
async def submit_form(
    request:Request,
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
        dict = {
            'Age':age,
            'Overall Rating':overall_rating,
            'Shooting': shooting,
            "Dribbling":dribbling,
            "Defending":defending,
            "Pace":pace,
            "Passing":passing,
            "Position":position
        }
        df = pd.DataFrame([dict])
        df['Age'] = age_scaler.transform(np.array(df['Age']).reshape(-1,1))
        df['Overall Rating'] = value_mm_scaler.transform(np.array(df['Overall Rating']).reshape(-1,1))
        df['Shooting'] = value_mm_scaler.transform(np.array(df['Shooting']).reshape(-1,1))
        df['Dribbling'] = value_mm_scaler.transform(np.array(df['Dribbling']).reshape(-1,1))
        df['Defending'] = value_mm_scaler.transform(np.array(df['Defending']).reshape(-1,1))
        df['Pace'] = value_mm_scaler.transform(np.array(df['Pace']).reshape(-1,1))
        df['Passing'] = value_mm_scaler.transform(np.array(df['Passing']).reshape(-1,1))
        df["Position"] = mm_position_scaler.transform(np.array(position_scaler.transform(np.array(df["Position"]).reshape(-1,1))).reshape(-1,1))
        x = torch.tensor(df.values,dtype=torch.float32)
        output = model(x).detach().numpy()
        euro_value = value_mm_scaler.inverse_transform(output)
        dollar_value = euro_to_dollar(euro_value)

        return templates.TemplateResponse("index.html",{
            "request":request,
            "name":name,
            "overall_rating":overall_rating,
            "shooting":shooting,
            "dribbling":dribbling,
            "defending":defending,
            "pace":pace,
            "passing":passing,
            "position":position,
            "currency":currency,
            "estimated_value":dollar_value if currency=="Dollar" else euro_value.item(),
            "error":None
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request":request,
            "error":str(e),
            "estimated_value":None
        })
