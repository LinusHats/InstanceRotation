""" This file conatins the template for the APIs. """


from fastapi import FastAPI, UploadFile, HTTPException, File, Depends
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import requests
import os
import time
from detector import inference


app = FastAPI()

class MetaData(BaseModel):
    callback_url: str
    

@app.get("/", response_class=HTMLResponse)
async def hello():
#    with open("static.html") as f:
#        content = f.read().join("\n")
    return """
    <html>
        <head>
            <title>DigiCo</title>
        <head>
        <body>
            <h1>Welcome at the DigiCo ModelAPI (FlowsheetRotation)<h1>
            <p>
                <a href="/docs">Try it yourself now!</a>
            </p>
            <p>
                <a href="https://www.pi-research.org/">Or go to our official Website</a>
            </p>
        <body>
    <html>
    """


@app.post("/tasks")
async def submit_task(metadata: MetaData = Depends(), image: UploadFile = File(...), previous_step_graphml: UploadFile = File(...)):    
    try:
        # Save the uploaded image temporarily
        with open("temp/temp_image.png", "wb") as  image_file:
            contents = await image.read()
            image_file.write(contents)
        with open("temp/temp_input_graphML.graphml", "wb") as graphML_file:
            contents = await previous_step_graphml.read()
            graphML_file.write(contents)

        # Process the image using the ML model
        result, runtime = process_ml_model(graphML_path="temp/temp_input_graphML.graphml", image_path="temp/temp_image.png")

        # Send the result to the callback URL via POST request
        print(f"callback url : {metadata.callback_url}")
        
        response = requests.post(metadata.callback_url, json={"result": result, "runtime": runtime})
        # return result, runtime
        if response.status_code == 200:
            return None  # Return 204 No Content
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def process_ml_model(graphML_path, image_path):
    start = time.time()

    g_string = inference.main(input_graphML_path=graphML_path, image_path=image_path)
    
    end = time.time()
    elapsed_time = end - start
    print("Elapsed time: ", elapsed_time, " s")
    print(g_string)
  
    return g_string, elapsed_time


# process_ml_model("./object_detection_output_05082023.graphml", "./Standard_P&ID.png")