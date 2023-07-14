import keras_ocr
import easyocr
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

description = "OCR Model"

app_config = {
    'title': 'OCR-API',
    'description': description,
    'version': '0.0.1'
}
app = FastAPI(**app_config)
pipeline = keras_ocr.pipeline.Pipeline()
reader = easyocr.Reader(['en'], gpu = True)

@app.post('/predict-keras')
async def predict(file: UploadFile = File(...)):
    image_path = "keras_image.jpg"
    with open(image_path, "wb") as image:
        image.write(await file.read())
    img = keras_ocr.tools.read(image_path)
    results = pipeline.recognize([img])
    pd.DataFrame(results[0], columns=['text', 'bbox'])
    fig, ax = plt.subplots(figsize=(15, 10))
    keras_ocr.tools.drawAnnotations(img, results[0], ax=ax)
    ax.set_title('Keras OCR Result')
    plt.savefig(image_path)
    file_image = open(image_path, mode="rb")
    return StreamingResponse(file_image, media_type="image/jpeg")

@app.post('/predict-easy')
async def predict(file: UploadFile = File(...)):
    image_path = "easy_image.jpg"
    with open(image_path, "wb") as image:
        image.write(await file.read())
    image = keras_ocr.tools.read(image_path)
    result = reader.readtext(image)
    img_df = pd.DataFrame(result, columns=['bbox','text','conf'])
    fig, ax = plt.subplots(figsize=(15, 10))
    easy_results = img_df[['text','bbox']].values.tolist()
    easy_results = [(x[0], np.array(x[1])) for x in easy_results]
    keras_ocr.tools.drawAnnotations(image, easy_results, ax=ax)
    ax.set_title('Easy OCR Result')
    plt.savefig(image_path)
    file_image = open(image_path, mode="rb")
    return StreamingResponse(file_image, media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)