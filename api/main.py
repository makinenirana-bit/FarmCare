from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../saved_models/1")
CLASS_NAMES = [
    'Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight',
    'Potato healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold',
    'Tomato Septoria leaf spot', 'Tomato Spider mites Two spotted spider mite', 'Tomato Target Spot',
    'Tomato YellowLeaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy'
]

CLASS_PRECAUTIONS = {
    'pepper_bell_bacterial_spot': ['1:  Clean tools and equipment regularly to prevent the transfer of bacteria.', '2: Regularly inspect your tomato plants for symptoms and take action if you notice any signs of infection.', '3: Pesticides: Copper-based Sprays, Agri-Mycin 50'],
    'pepper_bell_healthy': ['Your plant is healthy!'],
    'potato_early_blight': ['1:Early blight can be minimized by maintaining optimum growing conditions, including proper fertilization, irrigation, and management of other pests. ', '2:Grow later maturing, longer season varieties.', '3:Pestisides: Liquid Copper Spray, Garden dust'],
    'potato_late_blight': ['1: Late blight is controlled by eliminating cull piles and volunteer potatoes, using proper harvesting, and applying fungicides when necessary.' ,'2: Air drainage to facilitate the drying of foliage each day is important.' ,'3:Pesticides:Serenade Garden,Liquid Copper'],
    'potato_healthy': ['Your plant is healthy!'],
    'tomato_bacterial_spot': ['1:  Clean tools and equipment regularly to prevent the transfer of bacteria.', '2: Regularly inspect your tomato plants for symptoms and take action if you notice any signs of infection.', '3: Pesticides: Copper-based Sprays, Agri-Mycin 50'],
    'tomato_early_blight': ['1: maintain optimum growing conditions', '2: include proper fertilization, irrigation, and management of other pests'],
    'tomato_late_blight': ['1: Adequate spacing between plants allows for better air circulation, reducing humidity and the spread of the disease.', '2: Watering at the base of plants helps keep foliage dry and reduces conditions favorable for disease development.', '3: Pesticides: Presidio 4FL, Orondis Opti'],
    'tomato_leaf_mold': ['1: Isolate new plants for a period before introducing them to your garden to ensure they are not carrying the disease.', '2: Regularly inspect your tomato plants for signs of leaf mold, especially on the undersides of leaves.', '3: Pestisides: chlorothalonil, maneb'],
    'tomato_septoria_leaf_spot': ['1: Remove and destroy infected plant debris and fallen leaves to reduce the overwintering of the fungus.', '2: Avoid overcrowding plants, as this can lead to poor air circulation and higher humidity levels.', '3: Pesticides: Quadris, Fontelis'],
    'tomato_spider_mites_two_spotted_spider_mite': ['1: Proper watering, fertilization, and overall plant care can help minimize mite problems.', '2: Crowded plants are more prone to mite infestations due to reduced air circulation', '3: Pesticides: Cyfluthrin, Kelthane.'],
    'tomato_target_spot': ['1: Do not plant new crops next to older ones that have the disease','2: Plant as far as possible from papaya, especially if leaves have small angular spots','3: Pestisides: chlorothalonil, mancozeb'],
    'tomato_yellowleaf_curl_virus': ['1: Plant early to avoid peak populations of the whitefly', '2: Use nets to cover seedbeeds and prevent whiteflies from reaching the plats',],
    'tomato_mosaic_virus': ['1: Buying transplants from reputable sources', '2: Avoiding contact with infected plants', '3: Pesticides: Harvest-guard, Safer Soap'],
    'tomato_healthy': ['Your plant is healthy!']
}


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    cleaned_class = predicted_class.lower().replace(" ", "_")

    if confidence > 0.5 and cleaned_class in CLASS_PRECAUTIONS:
        precautions = CLASS_PRECAUTIONS[cleaned_class]
    else:
        precautions = []

    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'precautions': precautions
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
