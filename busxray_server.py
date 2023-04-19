import aiofiles, time
from pathlib import Path

import cv2, orjson
from sanic import Sanic
from sanic.exceptions import SanicException
from sanic.log import logger
from sanic.response import json
from sanic.worker.loader import AppLoader

from tridentnet_predictor import TridentNetPredictor

# change this if you want to use a different model
# predictor should be callable, take openCV-format image as input, and output JSON-compatible predictions
predictor = TridentNetPredictor(config_file="models/tridentnet_fast_R_50_C4_3x.yaml",
    opts=["MODEL.WEIGHTS", "models/model_final_e1027c.pkl"]
)

app = Sanic("busxray_detector", dumps=orjson.dumps)
app.update_config("./sanic_config.py")

@app.post("/")
async def upload(request):
    img = request.files.get("img")
    # error if the file isn't an image
    if Path(img.name).suffix not in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'):
        raise SanicException("Invalid file type. Accepted file types are: .png, .jpg, .jpeg, .tiff, .bmp, .gif", status_code=415)
    
    # save the image to disk
    img_path = Path(app.config.INPUT_FOLDER) / img.name
    async with aiofiles.open(img_path, "wb") as f:
        await f.write(img.body)
    
    # run AI prediction
    logger.info("Processing image " + str(img_path))
    start_time = time.perf_counter()
    img_cv2 = cv2.imread(str(img_path))
    predictions = predictor(img_cv2) # should be COCO format (json compatible)

    # save the prediction to json file (not needed as this is done on client side)
    # output_path = Path(app.config.OUTPUT_FOLDER) / Path(img.name).with_suffix(".json")
    # with open(output_path, "wb") as f:
    #     f.write(orjson.dumps(predictions, option=orjson.OPT_INDENT_2))
    logger.info(f"...done. ({(time.perf_counter() - start_time):.3f} s)")

    return json(predictions)
