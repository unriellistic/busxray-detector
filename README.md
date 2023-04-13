# busxray-detector

A pipeline to continuously read images from a folder and run inferences (for object detection, etc.)

## Usage
Model weights can be downloaded from [detectron2's TridentNet page](https://github.com/facebookresearch/detectron2/tree/main/projects/TridentNet).

```
$ pip install -r requirements.txt
$ python main.py
```

note: `Ikomia-dev/detectron2@v0.6-win10` is used instead of the official detectron2 repository, due to a bug that causes detectron2 installation to fail on Windows.


## Customizing the predictor
The predictor can be any `Callable`. It must take an OpenCV-format image as input, and output JSON-compatible predictions.
