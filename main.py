import os

from busxray_observer import BusxrayObserver
from tridentnet_predictor import TridentNetPredictor

if __name__ == "__main__":
    # change this if you want to use a different model
    # predictor should be callable, take openCV-format image as input, and output JSON-compatible predictions
    predictor = TridentNetPredictor(config_file="models/tridentnet_fast_R_50_C4_3x.yaml",
        opts=["MODEL.WEIGHTS", "models/model_final_e1027c.pkl"]
    )

    observer = BusxrayObserver("input", "output", predictor)
    observer.start()

    os.system("pause")