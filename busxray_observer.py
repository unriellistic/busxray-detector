import os
from typing import Union, Callable
from pathlib import Path

import cv2, orjson
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent, FileMovedEvent

from utils import parse_inference

class BusxrayEventHandler(FileSystemEventHandler):
    '''
    Event handler for the busxray system.
    Performs predictions whenever a file is created, moved or modified in the input folder.
    '''
    def __init__(self, model: Callable, output_folder: Union[str, bytes, os.PathLike]):
        self.model = model
        self.output_folder = output_folder

    def on_any_event(self, event):
        """
        Reacts to events in the folder being watched.
        Doesn't do anything if the event wasn't a file being created, modified or moved, or if the file isn't an image.
        If an image file was created, modified or moved, runs predictions, and dumps the output to the output folder.
        """
        if type(event) in [FileCreatedEvent, FileModifiedEvent, FileMovedEvent]:
            file_path = Path(event.src_path)
            if file_path.suffix in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'): # it's an image, so run inference
                print("Processing image", str(file_path))
                img = cv2.imread(str(file_path))
                predictions = self.model(img) # should be COCO format (json compatible)

                # dump the predictions to the relevant filename
                output_path = Path(self.output_folder) / (file_path.stem + ".json")
                with open(output_path, "wb") as f:
                    f.write(orjson.dumps(predictions, option=orjson.OPT_INDENT_2))

                print("...done.")

class BusxrayObserver(Observer):
    '''
    Observer class for the busxray system.

    BusxrayObserver.start() will start the listening process.

    The model argument must be a function or a class that implements __call__.
    It takes a PIL.Image as input and outputs predictions in COCO format.
    '''
    def __init__(self, input_folder: Union[str, bytes, os.PathLike], output_folder: Union[str, bytes, os.PathLike], model: Callable):
        super().__init__()
        self.input_folder = input_folder
        self.handler = BusxrayEventHandler(model=model, output_folder=output_folder)
        self.schedule(self.handler, self.input_folder)