# encoding: utf-8
"""Inference script for Sentinel-2 land use classification."""
import os
import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from helpers import NpuHelperForTF, SentinelHelper
from utils import lazyproperty

# --------------------------- GENERAL CONFIG ------------------------------------------
DIR = os.path.dirname(os.path.realpath(__file__))


class Inferer:
    """Implementation of the inference running procedure."""

    model_path = f"{DIR}/data/models/vgg_ms_transfer_final.44-0.969.hdf5"

    def __init__(self):
        self._bbox = list(map(float, sys.argv[-1].split(",")))
        self._sess = NpuHelperForTF().sess

    def run(self):
        """Execute inference procedure."""

        # ------------------------ LOAD IMAGE FROM SENTINEL ---------------------------
        sh = SentinelHelper(self._bbox)
        in_shp = self._model.layers[0].input_shape[0][1:3]
        gen, steps = sh.input_generator(in_shp), sh.image.shape[0]

        # ------------------------ GENERATING PREDICTIONS -----------------------------
        rows_classified = self._model.predict(gen, steps=steps, verbose=1)
        self._sess.close()  # close NPU session right after training

        # ------------------------ WRITING RESULT IMAGES ------------------------------
        sh.write_result(rows_classified, "fused.png", f"classes{self._bbox}.tiff")

    @lazyproperty
    def _model(self):
        """TF model loaded from the predefined path."""
        return load_model(self.model_path)


if __name__ == "__main__":
    Inferer().run()
