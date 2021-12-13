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

    @lazyproperty
    def _model(self):
        return load_model(self.model_path)

    def run(self, bbox, quartal, year, npu_config):
        """Execute inference procedure."""
        # ------------------------ INIT NPU BEFORE LOADING MODEL ----------------------
        sess = NpuHelperForTF(**npu_config).sess
        model = self._model
        in_shp = model.layers[0].input_shape[0][1:3]

        # ------------------------ LOAD IMAGE FROM SENTINEL ---------------------------
        sh = SentinelHelper(bbox, quartal, year)
        gen, steps = sh.input_generator(in_shp), sh.image.shape[0]

        # ------------------------ GENERATING PREDICTIONS -----------------------------
        rows_classified = model.predict(gen, steps=steps, verbose=1)
        sess.close()  # close NPU session right after inference

        # ------------------------ WRITING RESULT IMAGES ------------------------------
        bbox_str = "-".join(str(c) for c in bbox)
        fn_base = f"{DIR}/data/results/{year}-{quartal}-{bbox_str}"
        tc_filename = f"{fn_base}-fused.png"
        classes_filename = f"{fn_base}-classes.tiff"
        sh.write_result(rows_classified, tc_filename, classes_filename)


if __name__ == "__main__":
    bbox = list(map(float, sys.argv[-1].split(",")))
    quartal = int(sys.argv[-2])
    year = int(sys.argv[-3])
    device_id = sys.argv[-4]
    rank_id = sys.argv[-5]
    rank_size = sys.argv[-6]
    job_id = sys.argv[-7]
    npu_config = {
        "device_id": device_id,
        "rank_id": rank_id,
        "rank_size": rank_size,
        "job_id": job_id,
        "rank_table_file": f"{DIR}/2p.json",
    }
    Inferer().run(bbox, quartal, year, npu_config)
