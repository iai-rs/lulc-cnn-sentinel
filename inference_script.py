# encoding: utf-8
"""Inference script for Sentinel-2 land use classification."""
import os
import sys
from glob import glob
from skimage.io import imread

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from image_functions import simple_image_generator, preprocessing_image_ms

from helpers import NpuHelperForTF, SentinelHelper
from utils import lazyproperty

# --------------------------- GENERAL CONFIG ------------------------------------------
DIR = os.path.dirname(os.path.realpath(__file__))

class_indices = {
    "AnnualCrop": 0,
    "Forest": 1,
    "HerbaceousVegetation": 2,
    "Highway": 3,
    "Industrial": 4,
    "Pasture": 5,
    "PermanentCrop": 6,
    "Residential": 7,
    "River": 8,
    "SeaLake": 9,
}
num_classes = len(class_indices)

class Inferer:
    """Implementation of the inference running procedure."""

    # model_path = f"{DIR}/data/models/vgg_ms_transfer_final.44-0.969.hdf5"
    model_path = f"{DIR}/data/models/vgg_ms_transfer_final.47-0.935.hdf5"

    @lazyproperty
    def _model(self):
        return load_model(self.model_path)
    
    def confusion(self):
        print("************* STARTING CONFUSION *********************")
        path_to_split_datasets = "/home/slobodan/tutorials/sentinel/data/ms/"
        path_to_validation = os.path.join(path_to_split_datasets, "validation")
        for class_ in class_indices:
            path = f"{path_to_validation}/{class_}/*.tif"
            files = glob(path)
            pred_imgs = []
            for file_ in files:
                image = np.array(imread(file_), dtype=float)
                image = preprocessing_image_ms(image)
                pred_imgs.append(image)
            arr = np.array(pred_imgs)
            pred = self._model.predict(arr)
            pred_classes = np.argmax(pred, axis=1)
            confs = []
            for k in class_indices:
                conf = np.sum(pred_classes == class_indices[k]) / len(pred_classes)
                confs.append(conf)
            print(confs)
        # gen = simple_image_generator(
            # validation_files, class_indices, batch_size=64
        # )
        # print("************* CLASSIFYING ROWS *********************")
        # rows_classified = self._model.predict(gen, steps=10, verbose=1)

    def run(self, bbox, month, year, npu_config):
        """Execute inference procedure."""
        # ------------------------ PREPARE FILENAMES ----------------------------------
        bbox_str = "-".join(str(c) for c in bbox)
        # fn_base = f"{DIR}/data/monthly/{year}-{quartal}-{bbox_str}"
        # fn_base = f"{DIR}/data/monthly/{year}-{month + 1}-{bbox_str}"
        fn_base = f"{DIR}/data/test/{year}-{month + 1}-{bbox_str}"
        fused_filename = f"{fn_base}-fused.png"
        tc_filename = f"{fn_base}-tc.png"
        classes_filename = f"{fn_base}-classes.tiff"
        # if os.path.exists(fused_filename) or os.path.exists(classes_filename):
            # --- Don't repeat training already executed
        #    return

        # ------------------------ INIT NPU BEFORE LOADING MODEL ----------------------
        sess = NpuHelperForTF(**npu_config).sess
        model = self._model
        in_shp = model.layers[0].input_shape[0][1:3]
        print("*********** IN SHP: *******************")
        print(in_shp)

        # ------------------------ LOAD IMAGE FROM SENTINEL ---------------------------
        # sh = SentinelHelper(bbox, quartal, year)
        sh = SentinelHelper(bbox, month, year)
        gen, steps = sh.input_generator(in_shp), sh.image.shape[0]

        # ------------------------ GENERATING PREDICTIONS -----------------------------
        rows_classified = model.predict(gen, steps=steps, verbose=1)
        sess.close()  # close NPU session right after inference

        # ------------------------ WRITING RESULT IMAGES ------------------------------
        sh.write_result(rows_classified, fused_filename, classes_filename)
        # sh.write_tc(tc_filename)


if __name__ == "__main__":
    # bbox = list(map(float, sys.argv[-1].split(",")))
    # quartal = int(sys.argv[-2])
    # year = int(sys.argv[-3])
    # device_id = sys.argv[-4]
    # rank_id = sys.argv[-5]
    # rank_size = sys.argv[-6]
    # job_id = sys.argv[-7]
    # npu_config = {
        # "device_id": device_id,
        # "rank_id": rank_id,
        # "rank_size": rank_size,
        # "job_id": job_id,
        # "rank_table_file": f"{DIR}/2p.json",
    # }
    # Inferer().run(bbox, quartal, year, npu_config)
    rows = Inferer().confusion()
    print(rows)
