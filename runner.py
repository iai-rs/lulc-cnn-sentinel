# encoding: utf-8
"""Home of the inference runner across multiple NPU devices."""

import os
import csv
from multiprocessing import Pool, Process

from inference_script import Inferer

# YEARS = ["2016", "2017", "2018", "2019", "2020", "2021"]
YEARS = ["2021"]
# YEARS = ["2019", "2020", "2021"]
# MONTHS = [str(i) for i in range(12)]
MONTHS = ["9"]

DIR = os.path.dirname(os.path.realpath(__file__))
# DEVICE_IDS = ["0", "1"]
DEVICE_IDS = ["0"]
# RANK_SIZE = "2"
RANK_SIZE = "1"
# RANK_IDS = ["0", "1"]
# RANK_IDS = ["0", "0"]
RANK_IDS = ["0"]
JOB_ID = "10385"
# TABLE_FILE = f"{DIR}/2p.json"
TABLE_FILE = f""


def get_inputs(filename):
    """list of input arguments for inference."""
    with open(filename) as f:
        return [line.split(" ") for line in f.readlines()]


def load_bboxes_from_polygons(filename) -> list:
    """list of lists of str inputs for sentinel."""
    with open(filename, newline="") as f:
        reader = csv.reader(f, delimiter=",")
        rows = [row for row in reader][1:]
    polygons = {}
    for [id_, lon, lat, *row] in rows:
        if id_ not in polygons:
            polygons[id_] = {"lats": [], "lons": []}
        polygons[id_]["lats"].append(lat)
        polygons[id_]["lons"].append(lon)
    bboxes = {
        (
            min(polygons[id_]["lons"]),
            min(polygons[id_]["lats"]),
            max(polygons[id_]["lons"]),
            max(polygons[id_]["lats"]),
        )
        for id_ in polygons
    }
    return [
        {"year": year, "month": month, "bbox": bbox}
        for year in YEARS
        for month in MONTHS
        for bbox in bboxes
    ]



def run_inference(filename):
    """Run sentinel inference script on multiple NPU devices."""
    n_devices = len(DEVICE_IDS)
    data = load_bboxes_from_polygons(filename)
    print(data)
    inputs = [
        [
            row['bbox'],
            int(row['month']),
            int(row['year']),
        ]
        for row in data
    ]
    pconfigs = {
        DEVICE_IDS[i]: {
            "device_id": DEVICE_IDS[i],
            "rank_id": str(i),
            "rank_size": RANK_SIZE,
            "job_id": JOB_ID,
            "rank_table_file": f"{DIR}/2p.json",
        }
        # for i in range(2)
        for i in range(1)
    }

    tg = Inferer().run
    pool = {
        dev_id: Process(target=tg, args=(inputs.pop() + [pconfigs[dev_id]]))
        for dev_id in DEVICE_IDS
    }

    n_inputs = len(inputs)

    # --- Initial process start for all devices ---
    for dev_id in pool:
        pool[dev_id].start()

    while inputs:
        for dev_id in pool:
            if pool[dev_id].is_alive():
                continue
            print(f"******************** AT POLYGON: {len(inputs)} of {n_inputs} *******************")
            pool[dev_id] = Process(target=tg, args=(inputs.pop() + [pconfigs[dev_id]]))
            pool[dev_id].start()

    for dev_id in pool:
        pool[dev_id].join()


if __name__ == "__main__":
    run_inference("yearlyNoviSad.csv")
    # run_inference("filtVertex.csv")
