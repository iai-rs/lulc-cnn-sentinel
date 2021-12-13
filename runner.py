# encoding: utf-8
"""Home of the inference runner across multiple NPU devices."""

import os
from multiprocessing import Pool, Process

from inference_script import Inferer

DIR = os.path.dirname(os.path.realpath(__file__))
DEVICE_IDS = ["0", "1"]
# RANK_SIZE = "2"
RANK_SIZE = "1"
# RANK_IDS = ["0", "1"]
RANK_IDS = ["0", "0"]
JOB_ID = "10385"
# TABLE_FILE = f"{DIR}/2p.json"
TABLE_FILE = f""


def get_inputs(filename):
    """list of input arguments for inference."""
    with open(filename) as f:
        return [line.split(" ") for line in f.readlines()]


def run_inference(filename):
    """Run sentinel inference script on multiple NPU devices."""
    n_devices = len(DEVICE_IDS)
    inputs = [
        [
            bbox_str.strip("\n").split(","),
            int(quartal),
            int(year),
            {
                "device_id": DEVICE_IDS[i % n_devices],
                "rank_id": str(i % n_devices),
                "rank_size": RANK_SIZE,
                "job_id": JOB_ID,
                "rank_table_file": f"{DIR}/2p.json",
            },
        ]
        for i, (year, quartal, bbox_str) in enumerate(get_inputs(filename))
    ]

    tg = Inferer().run
    pool = {dev_id: Process(target=tg, args=inputs.pop()) for dev_id in DEVICE_IDS}
    for pid in pool:
        pool[pid].start()
    while inputs:
        for dev_id in pool:
            if pool[dev_id].is_alive():
                continue
            pool[dev_id] = Process(target=tg, args=inputs.pop())
            pool[dev_id].start()

    for dev_id in pool:
        pool[dev_id].join()


if __name__ == "__main__":
    run_inference("input.txt")
