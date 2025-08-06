
import argparse
import json
import logging as log
import re

import defusedxml.ElementTree as etree
import motmetrics as mm
import numpy as np
from tqdm import tqdm
import io
from collections import namedtuple
TrackedObj = namedtuple('TrackedObj', 'rect label')


import os
from tqdm import tqdm
import argparse
import json
import logging as log
import os
from os import path as osp
import pandas as pd
from io import StringIO
import motmetrics as mm
import numpy as np
from tqdm import tqdm

import logging


def read_gt_tracks(gt_filenames, size_divisor=1, skip_frames=1):
    min_last_frame_idx = -1
    camera_tracks = [[] for _ in gt_filenames]
    for i, filename in enumerate(gt_filenames):
        last_frame_idx = -1
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in tqdm(lines, desc='Reading ' + filename):
            if skip_frames > 0 and int(line.split(',')[0]) % skip_frames == 0:
                continue
            # id, x_left, y_top, width, height, frame_id, _, _, _, _ = line.strip().split(' ')
            frame_id, id, x_left, y_top, width, height, _, _, _, _ = line.strip().split(',')
            x_left = int(float(x_left)) // size_divisor
            y_top = int(float(y_top)) // size_divisor
            x_right = int((float(x_left) + float(width)) // size_divisor)
            y_bottom = int((float(y_top) + float(height)) // size_divisor)
            # assert x_right > x_left
            # assert y_bottom > y_top
            track = {'id': int(id), 'boxes': [], 'timestamps': []}
            track['boxes'].append([x_left, y_top, x_right, y_bottom])
            track['timestamps'].append(int(frame_id) // size_divisor)
            last_frame_idx = max(last_frame_idx, track['timestamps'][-1])
            camera_tracks[i].append(track)
        if min_last_frame_idx < 0:
            min_last_frame_idx = last_frame_idx
        else:
            min_last_frame_idx = min(min_last_frame_idx, last_frame_idx)
    return camera_tracks, min_last_frame_idx


def get_detections_from_tracks(tracks_history, time):
    active_detections = [[] for _ in tracks_history]
    for i, camera_hist in enumerate(tracks_history):
        for track in camera_hist:
            # print(f"Processing track {track} at time {time}")
            if time in track['timestamps']:
                idx = track['timestamps'].index(time)
                active_detections[i].append(TrackedObj(track['boxes'][idx], track['id']))
    return active_detections



def check_contain_duplicates(all_detections):
    for detections in all_detections:
        all_labels = [obj.label for obj in detections]
        uniq = set(all_labels)
        if len(all_labels) != len(uniq):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description='Multi-camera tracking evaluation')
    parser.add_argument('--history_file', type=str, required=True,
                        help='JSON file with tracking results')
    parser.add_argument('--gt_files', type=str, nargs='+', required=True,
                        help='MOT-format ground truth files (one per camera)')
    parser.add_argument('--size_divisor', type=int, default=1,
                        help='GT resolution divisor (for resizing)')
    parser.add_argument('--skip_frames', type=int, default=0,
                        help='Evaluate every n-th frame')

    args = parser.parse_args()

    with open(args.history_file) as f:
        history = json.load(f)

    assert len(args.gt_files) == len(history)

    gt_tracks, last_frame_idx = read_gt_tracks(
        args.gt_files, size_divisor=args.size_divisor, skip_frames=args.skip_frames
    )

    accs = [mm.MOTAccumulator(auto_id=True) for _ in args.gt_files]

    # Use StringIO to hold predictions in memory
    hota_writers = [io.StringIO() for _ in range(len(history))]

    # Process each frame
    for frame_id in tqdm(range(last_frame_idx + 1), desc="Processing detections"):
        active_detections = get_detections_from_tracks(history, frame_id)
        gt_detections = get_detections_from_tracks(gt_tracks, frame_id)

        if check_contain_duplicates(active_detections):
            print(f"[WARNING] Duplicate IDs at frame {frame_id}")

        for cam_id, camera_gt_detections in enumerate(gt_detections):
            gt_boxes, gt_labels = [], []
            for obj in camera_gt_detections:
                x1, y1, x2, y2 = obj.rect
                gt_boxes.append([x1, y1, x2 - x1, y2 - y1])
                gt_labels.append(obj.label)

            ht_boxes, ht_labels = [], []
            for obj in active_detections[cam_id]:
                x1, y1, x2, y2 = obj.rect
                ht_boxes.append([x1, y1, x2 - x1, y2 - y1])
                ht_labels.append(obj.label)

                # Write detection to HOTA format (in memory buffer)
                hota_writers[cam_id].write(
                    f"{frame_id},{obj.label},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1\n"
                )

            # Compute distance matrix and update accumulator
            distances = mm.distances.iou_matrix(
                np.array(gt_boxes), np.array(ht_boxes), max_iou=1.0
            )
            accs[cam_id].update(gt_labels, ht_labels, distances)

    # ===== MOTA Evaluation =====
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        metrics=mm.metrics.motchallenge_metrics,
        generate_overall=True,
        names=[f'cam {i}' for i in range(len(accs))]
    )

    print("\n===== MOTA Metrics =====")
    print(mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    ))



if __name__ == '__main__':
    main()
    