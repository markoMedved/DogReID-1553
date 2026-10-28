# encoding: utf-8
import glob
import re

import os
from pathlib import Path
import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle

import pandas as pd


class PetFace(BaseImageDataset):
    dataset_dir = "PetFace"

    def __init__(self, root="", verbose=True, pid_begin=0, **kwargs):
        """
        root: fake dataset dir(/data2/wuyurui_data/PetFace/{category})
        dataset_dir: dataset base dir(/data2/wuyurui_data/PetFace)
        """
        super(PetFace, self).__init__()
        self.dataset_dir, self.category = osp.split(root)
        self.train_dir = osp.join(self.dataset_dir, "split", self.category, "train.csv")
        self.test_dir = osp.join(self.dataset_dir, "split", self.category, "test.txt")

        self._check_before_run()
        self.pid_begin = pid_begin

        train = self._process_train_dir(self.train_dir, relabel=True)
        query, gallery = self._process_query_gallery_dir(self.test_dir, relabel=False)
        self.train, self.query, self.gallery = train, query, gallery

        if verbose:
            print("=> PetFace loaded")
            print("=> Category: " + ", ".join(self.category))
            self.print_dataset_statistics(train, query, gallery)

        (
            self.num_train_pids,
            self.num_train_imgs,
            self.num_train_cams,
            self.num_train_vids,
        ) = self.get_imagedata_info(self.train)
        (
            self.num_query_pids,
            self.num_query_imgs,
            self.num_query_cams,
            self.num_query_vids,
        ) = self.get_imagedata_info(self.query)
        (
            self.num_gallery_pids,
            self.num_gallery_imgs,
            self.num_gallery_cams,
            self.num_gallery_vids,
        ) = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
                raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
                raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_train_dir(self, train_path, relabel=False):
        df = pd.read_csv(train_path)
        img_paths = [
            osp.join(self.dataset_dir, "images", filename)
            for filename in df["filename"].tolist()
        ]
        label_lists = df["label"].tolist()

        pid_container = set()
        for index, img_path in enumerate(img_paths):
            pid = int(osp.basename(osp.dirname(img_path)))
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for index, img_path in enumerate(img_paths):
            pid = int(osp.basename(osp.dirname(img_path)))
            if pid == -1:
                continue  # junk images are just ignored
            # assert 0 <= pid <= 1501  # pid == 0 means background
            # assert 1 <= camid <= 6
            # camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]

            dataset.append((img_paths[index], self.pid_begin + pid, 1, 1))
        return dataset

    def _process_query_gallery_dir(self, test_path, relabel=False):
        img_paths = dict()
        query_dataset = []
        gallery_dataset = []

        with open(test_path, "r") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                pid = osp.basename(osp.dirname(line))
                if pid in img_paths.keys():
                    img_paths[pid].append(line)
                else:
                    img_paths[pid] = [line]

        for pid, images in img_paths.items():
            for img in images:
                query_dataset.append(
                    (
                        osp.join(self.dataset_dir, "images", img),
                        self.pid_begin + int(pid),
                        0,
                        1,
                    )
                )
                gallery_dataset.append(
                    (
                        osp.join(self.dataset_dir, "images", img),
                        self.pid_begin + int(pid),
                        1,
                        1,
                    )
                )

        return query_dataset, gallery_dataset
