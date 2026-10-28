import os.path as osp

import pandas as pd

from .bases import BaseImageDataset


def _resolve_dataset_dir(root, dataset_dir="wildlifereid-10k"):
    candidates = [
        root,
        osp.join(root, dataset_dir),
        osp.join(root, "WildlifeReID-10K"),
        osp.join(root, "WildlifeReID-10k"),
        osp.join(root, "wildlifereid-10k"),
    ]
    for path in candidates:
        if osp.isdir(path) and osp.isfile(osp.join(path, "metadata.csv")):
            return path
    return osp.join(root, dataset_dir)


class WildlifeReID10kBySpecies:
    """Load WildlifeReID-10K metadata and expose one dataset per species."""

    def __init__(
        self,
        root="",
        pid_begin=0,
        species_names=None,
        wildfile_names=None,
        train_split_mode="train",
        **kwargs,
    ):
        self.dataset_dir = _resolve_dataset_dir(root)
        self.metadata_path = osp.join(self.dataset_dir, "metadata.csv")
        self.pid_begin = pid_begin
        self.train_split_mode = train_split_mode
        self.name_filter = self._normalize_filter(
            species_names if species_names is not None else wildfile_names
        )
        self._check_before_run()
        self.datasets, self.num_pids = self._process_by_species()

    @staticmethod
    def _normalize_filter(name_filter):
        if name_filter is None:
            return None
        if isinstance(name_filter, str):
            return {name_filter.strip()}
        if isinstance(name_filter, (list, tuple, set)):
            return {str(x).strip() for x in name_filter}
        raise TypeError("species_names/wildfile_names must be a string or sequence.")

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"{self.dataset_dir} does not exist.")
        if not osp.exists(self.metadata_path):
            raise RuntimeError(f"{self.metadata_path} does not exist.")

    def _process_by_species(self):
        df = pd.read_csv(self.metadata_path)
        if self.name_filter:
            species_values = set(df["species"].dropna().unique())
            dataset_values = set(df["dataset"].dropna().unique())
            matched_species = [x for x in self.name_filter if x in species_values]
            matched_datasets = [x for x in self.name_filter if x in dataset_values]
            df = df[
                df["species"].isin(matched_species)
                | df["dataset"].isin(matched_datasets)
            ]

        datasets = {}
        pid_begin = self.pid_begin
        for species_name, species_df in df.groupby("species"):
            dataset = SpeciesDataset(
                self.dataset_dir,
                species_name,
                species_df,
                pid_begin=pid_begin,
                train_split_mode=self.train_split_mode,
            )
            datasets[species_name] = dataset
            pid_begin += dataset.num_train_pids
        return datasets, pid_begin - self.pid_begin


class SpeciesDataset(BaseImageDataset):
    def __init__(
        self,
        dataset_dir,
        species_name,
        df,
        pid_begin=0,
        verbose=False,
        train_split_mode="train",
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.species = species_name
        self.pid_begin = pid_begin

        if train_split_mode == "all":
            train_df = df
        else:
            train_df = df[df["split"] == train_split_mode]
        test_df = df[df["split"] == "test"]

        self.train = self._process_df(train_df, relabel=True)
        self.query = self._process_df(test_df, relabel=False)
        self.gallery = self.query

        if verbose:
            print(f"=> WildlifeReID-10K[{self.species}] loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

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

    def _process_df(self, df, relabel=False):
        unique_identities = sorted(df["identity"].unique())
        identity2label = {
            identity: label for label, identity in enumerate(unique_identities)
        }

        dataset = []
        for idx, row in df.iterrows():
            img_path = osp.join(self.dataset_dir, row["path"])
            if relabel:
                pid = self.pid_begin + identity2label[row["identity"]]
            else:
                pid = self.pid_begin + unique_identities.index(row["identity"])
            camid = hash(row["dataset"]) % 10000
            viewid = int(idx)
            dataset.append((img_path, pid, camid, viewid))
        return dataset
