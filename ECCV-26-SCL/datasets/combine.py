from .animals import Animal
from .bases import BaseImageDataset
from .petface import PetFace
from .wildlifereid10k import WildlifeReID10kBySpecies


_FACTORY = {
    "panda": Animal,
    "elephant": Animal,
    "seal": Animal,
    "zebra": Animal,
    "shark": Animal,
    "tiger": Animal,
    "pigeon": Animal,
    "giraffe": Animal,
    "hyenaid2022": Animal,
    "leopardid2022": Animal,
    "seaturtleid2022": Animal,
    "whalesharkid": Animal,
    "chimp": PetFace,
    "chinchilla": PetFace,
    "degus": PetFace,
    "ferret": PetFace,
    "guineapig": PetFace,
    "hamster": PetFace,
    "hedgehog": PetFace,
    "javasparrow": PetFace,
    "parakeet": PetFace,
    "pig": PetFace,
    "rabbit": PetFace,
    "cat": PetFace,
    "dog": PetFace,
    "wildlifereid10k_species": WildlifeReID10kBySpecies,
}


class CombineDataset(BaseImageDataset):
    def __init__(self, names, roots, combine_pid=True, **kwargs):
        super().__init__()
        if not combine_pid:
            raise NotImplementedError("MetaN expects DATASETS.COMBINE_PID=True.")
        if len(names) != len(roots):
            raise ValueError("DATASETS names and roots must have the same length.")

        self.train_dict = {}
        self.query_dict = {}
        self.gallery_dict = {}
        self.num_classes = 0
        self.num_queries = []

        train_statistic = {}
        query_statistic = {}
        gallery_statistic = {}

        pid_begin = 0
        for dataset_name, root in zip(names, roots):
            key = dataset_name.lower()
            if key not in _FACTORY:
                raise KeyError(f"Unknown dataset: {dataset_name}")

            print(f"Loading dataset {dataset_name}, path {root}")
            dataset_or_group = _FACTORY[key](
                root=root,
                pid_begin=pid_begin,
                verbose=False,
                **kwargs,
            )

            if key == "wildlifereid10k_species":
                for species_name, dataset in dataset_or_group.datasets.items():
                    self._add_dataset(species_name, dataset, train_statistic, query_statistic, gallery_statistic)
                pid_begin += dataset_or_group.num_pids
            else:
                self._add_dataset(key, dataset_or_group, train_statistic, query_statistic, gallery_statistic)
                pid_begin += self.get_imagedata_info(dataset_or_group.train)[0]

        self._build_train_list(train_statistic)
        self._print_summary(train_statistic, query_statistic, gallery_statistic)

    def _add_dataset(self, name, dataset, train_statistic, query_statistic, gallery_statistic):
        self.train_dict[name] = dataset.train
        self.query_dict[name] = dataset.query
        self.gallery_dict[name] = dataset.gallery
        train_statistic[name] = self.get_imagedata_info(dataset.train)
        query_statistic[name] = self.get_imagedata_info(dataset.query)
        gallery_statistic[name] = self.get_imagedata_info(dataset.gallery)
        self.num_classes += train_statistic[name][0]

    def _build_train_list(self, train_statistic):
        self.train = []
        self.num_marks = 0
        for spe_id, (name, train_dataset) in enumerate(self.train_dict.items()):
            for img_path, pid, camid, viewid in train_dataset:
                self.train.append((img_path, pid, camid, viewid, spe_id, name, self.num_marks))
                self.num_marks += 1
        for name in self.train_dict:
            self.num_queries.append(self.get_imagedata_info(self.query_dict[name])[1])

    def _print_summary(self, train_statistic, query_statistic, gallery_statistic):
        names = list(self.train_dict.keys())
        print("--------------------------------------------")
        print("The combine dataset contains:")
        print(names)
        print("--------------------------------------------")
        print("Training set contains:")
        print("subset                  | # ids | # images | # cameras")
        for name in names:
            print((name + (24 - len(name)) * " " + "| "), end="")
            print(
                "{:5d} | {:8d} | {:9d}".format(
                    train_statistic[name][0],
                    train_statistic[name][1],
                    train_statistic[name][2],
                )
            )
        total_train_statistic = self.get_dataset_info(self.train)
        print("--------------------------------------------")
        print("Total training set:")
        print("# ids | # images | # cameras | # species")
        print(
            "{:5d} | {:8d} | {:8d} | {:8d}".format(
                total_train_statistic[0],
                total_train_statistic[1],
                total_train_statistic[2],
                total_train_statistic[4],
            )
        )
        print("--------------------------------------------")

    @staticmethod
    def get_dataset_info(data):
        pids, cams, vids, speids = [], [], [], []
        for _, pid, camid, vid, spe_id, *_ in data:
            pids.append(pid)
            cams.append(camid)
            vids.append(vid)
            speids.append(spe_id)
        return len(set(pids)), len(data), len(set(cams)), len(set(vids)), len(set(speids))
