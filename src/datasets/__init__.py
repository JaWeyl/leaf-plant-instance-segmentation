from datasets.MyDataset import MyDataset
from datasets.CVPPPDataset import CVPPPDataset

def get_dataset(name, dataset_opts):
    if name == "mydataset":
        return MyDataset(**dataset_opts)
    elif name == "cvppp":
        return CVPPPDataset(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))