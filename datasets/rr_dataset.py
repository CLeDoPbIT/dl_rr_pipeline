from torch.utils.data import Dataset, DataLoader


def create_dataloader(dataloader_config, dataset_class, x, y):
    dataset = dataset_class(x, y)
    data_loader = DataLoader(dataset, batch_size=dataloader_config["batch_size"], shuffle=dataloader_config["shuffle"],
                             num_workers=dataloader_config["num_workers"], pin_memory=dataloader_config["pin_memory"])
    return data_loader


class DummyDataset(Dataset):
    def __init__(self, x, y):
        self.len = len(x)
        self.x = x
        self.y = y

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.y[index]

