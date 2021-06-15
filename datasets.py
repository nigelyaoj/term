import pandas as pd

class Coleridge1(object):
    def __init__(self, data_path) -> None:
        super().__init__()

        self.load_data(data_path)

    def load_data(self, data_path):
        
        data = pd.read_csv(data_path)
        
        self.sentences = list(data["Sentence"].values)
        self.can_datasets = list(data["Can_data"].values)
        self.labels = list(data["Labels"].values)
    
    def __len__(self):
        
        return len(self.sentences)

    def __getitem__(self, idx):

        return self.sentences[idx], self.can_datasets[idx], self.labels[idx]

