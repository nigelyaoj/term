import pandas as pd



class Coleridge1(object):
    def __init__(self, data_path) -> None:
        super().__init__()

        data = pd.read_csv(data_path)
        

