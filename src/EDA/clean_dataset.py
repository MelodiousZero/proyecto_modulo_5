import pandas as pd
import datetime as dt


class cleanDataFrame:
    def __init__(self,dataframe):
        self.dataframe = dataframe
    
    def make_date(self):
        self.dataframe['Date'] = pd.to_datetime(self.dataframe['Date'],format="mixed")
        self.dataframe['Month'] = self.dataframe["Date"].dt.month_name()
        self.dataframe['Day'] = self.dataframe["Date"].dt.strftime("%A")
        

    
    def make_weekend(self):
        self.dataframe['is_weekend'] = self.dataframe['Day'].apply(lambda x: True if x in ['Saturday','Sunday'] else False)
