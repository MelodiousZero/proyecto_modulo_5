import pandas as pd
import datetime as dt
from ucimlrepo import fetch_ucirepo 



class cleanDataFrame:
    def __init__(self):
        self.dataframe = self.get_dataset()
        self.target = self.fetch().data.targets.columns[0]
    
    def get_dataset(self): #proviene de la documentación oficial
        seoul_bike_sharing_demand = self.fetch()
        # metadata 
        print(seoul_bike_sharing_demand.metadata) 
        
        # variable information 
        print(seoul_bike_sharing_demand.variables) 


        seoul_bike_sharing_demand_df = pd.concat([seoul_bike_sharing_demand.data.features, seoul_bike_sharing_demand.data.targets], axis=1)
        return seoul_bike_sharing_demand_df


    def fetch(self):
        # fetch dataset 
        seoul_bike_sharing_demand = fetch_ucirepo(id=560) 
        
        # data (as pandas dataframes) 
        X = seoul_bike_sharing_demand.data.features 
        y = seoul_bike_sharing_demand.data.targets 
        return seoul_bike_sharing_demand
            
    def make_date(self):
        self.dataframe['Date'] = pd.to_datetime(self.dataframe['Date'],format="mixed")
        self.dataframe['Month'] = self.dataframe["Date"].dt.month_name()
        self.dataframe['Day'] = self.dataframe["Date"].dt.strftime("%A")
        self.dataframe.drop("Date",inplace=True,axis=1)
        

    
    def make_weekend(self):
        self.dataframe['is_weekend'] = self.dataframe['Day'].apply(lambda x: True if x in ['Saturday','Sunday'] else False)
    def drop_unneccesary_columns(self):
        self.dataframe.drop("Dew point temperature",inplace=True,axis=1)

    def replace_holiday(self):
        self.dataframe['Holiday'] = self.dataframe['Holiday'].apply(lambda x: 1 if x=="Holiday" else 0 )
    
    def replace_functioning_day(self):
        self.dataframe['Functioning Day'] = self.dataframe['Functioning Day'].apply(lambda x: 1 if x=="Yes" else 0 )
    
    def encode(self):
        season_map = {'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3}
        self.dataframe['Seasons'] = self.dataframe['Seasons'].map(season_map)
        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
                'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

        month_map = {'January': 0, 'February': 1, 'March': 2, 'April': 3,
                    'May': 4, 'June': 5, 'July': 6, 'August': 7,
                    'September': 8, 'October': 9, 'November': 10, 'December': 11}
        functioning_day_map = {"Yes":1,"No":0}

        self.dataframe['Day'] = self.dataframe['Day'].map(day_map)
        self.dataframe['Month'] = self.dataframe['Month'].map(month_map)
        self.dataframe["Functioning Day"]=self.dataframe["Functioning Day"].map(functioning_day_map)



    def run_clean(self):
        self.make_date()
        self.make_weekend()
        self.drop_unneccesary_columns()
        self.replace_holiday()
        self.encode()
        output_df = self.dataframe
        return output_df
