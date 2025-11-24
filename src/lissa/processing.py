import pandas as pd
import json

from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple
#from scipy import stats

from .picture import Traducao, Measures

#from pydantic import BaseModel, Field

from scipy import signal as sig

# class Equipment(BaseModel):
#     name: str
#     info: str
#     failure_info: str
#     data : pd.DataFrame





def DataPreparer(dirFile: str,fileName: str) -> pd.DataFrame:

    '''
    The one-hour frequency file is supposed to be loaded into baseData. Then, data has its types infered and time is defined as time. 

    Nonetheless, the duplicates are delete and the data is exported.

    But it's expected to work in any file that has the same headers that what came for us in the aforementioned file.

    '''

    filePath = dirFile + "/" + fileName #path to file

    #read data and convert to respective type
    baseData = pd.read_csv(filePath,delimiter=",") 
    baseData = baseData.infer_objects(copy=False)
    baseData["time"] = pd.to_datetime(baseData["time"])

    baseData["Well"] = baseData["Well Run"].str[:4]
    print("Pumps : #" + str(baseData["Well"] .unique().size) + "\n"+"Runs: #" + str(baseData["Well Run"].unique().size))

    dup = baseData[list(set(list(baseData))-{"Well_down","Well Run"})].duplicated()
    baseData.drop(index = baseData[dup].index,columns="Well",inplace=True)
    return baseData


def FailureMerge(baseData: pd.DataFrame ,dirFile:str ,fileName:str) -> Tuple[pd.DataFrame,list]:

    '''
    Then, the file respostas.csv is called. respostas.csv is an PEREGRINO ESP Run Life fail spreadsheet.

    This file is loaded with the intent of allowing our data be linked with faliure date.

    A column, called "Well_down" was created by Equinor, analyzing if the well was on off or online. 

    Having the faliure date at main data, it is possible to realize a comparision between the dates and infer if there will be a faliure or not. Therefore, with this approach, we can label all data, if it's relatable with a faliure or not.

    However, this approach does not point exactly when it was observed the faliure, but, in reality, only labels if the proababilty of the anomaly is inside the subset is greater.

    '''

    filePath = dirFile + "/" + fileName #path to file
    respostas = pd.read_csv(filePath, delimiter=",") #import faliure data

    pumpList = baseData["Well Run"].unique() #filter all pump names at the dataset

    
    respostas = respostas.loc[respostas["Well Run"].isin(pumpList)] #only analyzes data that is at the time dataset

    #faliure precise time is not considered on the dataset, since its only a report.
    respostas["Failure Date"] = pd.to_datetime(respostas["Failure Date"],format="%d/%m/%y").dt.tz_localize('UTC')

    entireData = pd.merge(baseData,respostas,how="left",on="Well Run") #do a left merge to unite failiure date with time info

    entireData["Failure"] = entireData["time"] >= entireData["Failure Date"] #check if failiure already happened

    dropList = ['Start-up Date','Start-up Date','ESP Failure?','Failure Date'] 

    entireData.drop(dropList,axis=1,inplace=True) #remove duplicated columns

    # pumpData.drop(entireData.loc[baseData["Well_down"]==True].index,inplace=True) #offline data is dropped
    # entireData.drop(columns="Well_down",inplace=True) #we dont need anymore to know when the well is down

    entireData.set_index("time",inplace=True) #time is set as index

    return entireData, pumpList

def FeatureCreation(entireData:pd.DataFrame)->pd.DataFrame:
    '''
    This function is intended to process columns for feature creation. 

    Currently is only synthetizing current A,B,C in a norm-2, doing the same for vibrations and converting columns written as 1/100 units in fractions.

    Users might want to deal with new features. It is not the best pratice, but they should be written here (inside the function).
    
    '''
    entireData["ESP Current Module"] = (
    entireData["ESP motor Current - phase A"].pow(2)+
    entireData["ESP motor Current - phase B"].pow(2)+
    entireData["ESP motor Current - phase C"].pow(2)).pow(1/2)

    # entireData["Current Mean"] = (
    #     entireData["ESP motor Current - phase A"]+
    #     entireData["ESP motor Current - phase B"]+
    #     entireData["ESP motor Current - phase C"])/3

    entireData["ESP Vibration Module"] = (entireData["ESP Vibration X"].pow(2)+entireData["ESP Vibration Y"].pow(2)).pow(1/2)

    entireData["ESP Motor Voltage"] = entireData["ESP Motor Voltage"]/1000

    # entireData["ESP Power"] = entireData["ESP Motor Voltage"]*entireData["Current Mean"]

    entireData["Choke Opening"] = entireData["Choke Opening"]/100
    entireData["Water Cut @ 20degC - 1 atm"] = entireData["Water Cut @ 20degC - 1 atm"]/100

    # entireData["POT Well Head"] = (entireData["Well head pressure"]/(entireData["Well head Temperature"]+273)).fillna(0)
    # entireData["POT Intake"] = (entireData["ESP intake Pressure"]/(entireData["ESP intake temperature"]+273)).fillna(0)
    # entireData["POT Discharge"] = entireData["ESP discharge pressure"]/(entireData["ESP discharge temperature sensor"]+273)

    entireData.drop(columns=[
        # "Current Mean",
        "ESP motor Current - phase A",
        "ESP motor Current - phase B",
        "ESP motor Current - phase C",
        "ESP Vibration X",
        "ESP Vibration Y",
        # "ESP Motor Voltage",
        'ESP differential pressure', #we are going to try to find this in PCA
        # "Well head pressure",
        # "ESP intake Pressure",
        # "ESP discharge pressure",
        # "Well head Temperature",
        # "ESP intake temperature",
        "ESP discharge temperature sensor" #there's too many few entries to be considered.

    ],inplace=True)

    return entireData


def SetHeaders():
    '''
    This function gives all headers into the original dataset.
    
    '''
    Temperature = [
    'ESP discharge temperature sensor',
    'ESP intake temperature',
    'ESP motor temperature',
    'Well head Temperature'
    ]

    Pressure = [
        'ESP intake Pressure',
        'ESP discharge pressure',
        'ESP differential pressure',
        'Well head pressure'
        ]

    Electrical = [
        "ESP motor Current - phase A",
        "ESP motor Current - phase B",
        "ESP motor Current - phase C",
        'VSD power frequency',
        'ESP Motor Voltage'
    ]

    Vibration = [
        "ESP Vibration X",
        "ESP Vibration Y"
    ]

    Other = [
        'Choke Opening',
        'Water Cut @ 20degC - 1 atm'
    ]

    return Temperature, Pressure, Electrical, Vibration, Other

def mad(series):
    return np.median(np.abs(series - np.median(series)))

def SeriesBack(Filter):
       return (Filter
        .reset_index()
        .set_index("time")
        .fillna(0)
        .sort_index()
        .drop(columns="Well_down")
    )

def FilterProcedure(entireData: pd.DataFrame, pump: str, windowSize: int)->pd.DataFrame:

    '''
    For each pump, for selected numerical properties, a low frequency filter is passed, trying to reduce the noise into the data. 

    '''

    exportData = (entireData.loc[entireData["Well Run"] == pump].copy()) #copies the original dataset

    numerical_headers = [
    'ESP intake temperature',
    'ESP motor temperature',
    'Well head Temperature',
    'ESP intake Pressure',
    'ESP discharge pressure',
    'Well head pressure',
    'VSD power frequency',
    'ESP Motor Voltage',
    'ESP Current Module',
    'ESP Vibration Module'
    ]

    Filter = exportData.groupby("Well_down")[numerical_headers].apply(lambda x: (x.ewm(span=24*windowSize).mean()-x.expanding().median())/x.expanding().std())

    

    #all things i had tried:

    # Filter = exportData.groupby("Well_down")[Headers].apply(
    #     lambda x: (1+x.ewm(span=24*windowSize).mean()/1+x.expanding().median()).apply(np.log10)
    #     )#/x.expanding().std()) #-> this one have interesting results

    #Filter = exportData.groupby("Well_down")[Headers].apply(lambda x: (x.ewm(span=24*windowSize).mean()-x.expanding().min())/(x.expanding().max()-x.expanding().min()))
    
    #Filter = exportData.groupby("Well_down")[Headers].apply(lambda x: (x.ewm(span=24*windowSize).mean()-x.expanding().median()))

    #Filter = exportData.groupby("Well_down")[Headers].apply(lambda x: (x.ewm(span=24*windowSize).mean()/x.ewm(span=24*windowSize).std())/(x.ewm(span=365).mean()/x.ewm(span=365).std()))

    # Measure = exportData.groupby("Well_down")[Headers].apply(lambda x: (x.ewm(span=24*windowSize).mean()-x.expanding().median()))
    # MAD = Measure.groupby("Well_down")[Headers].apply(lambda x: x.abs().expanding().median())

    # Filter = SeriesBack(Measure)/SeriesBack(MAD)

    
    Filter = (Filter
        .reset_index()
        .set_index("time")
        .fillna(0)
        .sort_index()
        .drop(columns="Well_down")
    )

   
    removedHeaders = [
    'Water Cut @ 20degC - 1 atm',
    'Choke Opening',
    "Well aligned to Train A",
    "Well aligned to Train B",
    'Failure Info',
    'Pump Info',
    "Well Run",
    "Failure",
    "Well_down"
    
    ]


    return pd.merge(Filter,exportData[removedHeaders],how="left",on="time")


def ProcessData(pumpList: list ,entireData: pd.DataFrame ,windowSize=1,totalDataPath="/")->pd.DataFrame:
    '''

    Apply filter to all pumps and export the resulting dataset.
    
    '''
    totalData = pd.DataFrame(columns=list(entireData))

    for pump in pumpList:
        totalData = pd.concat([totalData,
                            FilterProcedure(entireData, pump,windowSize)
                            ],axis=0)

    totalData.reset_index(inplace=True)
    totalData.to_csv(totalDataPath)
    totalData.set_index("index",inplace=True)
    
    return totalData


def relevantHeader(data: pd.DataFrame) -> np.ndarray:

    '''
        Lists only numerical headers in dataset. 
    '''
    Headers = list(data) #all headers
    notNumericalHeaders = [
        "Well Run",
        "Failure",
        "Well_down",
        "time",
        "Well aligned to Train A",
        "Well aligned to Train B",
        'Water Cut @ 20degC - 1 atm',
        'Choke Opening',
        'Failure Info',
        'Pump Info'
        ]
    return np.sort(list(set(Headers)-set(notNumericalHeaders)))



def PropertyTableGenerator(
        tabelaProps : pd.DataFrame,
        path = "./",
        name="tabela.csv"):
    
    '''
        Creates a table with translated names and units 
    '''
    tabelaProps.reset_index(inplace=True)

    tabelaProps["Unidade"] = tabelaProps["index"].apply(lambda x: Measures(x))
    tabelaProps["index"]= tabelaProps["index"].apply(lambda x: Traducao(x))
    tabelaProps.to_csv(path+name)


    return tabelaProps




'''
General things here:

All filters are normalized here, this means that filter return is f(x) = (H(x) - mean(x))/std(x). 
A separated approach was tried, but the results were strange.
As a standard, all windows sizes are set in matter of days. If Equinor releases full granularity data, we will be allowed to change this parameter.
'''

#High Frequency Filter -> H_high= 1 - H_low
def HighFrequencyFilter(x,windowSize):
    return (x-x.ewm(span=24*windowSize).mean()-x.expanding().mean())/x.expanding().std()

#Low Frequency Filter
def LowFrequencyFilter(x,windowSize):
    return (x.ewm(span=24*windowSize).mean()-x.expanding().mean())/x.expanding().std()

#Mean Average Filter
def MeanAverageFilter(x,windowSize):
    return (x.rolling(24*windowSize,min_periods=0).mean()-x.expanding().mean())/x.expanding().std()

#Standard filter
def StandardizerFilter(x):
    return (x-x.expanding().mean())/x.expanding().std()


#Get all numerical Headers 

#improve: select numerical only

def FilterCreation(exportData,windowSize,filterType):

    groupData = exportData.copy()
    groupData["group"] = (exportData["Well_down"] != exportData["Well_down"].shift()).cumsum()
    groupData = groupData.groupby('group')

    Filter = groupData[relevantHeader(exportData)].apply(lambda x: filterType(x,windowSize))
    
    return (Filter
        .reset_index()
        .set_index("time")
        .drop(columns="group")
        .fillna(0))
    
