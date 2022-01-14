
### Chargement des packages 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


"""L'imputation des données manquantes a été fait par interpolation 
avec des méthodes différentes pour le train et le test set: les règles du défi 
nous imposaient d'enlever toutes les données manquantes pour les 85140 Ids à prédire, 
on a choisi de ne chercher que parmi les cinq stations les plus proches pour imputer 
les NaNs dans le train set afin de ne pas introduire trop de biais lors de l'apprentissage
puis on avons poussé l'imputation plus loin pour le test set pour remplacer tous les NaNs"""

### ************************* Preprocessing Train *******************
    
#### Chargement des données Train #################################
DATA_PATH = "../data/"
X_station_train = pd.read_csv(DATA_PATH+ 'Train/X_station_train.csv',
                              parse_dates=['date'],infer_datetime_format=True)

Y_train= pd.read_csv(DATA_PATH+ 'Train/Y_train.csv',
                     parse_dates=['date'],infer_datetime_format=True)

BaseF_train= pd.read_csv(DATA_PATH+ 'Train/Baselines/Baseline_forecast_train.csv',
                         parse_dates=['date'],infer_datetime_format=True)

BaseO_train= pd.read_csv(DATA_PATH+'Train/Baselines/Baseline_observation_train.csv',
                         parse_dates=['date'],infer_datetime_format=True)

#### Feature engineering Train set #################################

x = X_station_train.sort_values(by=["number_sta","date"]) #sort by station, then by date 
x['number_sta']=x['number_sta'].astype('category') 

# Considérer la moyenne de ff,t,td,dd and hu par jour 
X_station_date = x[{"number_sta","date","ff","t","td","dd","hu"}]
X_station_date.set_index('date',inplace = True)  

X_station_date = X_station_date.groupby('number_sta').resample('D').agg(pd.Series.mean)
X_station_date = X_station_date.reset_index(['date','number_sta'])
X_station_date['number_sta'] = X_station_date['number_sta'].astype('category') 

# On rajoute les baselines et Ytrain pour les imputer en même temps 
BaseO_train.columns= ['number_sta','date','Base_Obs','Id']
dataTrain= pd.merge(X_station_date, BaseO_train, on=['number_sta','date'], how= 'left')
dataTrain= pd.merge(dataTrain, BaseF_train, on=['number_sta','date','Id'], how= 'left')
dataTrain= pd.merge(dataTrain, Y_train, on=['number_sta','date','Id'], how='left')


####  Calcul de la distance entre les stations  #################################

# coordonnées des stations
coords_fname  = DATA_PATH+ 'Other/stations_coordinates.csv'
coords = pd.read_csv(coords_fname)

# calcule de la matrice de distance en utilisant le module geopy
from geopy import distance, Point

def Distance(stationA, stationB):  #Distance entre deux stations stationA et stationB 
    
    lonA= coords[coords['number_sta']==stationA]['lon'].values
    latA= coords[coords['number_sta']==stationA]['lat'].values
    lonB= coords[coords['number_sta']==stationB]['lon'].values
    latB= coords[coords['number_sta']==stationB]['lat'].values
    PA= Point(latA, lonA)
    PB= Point(latB, lonB)
    return distance.distance(PA,PB).km

def PairwiseDist(ListOfStations): 
    #Calcul la matrice de distance sur les stations considérées (dans ListOfStations)
    nbstations = len(ListOfStations)
    DistanceMatrix= np.zeros((nbstations, nbstations))
    for i in range (nbstations):
        for j in range(nbstations):
            DistanceMatrix[i,j]= Distance(ListOfStations[i], ListOfStations[j])
    D= pd.DataFrame(DistanceMatrix)
    min_dist = 1e-6 
    D.set_axis( ListOfStations, axis=1, inplace=True)
    D.set_axis( D.columns, axis=0, inplace=True)
    D[D < min_dist]= 1e+6 # (pour eviter de considérer qu'une station est voisine d'elle même: distance nulle)
    return D

def StationVoisine(Station, D): 
    #retourne le plus proche voisin de Station en utilisant la matrice de distance  
    #en utilisant D la matrice des distances
    index_neighbor= np.argmin(D.loc[str(Station),:])
    L= D.columns #liste des stations
    return L[index_neighbor]

def CinqStationsVoisines(Station, D): 
    #retourne les cinq voisins les plus proches de Station 
    #en utilisant D la matrice des distances
    L= list(range(5))
    D2= D.copy()
    for i in range(5):
        L[i]=D2.columns[np.argmin(D2.loc[str(Station),:])]
        D2.drop(''+str(L[i]), axis=0, inplace=True)
        D2.drop(''+str(L[i]), axis=1, inplace=True)
    return L

#sauvegarder la matrice des distances car execution longue 

#DStationTrain = PairwiseDist(X_station_date['number_sta'].unique())  
#ListeOfStations = toutes les stations présentes dans X_station_train 
#output_file = "DistanceMatXstationTrain.csv"
#DStationTrain.to_csv('' + output_file,index=False)

# Prochaine Lecture de la matrice des distances 

DXstationTrain= pd.read_csv( DATA_PATH + "PreprocessingData/DistanceMatXstationTrain.csv")
DXstationTrain.set_axis( DXstationTrain.columns, axis=0, inplace=True)

#Remplissage des données matéorologiques manquantes en choisissant la station la plus proche 

def ImputeNaNTrain(dataTrain):
    dataTrain_completed = dataTrain.copy()
    
    for v in ['dd','hu','ff','t','td','Ground_truth', 'Base_Obs','Prediction']: 
        #tableau des NaNs pour la variable v
        NanTab= dataTrain_completed[(dataTrain_completed[v].isna())].reset_index()
        
        #on parcours les lignes ie les stations où il y a presence des NaNs
        for i in range(NanTab.shape[0]): 
            a=np.empty(0)
            Compteur=1
            StationsNA= NanTab['number_sta'].values[i]
            D= DXstationTrain.copy()
            while a.size==0 and Compteur <= 5: #on regarde sur un rayon maximal de cinq voisins
                neighbor= StationVoisine(StationsNA, D)
                a= dataTrain_completed[(dataTrain_completed['number_sta']== neighbor) & 
                                  (dataTrain_completed['date']== NanTab.loc[i,'date'])][v].values
                Compteur+=1
                StationsNA= neighbor
                D.loc[str(StationsNA),str(neighbor)]+=1e+4 # pour eviter de retomber sur les mêmes voisins 
                D.loc[str(neighbor),str(StationsNA)]+=1e+4 # pour eviter de retomber sur les mêmes voisins
                
            if a.size == 0: # cas où les données dans les 5 voisins sont toutes manquantes 
                a= np.append(a,[np.nan],0) #On met un NaN 
            #On remplace le NaN par la nouvelle valeur a dans la base d'origine 
            dataTrain_completed.loc[NanTab['index'].values[i],v]=a 
            
    return dataTrain_completed
    
    
#DataTrain_completed= ImputeNaNTrain()
#output_file = "DataTrain_completed.csv"
#dataTrain_completed.to_csv('' + output_file,index=False)



### ************************* Preprocessing Test *******************
    
#### Chargement des données Test #################################
DATA_PATH = "../data/"

X_station_test = pd.read_csv(DATA_PATH + 'Test/X_station_test.csv')
BaseF_test= pd.read_csv(DATA_PATH + 'Test/Baselines/Baseline_forecast_test.csv')
BaseO_test= pd.read_csv(DATA_PATH +'Test/Baselines/Baseline_observation_test.csv')
BaseO_test.columns=['Id','Base_Obs']

####  Feature engineering  #################################
# Construction des variables moyennes de ff,t,td,hu,dd par jour 

#On utilise l'identifiant pour retrouver le jour 
Identifiants = X_station_test['Id'].values
ListeId=[]
for i in range(len(Identifiants)):
    identif= Identifiants[i].split("_")
    ListeId.append([Identifiants[i], int(identif[0]), int(identif[1]), int(identif[2])])
    
Id_new= pd.DataFrame(ListeId, columns = ['Id', 'Station', 'Day', 'Hour'])
X_station_test= pd.merge(X_station_test, Id_new, on=['Id'])

#### 
X_station_test_date= X_station_test[['Station','Day','Hour','ff','hu','dd','t','td']]
X_station_test_date= X_station_test_date.groupby(['Station','Day']).agg(pd.Series.mean) 
X_station_test_date= X_station_test_date.reset_index(['Station','Day'])
#### Construction d'un nouveau Id station_jour 
X_station_test_date['Id'] = X_station_test_date['Station'].astype(str) + '_' + \
                 X_station_test_date['Day'].astype(str)
X_station_test_date= X_station_test_date[['Station','Day','Id','ff','hu','dd','t','td']]

# ajout des baselines 
data= pd.merge(X_station_test_date, BaseO_test, on=['Id'], how='right')
data= pd.merge(data, BaseF_test, on=['Id'], how= 'left')

#### Construction de clusters de stations #####################

# Distance entre les stations du Test set 

#DXstationTest = PairwiseDist(X_station_test_date['Station'].unique())  
#output_file = "DistanceXStationTest.csv"
#DXstationTest.to_csv('' + output_file,index=False)
DXstationTest= pd.read_csv('DistanceXStationTest.csv')
DXstationTest.set_axis(DXstationTest.columns, axis=0, inplace=True)
MatriceDistanceTest= DXstationTest.to_numpy()
np.fill_diagonal(MatriceDistanceTest, 0)  #on n'a plus besoin d'éviter les 0 sur la diagonale

# Clustering 
from scipy.cluster.hierarchy import ward, dendrogram
linkage_matrix = ward(MatriceDistanceTest) 

## Dendrogramme des barycentres des classes
plt.figure(figsize=(20, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(linkage_matrix,leaf_font_size=8.)
plt.axhline(linestyle='--', y=1500) 
plt.show()

import scipy.cluster.hierarchy as cah
clusters_cah = cah.fcluster(linkage_matrix,t=1500,criterion='distance') #decoupage au niveau 1500
stations = DXstationTest.columns.values
df = pd.DataFrame(np.hstack((stations.reshape(-1,1).astype('int'),
                             clusters_cah.reshape(-1,1))), columns=["Station","Classe"])

data2= pd.merge(data, df)

# On commence par calculer la moyenne par jour sur chaque cluster: 
d= data2[['ff','hu','dd','t','td','Prediction', 'Day', 'Classe']]
ListeJours=  np.sort(d['Day'].unique().astype('int'))
ListeClasses= np.sort(d['Classe'].unique().astype('int'))
L=[]
for jour in ListeJours:
    for classe in ListeClasses:
        liste= [jour, classe ]
        for v in ['ff','hu','dd','t','td','Prediction']:
            mpred= d[(d['Day']==jour) & (d['Classe']==classe)][v].mean()
            liste.append(mpred)
        L.append(liste)
L= pd.DataFrame(L, columns = ['Day', 'Classe','MeanFF','MeanHU','MeanDD','MeanT','MeanTD','MeanPrediction' ])

data3= pd.merge(data2,L, on=['Day', 'Classe'], how='left')

#On veut remplacer les données manquantes par celles les plus récentes pour le même cluster. 
def NearestDate(items, pivot):  #retourne la liste des dates les plus proches de pivot 
    liste=[pivot]
    for i in range(1,len(items)):
        a=pivot+i
        b=pivot- i
        if a <= max(items):
            liste.append(a)
        if b >= min(items):
            liste.append(b)
    return liste

# Imputation des NaNs 

def ImputeNaNTest(data3):
    data_completed= data3.copy()
    for v in ['ff','hu','dd','t','td','Prediction']:
        NanTab= data_completed[(data_completed[v].isna())].reset_index()
        if v== 'ff':
            mv= 'MeanFF'
        if v== 'hu':
            mv= 'MeanHU'
        if v== 'dd':
            mv= 'MeanDD'
        if v== 't':
            mv= 'MeanT'
        if v== 'td':
            mv= 'MeanTD'
        if v== 'Prediction':
            mv= 'MeanPrediction'
    
        for i in range(NanTab.shape[0]):
            JoursProches= NearestDate(ListeJours, NanTab['Day'].values[i])
            valeur= np.nan
            # Règle du défi sur Kaggle: pas de NaNs sur le test set donc on doit toutes les imputer
            while np.isnan(valeur):   
                for j in JoursProches:
                    valeur= data_completed[(data_completed['Day']== j) & 
                                           (data_completed['Classe']== NanTab['Classe'].values[i])][mv].values[0]
                    if np.isnan(valeur)== False:
                        break
                break
            data_completed.loc[NanTab['index'].values[i],v]= valeur
    return data_completed

#data_completed= ImputeNaNTest(data3)
#output_file = "DataTestComplete.csv"
#data_completed.to_csv('' + output_file,index=False)