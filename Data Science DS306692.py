#Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import sklearn.model_selection as ms

import missingno as msno

import plotly
import plotly.graph_objects as go
import plotly.express as px

import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import dendrogram
from scipy import stats

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

pd.set_option("display.max_columns", None)
pd.set_option('display.max_colwidth', None)

#Reading the datafile
df = pd.read_csv("tripadvisor_european_restaurants.csv", low_memory=False)
#Leaving only my country
df=df.drop(df[df.country!="Poland"].index)

# Not important columns are dropped
df=df.drop(columns=['country','latitude','longitude','restaurant_link','restaurant_name','original_location','region','address','city','price_level','popularity_detailed','top_tags','original_open_hours','default_language','reviews_count_in_default_language','atmosphere','keywords'])

# average price in euro
df['minimum_range'] = df['price_range'].str.split('-').str[0].str.replace('€', '').str.replace(',', '')
df['minimum_range'] = pd.to_numeric(df['minimum_range'], errors='coerce')
df['maximum_range'] = df['price_range'].str.split('-').str[1].str.replace('€', '').str.replace(',', '')
df['maximum_range'] = pd.to_numeric(df['maximum_range'], errors='coerce')
df['average_price'] = (df['minimum_range'] + df['maximum_range']) / 2

# drop the fields used for average_price calculation
df=df.drop(columns=['minimum_range', 'maximum_range','price_range'])

#fill nan's
df=df.fillna({'value':df["value"].mean()})
df=df.fillna({'open_days_per_week':df["open_days_per_week"].mean()})
df=df.fillna({'open_hours_per_week':df["open_hours_per_week"].mean()})
df=df.fillna({'working_shifts_per_week':df["working_shifts_per_week"].mean()})
df=df.fillna({'food':df["food"].mean()})
df=df.fillna({'service':df["service"].mean()})
df=df.fillna({'excellent':0.0})
df=df.fillna({'very_good':0.0})
df=df.fillna({'average':0.0})
df=df.fillna({'poor':0.0})
df=df.fillna({'terrible':0.0})
df=df.fillna({'average_price':0.0})

df=df.fillna({'avg_rating':0.0})
df=df.fillna({'total_reviews_count':0.0})

#Check
df.head(5)

msno.matrix(df, fontsize=20)

df['province']=df[['province']].fillna(value='Unknown')

#Mapping
x=df['province'].value_counts()
item_type_mapping={}
item_list=x.index
for i in range(0,len(item_list)):
    item_type_mapping[item_list[i]]=i

df['province']=df['province'].map(lambda x:item_type_mapping[x]) 
df.head(5)

#We can found how many other places there are in the same city
def findLargestNumber(text):
    ls = list()
    for w in text.split():
        try:
            ls.append(float(w))
        except:
            pass
    try:
        return max(ls)
    except:
        return None

df['popularity_generic']=df[['popularity_generic']].fillna(value='Unknown')
df.popularity_generic = [findLargestNumber(item) for item in df.popularity_generic]
df['popularity_generic']=df[['popularity_generic']].fillna(value=1)
df=df.rename( columns={'popularity_generic' : 'one_of_x_in_city'})
df.head(5)

df['meals']=df[['meals']].fillna(value='')
df.meals=[item.count(',')+1 if item!='' else 0 for item in df.meals]

df['awards']=df[['awards']].fillna(value='')
df.awards=[item.count(',')+1 if item!='' else 0 for item in df.awards]
 
df['features']=df[['features']].fillna(value='')
df.features=[item.count(',')+1 if item!='' else 0 for item in df.features]
    
df['cuisines']=df[['cuisines']].fillna(value='')
df.cuisines=[item.count(',')+1 if item!='' else 0 for item in df.cuisines]
 
vege = {'N': 0,'Y': 1}
claimed = {'Unclaimed': 0,'Claimed': 1}

df.vegetarian_friendly = [vege[item] for item in df.vegetarian_friendly]
df.vegan_options = [vege[item] for item in df.vegan_options]
df.gluten_free = [vege[item] for item in df.gluten_free]

df['claimed']=df[['claimed']].fillna(value='Unclaimed')
df.claimed=[claimed[item] for item in df.claimed]

df['special_diets']=df[['special_diets']].fillna(value='Unknown')
df.special_diets = [item.count("Unknown") for item in df.special_diets]

specialDiet = {0: 1,1: 0}
df.special_diets = [specialDiet[item] for item in df.special_diets]


df.head(15)

msno.matrix(df, fontsize=20)

#2
df.describe()
df.mode()
df.median()

#3 Detecting outliers
def IQR_outliner_detection(var):
    q_1,q_3=np.percentile(var,[25,75])
    IQR=q_3-q_1
    lower_bound=q_1-(IQR*1.5)
    upper_bound=q_3+(IQR*1.5)
    return np.where((var>upper_bound) | (var<lower_bound))

pom=IQR_outliner_detection(df.province)[0]
OutliersDF = pd.DataFrame(pom, columns=['province'])

pom=IQR_outliner_detection(df.claimed)[0]
OutliersDF2=pd.DataFrame(pom, columns=['claimed'])

pom=IQR_outliner_detection(df.awards)[0]
OutliersDF3=pd.DataFrame(pom, columns=['awards'])

pom=IQR_outliner_detection(df.one_of_x_in_city)[0]
OutliersDF4=pd.DataFrame(pom, columns=['one_of_x_in_city'])

pom=IQR_outliner_detection(df.meals)[0]
OutliersDF5=pd.DataFrame(pom, columns=['meals'])

pom=IQR_outliner_detection(df.cuisines)[0]
OutliersDF6=pd.DataFrame(pom, columns=['cuisines'])

pom=IQR_outliner_detection(df.special_diets)[0]
OutliersDF7=pd.DataFrame(pom, columns=['special_diets'])

pom=IQR_outliner_detection(df.vegetarian_friendly)[0]
OutliersDF8=pd.DataFrame(pom, columns=['vegetarian_friendly'])

pom=IQR_outliner_detection(df.open_days_per_week)[0]
OutliersDF9=pd.DataFrame(pom, columns=['open_days_per_week'])

pom=IQR_outliner_detection(df.open_hours_per_week)[0]
OutliersDF10=pd.DataFrame(pom, columns=['open_hours_per_week'])

pom=IQR_outliner_detection(df.working_shifts_per_week)[0]
OutliersDF11=pd.DataFrame(pom, columns=['working_shifts_per_week'])

pom=IQR_outliner_detection(df.avg_rating)[0]
OutliersDF12=pd.DataFrame(pom, columns=['avg_rating'])

pom=IQR_outliner_detection(df.total_reviews_count)[0]
OutliersDF13=pd.DataFrame(pom, columns=['total_reviews_count'])

pom=IQR_outliner_detection(df.excellent)[0]
OutliersDF14=pd.DataFrame(pom, columns=['excellent'])

pom=IQR_outliner_detection(df.very_good)[0]
OutliersDF15=pd.DataFrame(pom, columns=['very_good'])

pom=IQR_outliner_detection(df.average)[0]
OutliersDF16=pd.DataFrame(pom, columns=['average'])

pom=IQR_outliner_detection(df.poor)[0]
OutliersDF17=pd.DataFrame(pom, columns=['poor'])

pom=IQR_outliner_detection(df.terrible)[0]
OutliersDF18=pd.DataFrame(pom, columns=['terrible'])

pom=IQR_outliner_detection(df.food)[0]
OutliersDF19=pd.DataFrame(pom, columns=['food'])

pom=IQR_outliner_detection(df.service)[0]
OutliersDF20=pd.DataFrame(pom, columns=['service'])

pom=IQR_outliner_detection(df.value)[0]
OutliersDF21=pd.DataFrame(pom, columns=['value'])

pom=IQR_outliner_detection(df.average_price)[0]
OutliersDF22=pd.DataFrame(pom, columns=['average_price'])

pom=IQR_outliner_detection(df.awards)[0]
OutliersDF23=pd.DataFrame(pom, columns=['features'])

pom=IQR_outliner_detection(df.awards)[0]
OutliersDF24=pd.DataFrame(pom, columns=['vegan_options'])

pom=IQR_outliner_detection(df.awards)[0]
OutliersDF25=pd.DataFrame(pom, columns=['gluten_free'])


result=pd.concat([OutliersDF,OutliersDF2,OutliersDF3,OutliersDF4,
                 OutliersDF5,OutliersDF6,OutliersDF7,OutliersDF8,
                 OutliersDF9,OutliersDF10,OutliersDF11,OutliersDF12,
                 OutliersDF13,OutliersDF14,OutliersDF15,OutliersDF16,
                 OutliersDF17,OutliersDF18,OutliersDF19,OutliersDF20,
                 OutliersDF21,OutliersDF22,OutliersDF23,OutliersDF24,
                 OutliersDF25],axis=1)
result

#4
dend = shc.dendrogram(shc.linkage(df, method='ward'))

#5
x=df.drop(columns=['avg_rating'])
pca = PCA(n_components=2)
princComp=pca.fit_transform(x)
print(princComp)

tempDf = pd.DataFrame(princComp,columns = ['0','1'])

finalDF = pd.concat([tempDf,df[['avg_rating']]],axis =1)

groups = finalDF.groupby("avg_rating")

plt.rcParams["figure.figsize"] = (10,10)#Plot size

for name, group in groups:
    plt.plot(group["0"],group["1"],marker="o",linestyle="",label=name)

plt.legend()

#6
df=df.rename( columns={'avg_rating' : 'target'})
abs(df.corr().sort_values(by='target'))

#7
df=df.drop(columns=['excellent','very_good','average','poor','terrible'])
train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('target')
test_labels = test_features.pop('target')

normalizer = preprocessing.Normalization()

normalizer.adapt(np.array(train_features))

def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation="selu"),
        layers.Dense(64, activation="selu"),
        layers.Dense(64, activation="selu"),
        layers.Dense(64, activation="selu"),
        layers.Dense(1)
    ])
    
    model.compile(
        #loss='mean_absolute_percentage_error',
        loss='mean_absolute_error',
        optimizer = tf.keras.optimizers.Adam(0.002))
    return model

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_features, train_labels,
    validation_split = 0.2,
    verbose=1, epochs = 100)

def plot_loss(history):
    plt.plot(history.history['loss'], label = 'loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.ylim([0, 1.0])
    plt.xlabel('Epoch')
    plt.ylabel('Error [quality]')
    plt.legend()
    plt.grid(True)
    
plot_loss(history)

test_predictions = dnn_model.predict(test_features).flatten()

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error')
_ = plt.ylabel('Count')