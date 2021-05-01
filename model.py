# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:29:55 2021

@author: Yassin Fahmy
"""

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import accuracy_score,plot_roc_curve, auc, roc_auc_score, roc_curve, precision_score, recall_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
import matplotlib.cm as cm


#Read csv file
data=pd.read_csv('heart_failure_clinical_records_dataset.csv')
# creat a summary object to check for data distribution
summary=data.describe()
#plot frequency distribution
data.hist()
#correlation plot
scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='hist')

#make a list of column names
col=data.columns.values
names=[]
#indicate the columns that will need a power transformation to correct skewness with a prefix 't'
for i in np.array([2,4,6,7,8]):
    names.append(col[i])
names=['t'+ i for i in names]

#power transform some columns
pt = PowerTransformer(method='box-cox')
tData = pd.DataFrame(pt.fit_transform(data.iloc[:,[2,4,6,7,8]]),columns=names)

#visualize data ditribution after transformations
tData.hist()
tsummary=tData.describe()

#remove untransformed columns
for i in np.array([2,4,6,7,8]):
    data.pop(col[i])
#bin age data
data['age_by_decade'] = pd.cut(x=data['age'], bins=[39, 49, 59, 69,79,89,99], labels=['40s', '50s', '60s', '70s', '80s', '90s'])
data.pop('age')
data.pop('time')

#dummy coding categorical variables
dData=data.copy()
dData.pop('DEATH_EVENT')
dData=pd.get_dummies(dData)

#join transformed continous data and categorical data
df=pd.concat([dData,tData,data.iloc[:,5]],axis=1)

#visualize data distribution in each outcome class
df.groupby('DEATH_EVENT').hist()

#################################################################################
# set input and target variables
y=df.iloc[:,16].copy()
x=df.iloc[:,:16].copy()
#split data into training and testing cohorts
xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.2,shuffle=True)


#################################################################################
#ELastic net model
#10 fold cross validation grid search to find optimum l1-ratio
logModel = LogisticRegression(penalty='elasticnet',solver='saga',random_state=40,max_iter=1000)
cv=RepeatedKFold(n_splits=10, n_repeats=3,random_state=40)
grid=dict()
# make a list of l1_ratio to search accross
grid['l1_ratio'] = np.arange(0, 1, 0.01)
search = GridSearchCV(logModel, grid,cv=cv,scoring='balanced_accuracy', n_jobs=-1)
results=search.fit(xTrain,yTrain)
l1_ratio=results.best_params_['l1_ratio']
#build log model based on optimum l1-ratio
log=LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=l1_ratio,random_state=40,max_iter=1000)
log.fit(xTrain,yTrain)
#predict validation set
yPredLog=log.predict(xTest)

acc=accuracy_score(yTest,yPredLog)
precision=precision_score(yTest,yPredLog)
recall=recall_score(yTest,yPredLog)
fpr, tpr ,_=roc_curve(yTest,yPredLog)
AUC =roc_auc_score(yTest, yPredLog)

plt.figure()
plt.plot(fpr,tpr)
plt.title("ROC for Logestic regression model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

print("AUC for logestic regression model:",AUC)
print("Accuracy for logestic regression model:",acc)
print("Precision for logestic regression model:",precision,"Recall for logestic regression model:",recall)

coef=pd.DataFrame(log.coef_.transpose().flatten(),columns=['coef'])
colNames=pd.DataFrame(x.columns.values.tolist(),columns=['Names'])
coef=pd.concat((colNames,coef),axis=1)
#rename some variables for bar chart
coef.iloc[3,0]=['Sex Male']
coef.iloc[5,0]=['Age = 40s']
coef.iloc[6,0]=['Age = 50s']
coef.iloc[7,0]=['Age = 60s']
coef.iloc[8,0]=['Age = 70s']
coef.iloc[9,0]=['Age = 80s']
coef.iloc[10,0]=['Age = 90s']
coef.iloc[11,0]=['Creatine Phosphokinase level']
coef.iloc[12,0]=['Ejection Fraction']
coef.iloc[13,0]=['Platlets count']
coef.iloc[14,0]=['Serum Creatinine levels']
coef.iloc[15,0]=['Serum sodium']

plt.barh(coef.iloc[:,0],coef.iloc[:,1],align='center')
plt.title('Logestic regression coef')
plt.ylabel('Variables')
plt.xlabel('Coef')
plt.show()

########################################################################################
####build Random forest model
rf=RandomForestClassifier(n_jobs=-1,max_depth=7,random_state=40)
rf.fit(xTrain, yTrain)
yPredRf = rf.predict(xTest)

featureImp=pd.DataFrame(rf.feature_importances_,columns=['Feature importance'])
featureImp=pd.concat((colNames,featureImp),axis=1)
#rename some variables for bar chart
featureImp.iloc[3,0]=['Sex Male']
featureImp.iloc[5,0]=['Age = 40s']
featureImp.iloc[6,0]=['Age = 50s']
featureImp.iloc[7,0]=['Age = 60s']
featureImp.iloc[8,0]=['Age = 70s']
featureImp.iloc[9,0]=['Age = 80s']
featureImp.iloc[10,0]=['Age = 90s']
featureImp.iloc[11,0]=['Creatine Phosphokinase level']
featureImp.iloc[12,0]=['Ejection Fraction']
featureImp.iloc[13,0]=['Platlets count']
featureImp.iloc[14,0]=['Serum Creatinine levels']
featureImp.iloc[15,0]=['Serum sodium']

plt.barh(featureImp.iloc[:,0],featureImp.iloc[:,1],align='center')
plt.title('Random forest features importance')
plt.ylabel('Variables')
plt.xlabel('Feature value')
plt.show()


acc=accuracy_score(yTest,yPredRf)
precision=precision_score(yTest,yPredRf)
recall=recall_score(yTest,yPredRf)
fpr, tpr ,_=roc_curve(yTest,yPredRf)
AUCrf =roc_auc_score(yTest, yPredRf)

plt.figure()
plt.plot(fpr,tpr)
plt.title("ROC for Random Forest model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

print("AUC for Random Forest model:",AUCrf)
print("Accuracy for Random Forest model:",acc)
print("Precision for Random Forest model:",precision,"Recall for Random Forest model:",recall)

####################################################################################
### MLP model

hl=[200]*6
clf = MLPClassifier(random_state=40, max_iter=2000,hidden_layer_sizes=(hl))
clf.fit(xTrain, yTrain)
yPredclf=clf.predict(xTest)

acc=accuracy_score(yTest,yPredRf)
precision=precision_score(yTest,yPredRf)
recall=recall_score(yTest,yPredRf)
fpr, tpr ,_=roc_curve(yTest,yPredRf)
AUCclf =roc_auc_score(yTest, yPredRf)

plt.figure()
plt.plot(fpr,tpr)
plt.title("ROC for MLP")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

print("AUC for MLP:",AUCclf)
print("Accuracy for MLP:",acc)
print("Precision for MLP:",precision,"Recall for MLP:",recall)


#####################################################################################
#### K_means clustering
#define some variables
clusters                =[]
inertia                 =[]
silhouette_coefficients =[]

#try different number of clusters
for i in np.arange(2,7,1):
    km              = KMeans(n_clusters=i, random_state=42).fit(x)
    inertia.append(km.inertia_)
    silhouette_coefficients.append(silhouette_score(x, km.labels_))
    
#plot Sum of squared distances of samples to their closest cluster center vs # of clusters
plt.plot(np.arange(2,7,1),inertia,color='r',marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()
plt.plot(np.arange(2,7,1),silhouette_coefficients,color='g',marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.show()

range_n_clusters = [2, 3, 4, 5, 6]
silhouette_avg_n_clusters = []

for n_clusters in range_n_clusters:
    fig, ax1 = plt.subplots(1)
    fig.set_size_inches(18, 7)

    # Initialize the clusterer with n_clusters value and a random generator
    # seed for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(x)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed clusters
    silhouette_avg = silhouette_score(x, cluster_labels)

    silhouette_avg_n_clusters.append(silhouette_avg)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(x, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

plt.show()

#chosen model of 3 clusters based on visual inspection of silhoutte plots
km              = KMeans(n_clusters=3, random_state=42).fit(x)
clusters        =km.labels_
kmDf=pd.DataFrame(np.concatenate((np.array(df),np.reshape(clusters,(len(clusters),1))),axis=1))

#separate clusters to run statistics
kmClust1=kmDf.loc[clusters==0]
kmClust2=kmDf.loc[clusters==1]
kmClust3=kmDf.loc[clusters==2]
# clust1Counts=[]
# clust2Counts=[]
# clust3Counts=[]
# for i in np.arange(0,11):
#     clust1Counts.append(kmClust1.iloc[:,i].value_counts())
# clust1Counts.append(kmClust1.iloc[:,16].value_counts())
# for i in np.arange(0,11):
#     clust2Counts.append(kmClust2.iloc[:,i].value_counts())
# clust2Counts.append(kmClust2.iloc[:,16].value_counts())
# for i in np.arange(0,11):
#     clust3Counts.append(kmClust3.iloc[:,i].value_counts())
# clust3Counts.append(kmClust3.iloc[:,16].value_counts())
labels=['Anaemia','Diabetes','High BP','Sex Male','Smoking',"Age 40s",'Age 50s','Age 60s','Age 70s','Age 80s','Age 90s','Death Event']
c=np.empty([3,len(labels)])
for i in range(3):
    c[i]=[
    sum(df.loc[clusters==i,'anaemia']),\
    sum(df.loc[clusters==i,'diabetes']),\
    sum(df.loc[clusters==i,'high_blood_pressure']),\
    sum(df.loc[clusters==i,'sex']),\
    sum(df.loc[clusters==i,'smoking']),\
    sum(df.loc[clusters==i,'age_by_decade_40s']),\
    sum(df.loc[clusters==i,'age_by_decade_50s']),\
    sum(df.loc[clusters==i,'age_by_decade_60s']),\
    sum(df.loc[clusters==i,'age_by_decade_70s']),\
    sum(df.loc[clusters==i,'age_by_decade_80s']),\
    sum(df.loc[clusters==i,'age_by_decade_90s']),\
    sum(df.loc[clusters==i,'DEATH_EVENT'])\
    ]

width=0.2
x=np.arange(len(labels))
f, ax =plt.subplots()
r1=ax.bar(x - width,c[0],width,label='Cluster 1')
r2=ax.bar(x ,c[1],width,label='Cluster 2')
r3=ax.bar(x + width,c[2],width,label='Cluster 3')

ax.set_ylabel('Counts')
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.xticks(rotation=70)
ax.legend()
plt.show()



#show characteristics of each cluster