import warnings
warnings.filterwarnings('ignore')
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.preprocessing as prep
from sklearn.datasets import make_blobs
from plotnine import *   
# StandardScaler is a function to normalize the data 
# You may also check MinMaxScaler and MaxAbsScaler 
#from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import KElbowVisualizer
import scipy.cluster.hierarchy as shc





# helper function that allows us to display data in 2 dimensions an highlights the clusters
def display_cluster(X,km=[],num_clusters=0):
    color = 'brgcmyk'  #List colors
    alpha = 0.5  #color obaque
    s = 20
    if num_clusters == 0:
        plt.scatter(X[:,0],X[:,1],c = color[0],alpha = alpha,s = s)
    else:
        for i in range(num_clusters):
            plt.scatter(X[km.labels_==i,0],X[km.labels_==i,1],c = color[i],alpha = alpha,s=s)
            plt.scatter(km.cluster_centers_[i][0],km.cluster_centers_[i][1],c = color[i], marker = 'x', s = 100)




plt.rcParams['figure.figsize'] = [8,8]
sns.set_style("whitegrid")
sns.set_context("talk")

n_bins = 6  
centers = [(-3, -3), (0, 0), (5,2.5),(-1, 4), (4, 6), (9,7)]
Multi_blob_Data, y = make_blobs(n_samples=[100,150, 300, 400,300, 200], n_features=2, cluster_std=[1.3,0.6, 1.2, 1.7,0.9,1.7],
                  centers=centers, shuffle=False, random_state=42)
display_cluster(Multi_blob_Data)



def KMEANS(data) :
    model = KMeans() 
    visualizer = KElbowVisualizer(model, k=(2,15), metric='distortion', timings=False)
    visualizer.fit(data)    
    visualizer.poof()
    
    n=visualizer.elbow_value_.T
    
    visualizer2 = KElbowVisualizer(model, k=(2,15), metric='silhouette', timings=False)
    visualizer2.fit(data)
     
    elbow_score=visualizer2.elbow_score_.T
    
    
    km =KMeans(n_clusters =n).fit(data)
    plt.figure(figsize=(10, 7))
    plt.scatter(data[:,0], data[:,1], c=km.labels_, cmap='rainbow')
    print("the best value of k is: " +str(n))
    print("the max silhouette score is :  " + str(elbow_score))
  
    return elbow_score




def DBSCANNER(data):

    
    ii=np.arange(0.1,3,0.1)
    jj=range(5,25)
    sil=np.empty((len(ii),len(jj)),float)
    
    a=0
    b=0
    
    
    #x=[]
    #y=[]
    #z=[]
    
    for i in ii :
        
        for j in jj :
            DBScan = DBSCAN(eps=i, min_samples=j).fit(data)
            plt.scatter(data[:,0], data[:,1], c=DBScan.labels_, cmap='rainbow')
            plt.title("eps="+str(i)[0:3]+"   min_samples"+str(j))
            plt.show()
            n=len(set(DBScan.labels_))
            
            if(n>1):
                sil[a][b]=silhouette_score(data,DBScan.labels_)
                #if(n==6) :
                    #x.append(a)
                    #y.append(b)
                    #z.append(silhouette_score(Multi_blob_Data,DBScan.labels_))
                    
            else :
                sil[a][b]=-1
                
            b+=1
    
        a+=1
        b=0
        
    
    
    plt.colorbar(plt.imshow(sil,cmap='viridis'))
    plt.clf()
    
    max_index=np.unravel_index(sil.argmax(), sil.shape)
    
    best_eps=ii[max_index[0]]
    best_min_samples=jj[max_index[1]]
    
    DBScan = DBSCAN(eps=best_eps, min_samples=best_min_samples).fit(data)
    plt.scatter(data[:,0], data[:,1], c=DBScan.labels_, cmap='rainbow')
    plt.title("eps="+str(best_eps)+"   min_samples"+str(best_min_samples))
    plt.show()
    
    max_score=sil[max_index[0]][max_index[1]]

    print("the best value of eps is: " +str(best_eps))
    print("the best value of min_samples is: " +str(best_min_samples))
    print("the max silhouette score is :  " + str(max_score))
  


       
    return max_score    







def Hierarchal_clustering(data) :


    top_scores=[]
    affinity_ziad=['manhattan','cosine','euclidean']
    linkage_ziad=['single','average']
    
    for i in affinity_ziad :
        for j in linkage_ziad :
            
            print("  ")
            print("heyheyhey  " +i +"  " +j)
            plt.figure(figsize=(10, 7))
            plt.title("Dendogram " + i +" " +j)
            
            if(i=='manhattan') :
                dend = shc.dendrogram(shc.linkage(data, method=j,metric='cityblock'))
            else :
                dend = shc.dendrogram(shc.linkage(data, method=j,metric=i))
              
            
            
            score=[]
            ncluster=[]
            
            
            a=np.min(dend['dcoord'])
            b=np.max(dend['dcoord'])
            s=(b-a)/10
            
            
            
            
            for c in np.arange(a+s,b,s):
                cluster = AgglomerativeClustering(n_clusters=None, affinity=i, linkage=j,distance_threshold=c)
                cluster.fit_predict(data)
                ncluster.append(len(np.unique(cluster.labels_)))
                score.append(silhouette_score(data, cluster.labels_, metric=i))
                plt.figure(figsize=(10, 7))
                plt.title("figure " + i +" " +j +" distance_threshold: "+str(c))
                plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
                
            
            bestsil=max(score)
            print("the best silhouette score is:" +' '+str(bestsil)+"  " +i +"  "+j)
            print("The number of clusters is "+str(ncluster[score.index(max(score))]))
            print("the distance threshold is : " +str(a+s+s*score.index(max(score))))
            top_scores.append(bestsil)

    return top_scores



# 'full' (each component has its own general covariance matrix) 
# 'tied' (all components share the same general covariance matrix)
# 'diag' (each component has its own diagonal covariance matrix)
# 'spherical' (each component has its own single variance)

def Gaussian(data):
    
    types=['full','tied','diag','spherical']
    score=-100
    for i in types :
        for j in range(2,10):
            print(" ")
            print(j)
            GM = GaussianMixture(n_components=j,covariance_type=i).fit(data)
            plt.scatter(data[:, 0], data[:, 1], c=GM.predict(data), cmap='rainbow');
            X, Y = np.meshgrid(np.linspace(-6,15), np.linspace(-6,15))
            XX = np.array([X.ravel(), Y.ravel()]).T
            Z = GM.score_samples(XX)
            Z = Z.reshape((50,50))
            plt.title(i +" n_components: "+str(j))
            plt.contour(X, Y, Z,levels=50) 
            plt.scatter(data[:, 0], data[:, 1], c=GM.predict(data), cmap='rainbow');
            plt.show()
            
 
    
    return score











