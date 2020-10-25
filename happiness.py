import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


if __name__ == '__main__':
    df = pd.read_csv('data/happiness/2016.csv')
    print(df.columns)
    print(len(pd.unique(df['Country'])))
    happinessScore = df[['Country', 'Happiness Score']]
    gdpByHappiness = df[['Country', 'Happiness Score', 'Economy (GDP per Capita)']]
    print(gdpByHappiness)
    plt.figure(figsize=(9, 8))
    sns.scatterplot(data=gdpByHappiness, x="Happiness Score", y=gdpByHappiness['Economy (GDP per Capita)'].values,
                    hue=df.Country)
    #markers.MarkerStyle.markers.keys()
    #Best fit so far
    #plt.legend(loc=3, bbox_to_anchor=(0, 1, 0.4, 0.4), fontsize='xx-small', ncol=8)
    plt.legend(bbox_to_anchor=(0, 1), loc="lower left", ncol=5, fontsize='xx-small')
    plt.xlabel('Happiness Score')
    plt.ylabel('GDP per capita')
    fig = plt.gcf()
    #fig.set_size_inches(4, 12)
    plt.savefig("data/results/happiness/HappinessVsGDPWithLegend.png", bbox_inches="tight")
    plt.show()

    # scaling values for elbow method
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(gdpByHappiness[['Happiness Score', 'Economy (GDP per Capita)']])

    #Elbow Method

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(gdpByHappiness[['Happiness Score', 'Economy (GDP per Capita)']])
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig("data/results/happiness/ElbowMethod.png")
    plt.show()


    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(gdpByHappiness[['Happiness Score', 'Economy (GDP per Capita)']])
    sns.scatterplot(data=gdpByHappiness, x="Happiness Score", y=gdpByHappiness['Economy (GDP per Capita)'].values,
                    hue=df.Country, legend=False)
    sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, color='red')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Happiness Score')
    plt.ylabel('GDP per capita')
    plt.savefig("data/results/happiness/HappinessVsGDPPerCapitaWithClusters.png")
    plt.show()





