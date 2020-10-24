import pandas as pd
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

def read_dataset(path):
    return pd.read_excel(path, index_col=0)


def readBulliedStats(dataset):
    return dataset.loc[:, ["Bullied (N = 14) M", "Bullied (N = 14) SD"]]


def readWitnessStats(dataset):
    return dataset.loc[:, ["Witness (N = 31) M", "Witness (N = 31) SD"]]


def readReferentsStats(dataset):
    return dataset.loc[:, ["Referents (N = 202) M", "Referents (N = 202) SD"]]

def plot2Aspects(dataset, n_samples, ):
    print("kkkk")


# Creates samples based on the mean, standard deviation and
# total number of people on the group (1 group at a time)
def createSamplesForEachGroup(dataset, n, index_row1, index_row2):
    df1_meanColumnName = dataset.columns[0]
    df_1_mean = dataset.loc[[index_row1], df1_meanColumnName]
    df1_SDColumnName = dataset.columns[1]
    df_1_sd = dataset.loc[[index_row1], df1_SDColumnName]

    df2_meanColumnName = dataset.columns[0]
    df_2_mean = dataset.loc[[index_row2], df2_meanColumnName]
    df2_SDColumnName = dataset.columns[1]
    df_2_sd = dataset.loc[[index_row2], df2_SDColumnName]
    samples_df_x = generateSamples(n_samples=n, desired_mean=float(df_1_mean.values), desired_std_dev=float(df_1_sd.values))
    samples_df_y = generateSamples(n_samples=n, desired_mean=float(df_2_mean.values), desired_std_dev=float(df_2_sd.values))
    return [samples_df_x, samples_df_y]

#Solução encontrada em https://stackoverflow.com/a/51515765
def generateSamples(n_samples, desired_mean, desired_std_dev):
    print("Number of samples: ", n_samples)
    print("Desired mean: ", desired_mean)
    print("Desired SD: ", desired_std_dev)

    samples = np.random.normal(loc=0.0, scale=desired_std_dev, size=n_samples)

    actual_mean = np.mean(samples)
    actual_std = np.std(samples)
    print("Initial samples stats   : mean = {:.4f} stdv = {:.4f}".format(actual_mean, actual_std))

    zero_mean_samples = samples - (actual_mean)

    zero_mean_mean = np.mean(zero_mean_samples)
    zero_mean_std = np.std(zero_mean_samples)
    print("True zero samples stats : mean = {:.4f} stdv = {:.4f}".format(zero_mean_mean, zero_mean_std))

    scaled_samples = zero_mean_samples * (desired_std_dev / zero_mean_std)
    scaled_mean = np.mean(scaled_samples)
    scaled_std = np.std(scaled_samples)
    print("Scaled samples stats    : mean = {:.4f} stdv = {:.4f}".format(scaled_mean, scaled_std))

    final_samples = scaled_samples + desired_mean
    final_mean = np.mean(final_samples)
    final_std = np.std(final_samples)
    print("Final samples stats     : mean = {:.4f} stdv = {:.4f}".format(final_mean, final_std))
    return final_samples


if __name__ == '__main__':
    # 1st step: Generate samples for each of the 3 groups: Bullied, Witness and Referents
    # 1.1 step: Select the 2 aspects from SSP and JCQ questionnaire that you want to measure
    # 2nd step: After generating the samples, display the points in a graph
    # Do K-Means Clustering
    # Already known group data
    total = 247
    n_bullied = 14
    n_witness = 31
    n_referents = 202
    dataset = read_dataset("data/table3.xlsx")
    bullied_data = readBulliedStats(dataset)
    witness_data = readWitnessStats(dataset)
    referents_data = readReferentsStats(dataset)
    aspectoSSP = "Mistrust"
    aspectoJCQ = "Support from managers"

    [bullied_samples_x, bullied_samples_y] = createSamplesForEachGroup(n=n_bullied, dataset=bullied_data, index_row1=aspectoSSP,
                                                index_row2=aspectoJCQ)
    [witness_samples_x, witness_samples_y] = createSamplesForEachGroup(n=n_witness, dataset=witness_data, index_row1=aspectoSSP,
                                                index_row2=aspectoJCQ)
    [referents_samples_x, referents_samples_y] = createSamplesForEachGroup(n=n_referents, dataset=referents_data,
                                                  index_row1=aspectoSSP,
                                                  index_row2=aspectoJCQ)

    print("Bullied Samples x: ", bullied_samples_x)
    print("Bullied Samples y: ", bullied_samples_y)
    print("Witness Samples x: ", witness_samples_x)
    print("Witness Samples y: ", witness_samples_y)
    print("Referents Samples x: ", referents_samples_x)
    print("Referents Samples y: ", referents_samples_y)


    ax = plt.subplots(1, 1)
    ax = sns.scatterplot(x=bullied_samples_x, y=bullied_samples_y, color='red', alpha=0.5)
    ax = sns.scatterplot(x=witness_samples_x, y=witness_samples_y, color='blue', alpha=0.5)
    ax = sns.scatterplot(x=referents_samples_x, y=referents_samples_y, color='green', alpha=0.5)
    ax.plot()
    plt.show()

