import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    df = pd.read_csv('data/happiness/2016.csv')
    print(df.columns)
    print(np.shape(df))
    df_a = df.drop(['Happiness Score'], axis=1)
    df = df.drop(['Happiness Rank', 'Lower Confidence Interval', 'Upper Confidence Interval', 'Country', 'Region'], axis=1)
    correlation_matrix = df.corr()
    happiness_iv = df.corr()[['Happiness Score']].sort_values(by='Happiness Score', ascending=False)

    print(correlation_matrix)
    plt.figure(figsize=(10, 6))
    mask = np.triu(np.ones_like(correlation_matrix))
    heatmap = sns.heatmap(correlation_matrix, annot=True, vmin=-1, vmax=1, cmap='BrBG', mask=mask)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
    plt.savefig('data/results/happiness/CorrelationMatrix.png')
    plt.show()

    heatmap = sns.heatmap(happiness_iv, vmin=-1,
                          vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Features correlating with Happiness Score',  fontdict={'fontsize': 12}, pad=12)
    plt.savefig('data/results/happiness/FeaturesCorrelatingHappiness.png')
    plt.show()
