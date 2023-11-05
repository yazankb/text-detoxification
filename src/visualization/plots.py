import matplotlib.pyplot as plt
import seaborn as sns

def plotWordCountFrequency(dataframe):
    """
    Visualizes the word count distribution for the 'reference' and 'translation' columns in the DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing 'reference' and 'translation' columns.

    This function calculates and displays histograms of word counts for both 'reference' and 'translation' columns.
    It also prints the maximum word counts for reference and translation.

    Example Usage:
    plotWordCountFrequency(df)
    """
    def count_words(text):
        words = text.split()
        return len(words)
    dataframe['translation_word_count'] = dataframe['translation'].apply(count_words)
    dataframe['reference_word_count'] = dataframe['reference'].apply(count_words)
    print('Max number of words in reference :', max(dataframe['reference_word_count']))
    print('Max number of words in translation :', max(dataframe['translation_word_count']))
    

    # Create subplots for reference and translation word counts
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(dataframe['reference_word_count'], bins=30, color='blue', alpha=0.7)
    plt.title('Reference Word Count')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(dataframe['translation_word_count'], bins=30, color='green', alpha=0.7)
    plt.title('Translation Word Count')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def plotRefToxDistribution(dataframe):
    """
    Visualizes the distribution of 'ref_tox' values in the DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the 'ref_tox' column.

    This function displays a histogram to visualize the distribution of 'ref_tox' values.

    Example Usage:
    plotRefToxDistribution(df)
    """
    plt.figure(figsize=(6, 5))
    plt.hist(dataframe['ref_tox'], bins=60, color='blue', alpha=0.7)
    plt.title('Distribution of ref_tox')
    plt.xlabel('ref_tox Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plotTrnToxDistribution(dataframe):
    """
    Visualizes the distribution of 'ref_tox' values in the DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the 'ref_tox' column.

    This function displays a histogram to visualize the distribution of 'ref_tox' values.

    Example Usage:
    plotRefToxDistribution(df)
    """
    plt.figure(figsize=(6, 5))
    plt.hist(dataframe['trn_tox'], bins=60, color='green', alpha=0.7)
    plt.title('Distribution of trn_tox')
    plt.xlabel('trn_tox Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plotSimilarityDistribution(dataframe):
    """
    Visualizes the distribution of 'similarity' values in the DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the 'similarity' column.

    This function displays a histogram to visualize the distribution of 'similarity' values.

    Example Usage:
    plotSimilarityDistribution(df)
    """
    plt.figure(figsize=(6, 5))
    plt.hist(dataframe['similarity'], bins=60, color='orange', alpha=0.7)
    plt.title('Distribution of similarity')
    plt.xlabel('Similarity Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plotLengthDiffDistribution(dataframe):
    """
    Visualizes the distribution of 'similarity' values in the DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the 'similarity' column.

    This function displays a histogram to visualize the distribution of 'similarity' values.

    Example Usage:
    plotSimilarityDistribution(df)
    """
    plt.figure(figsize=(6, 5))
    plt.hist(dataframe['lenght_diff'], bins=60, color='red', alpha=0.7)
    plt.title('Distribution of length_diff')
    plt.xlabel('Length Difference')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def visualizeCorrelationHeatmap(dataframe, numerical_columns):
    """
    Visualizes the correlation heatmap for selected numerical columns in the DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        numerical_columns (list): A list of numerical column names for correlation analysis.
    """
    # Calculate the correlation matrix
    correlation_matrix = dataframe[numerical_columns].corr()

    # Create a heatmap to visualize the correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.show()