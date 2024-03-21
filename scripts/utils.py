import matplotlib.pyplot as plt
from logger import logger
from dotenv import load_dotenv,find_dotenv
from openai import OpenAI
import pandas as pd

load_dotenv(find_dotenv())

def plot_bar_graph(df:pd.DataFrame, column:str)-> None:
    """
    Plots a bar graph showing the distribution of values in the specified column of the DataFrame.
    
    Parameters:
        df (DataFrame): The pandas DataFrame containing the data.
        column (str): The name of the column to plot.
    """
    try:
        # Count the occurrences of each unique value in the column
        value_counts = df[column].value_counts(dropna=False)

        # Plotting
        plt.figure(figsize=(8, 6))
        value_counts.plot(kind='bar', color='skyblue')
        plt.title(f'Distribution of {column} Values')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

        '''
        # Plotting
plt.figure(figsize=(10, 6))
top_browsers.plot(kind='bar', color='skyblue')
plt.title('Top 10 Browsers')
plt.xlabel('Browser')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
'''


    
    except KeyError as e:
        logger.error(f"Error: Column '{column}' not found in the DataFrame. error: {e}")



def plot_histogram(df:pd.DataFrame, column:str)-> None:
    """
    Plots a histogram showing the distribution of values in the specified column of the DataFrame.
    
    Parameters:
        df (DataFrame): The pandas DataFrame containing the data.
        column (str): The name of the column to plot.
    """
    try:
        # Plotting
        plt.figure(figsize=(8, 6))
        plt.hist(df, bins=20, color="skyblue", edgecolor="black")
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

  
    
    except KeyError as e:
        logger.error(f"Error: Column '{column}' not found in the DataFrame. error: {e}")


def plot_pie_chart(df:pd.DataFrame, column:str)-> None:
    """
    Plots a pie chart showing the distribution of values in the specified column of the DataFrame.
    
    Parameters:
        data (Series): The pandas Series containing the data.
        column (str): The name of the column to plot.
    """
    try:
        # Count the occurrences of each unique value in the column
        value_counts = df.value_counts(dropna=False)

        # Plotting
        plt.figure(figsize=(8, 8))
        value_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
        plt.title(f'Distribution of {column}')
        plt.ylabel('')
        plt.show()
    
    except KeyError as e:
        logger.error(f"Error: Column '{column}' not found in the DataFrame. Error: {e}")


def get_chat_completion(prompt: str) -> str:
    """
    Generates a chat completion response using OpenAI's GPT-3 model.
    
    Parameters:
        prompt (str): The prompt message for the conversation.
    
    Returns:
        str: The completion response from the GPT-3 model.
    """
    try:
        # Importing necessary library
        
        # Creating OpenAI client
        client = OpenAI()

        # Generating chat completion response
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )

        completion_response = response.choices[0].message.content
        
        # logger
        logger.info("chat completion success")

        return completion_response

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return ""

def plot_top_items(df: pd.DataFrame, group_column: str, metric_column: str, sort_by:str, n: int = 10, ascending: bool = False) -> None:
    """
    Plots the top items based on the specified metric.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        group_column (str): The column to group by.
        metric_column (str): The column representing the metric to sort by.
        sort_by(str): a column in which it is going to be sorted with.
        n (int): The number of top items to display. Default is 10.
        ascending (bool): Whether to sort the items in ascending order. Default is False.
    """
    try:
        # Grouping data by the specified column and calculating sum of metrics
        grouped_data = df.groupby(group_column)[metric_column].sum()

        # Sorting the data based on the specified metric
        sorted_data = grouped_data.sort_values(by=sort_by, ascending=ascending)

        # Selecting the top n items
        top_n_items = sorted_data.head(n)

        # Plotting
        plt.figure(figsize=(10, 6))
        top_n_items.plot(kind='bar', stacked=True)
        plt.title(f'{"Bottom" if ascending else "Top" } {n} Items by {sort_by.capitalize()}')
        plt.xlabel(group_column.capitalize())
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    
    except KeyError as e:
        print(f"Error: {e}. Please ensure the provided columns exist in the DataFrame.")
