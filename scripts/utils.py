import matplotlib.pyplot as plt
from logger import logger
from dotenv import load_dotenv,find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

def plot_bar_graph(df, column):
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
    
    except KeyError as e:
        logger.error(f"Error: Column '{column}' not found in the DataFrame. error: {e}")



def plot_histogram(data, column):
    """
    Plots a histogram showing the distribution of values in the specified column of the DataFrame.
    
    Parameters:
        df (DataFrame): The pandas DataFrame containing the data.
        column (str): The name of the column to plot.
    """
    try:
        # Plotting
        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=20, color="skyblue", edgecolor="black")
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
    
    except KeyError as e:
        logger.error(f"Error: Column '{column}' not found in the DataFrame. error: {e}")


def plot_pie_chart(data, column):
    """
    Plots a pie chart showing the distribution of values in the specified column of the DataFrame.
    
    Parameters:
        data (Series): The pandas Series containing the data.
        column (str): The name of the column to plot.
    """
    try:
        # Count the occurrences of each unique value in the column
        value_counts = data.value_counts(dropna=False)

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