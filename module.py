import os
import pytz
import psycopg2
import warnings
import pandas as pd
from google import genai
from datetime import datetime
import torch.nn.functional as F
from torch import Tensor
from template_prompt import template_prompt_chatbot
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv
load_dotenv("./.env")


from transformers import AutoModel, AutoTokenizer

# Load embedding model and tokenizer
embedding_model = AutoModel.from_pretrained("./vector_models/embedding_model")
embedding_tokenizer = AutoTokenizer.from_pretrained("./vector_models/embedding_tokenizer")


QUERY_DIR_PATH="./queries"

def postgresql_connect():
    """
    Connect to postgresql database
    """
    database_client = psycopg2.connect(
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USERNAME"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

    return database_client


def insert_data_to_db(data: dict):
    """
    Insert processed data to database
    """
    connection = postgresql_connect()
    cursor = connection.cursor()

    with open(f"{QUERY_DIR_PATH}/insert_data.sql", "r") as openfile:
        query_file = openfile.read()

    values = (
        data["input_prompt"],
        data["result"],
        data["input_token"],
        data["output_token"],
        data["retrieved_knowledge_base"],
        round(float(data["embedding_similarity"]), 2),
        datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")
    )
    cursor.execute(query_file, values)
    connection.commit()

    cursor.close()
    connection.close()


def average_pooling(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    """
    Performing average pooling based on the vector data produced by embedding model
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def retrieve_vector_data(input_prompt: str,
                         embedding_model,
                         embedding_tokenizer):
    warnings.filterwarnings("ignore")

    # Vectorization (embedding)
    reconstructed_input_prompt = f"query: {input_prompt}"
    input_token_data = embedding_tokenizer(reconstructed_input_prompt, max_length=512, padding=True, truncation=True, return_tensors="pt")
    embedding_output = embedding_model(**input_token_data)
    embedding_output = average_pooling(embedding_output.last_hidden_state, input_token_data["attention_mask"])
    normalized_embedding_output = F.normalize(embedding_output, p=2, dim=1)
    vector_data = normalized_embedding_output.detach().numpy().tolist()

    # Get vector data
    connection = postgresql_connect()
    with open(f"{QUERY_DIR_PATH}/get_vector_data.sql", "r") as openfile:
        query = openfile.read()
        query = query.replace('@VECTOR_INPUT', str(vector_data[0]))
    
    df = pd.read_sql_query(query, connection)
    connection.close()

    # Select best matched data
    context = ""
    similarity_1 = 0
    for idx in range(len(df)):
        knowledge = df["knowledge"].values[idx]
        similarity = df["similarity"].values[idx]
        if not context and similarity >= 0.84:
            context = f"CONTEXT 1:\n{knowledge}\n\n"
            similarity_1 = similarity
        else:
            if context and abs(similarity_1 - similarity) <= 0.01:
                context += f"CONTEXT {idx+1}:\n{knowledge}\n\n"

    
    return context, similarity_1


def retrieve_from_tabular_data(file_path,
                               input_message):
    """
    Within this function, I used pandas dataframe (not database) due to unavailability
    to use my database to store huge amount of fraud data, so I just used data which
    is loaded from csv file.
    """
    # Read the data
    df = pd.read_csv(file_path)

    os.environ["GOOGLE_API_KEY"] = str(os.getenv("GOOGLE_CLOUD_API_KEY"))

    # Define LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0
    )

    system_prompt="""
THE IMPORTANT RULE:
1. If the answer can't be found, return output NOT FOUND.
2. Return the output with the prefix sentence, for example, "the total transaction is 4000 during 2021". Don't just return 4000.
"""

    # Define agent
    agent_pandas = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True,
        prefix=system_prompt,
        handle_parsing_errors=True
    )

    # Get response
    response = agent_pandas.invoke(input_message)

    return response["output"]


def construct_response(input_prompt: str):
    """
    Construct LLM response for every input prompt from users
    """
    # Define the google client service
    client = genai.Client(api_key=os.getenv("GOOGLE_CLOUD_API_KEY"))

    # Retrieve relevant data from vector database
    vector_context, embedding_similarity = retrieve_vector_data(
        input_prompt,
        embedding_model,
        embedding_tokenizer
    )
    # Retrieve relevant data from tabular
    tabular_data_context = retrieve_from_tabular_data(
        "./fraudTest.csv",
        input_prompt
    )

    # Construct context (combine context from the vector and tabular data), the LLM then chooses the
    # appropriate context.
    final_context = ""
    if vector_context:
        final_context += f"CONTEXT FROM SOURCE 1:\n{vector_context}"
    if tabular_data_context != "NOT FOUND":
        final_context += f"\n\nCONTEXT FROM SOURCE 2:\n{tabular_data_context}"

    template_prompt = template_prompt_chatbot.replace('@CONTEXT', str(final_context))

    # Get response from llm
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            template_prompt,
            input_prompt
        ]
    )

    result = response.text

    response_data = {
        "input_prompt": input_prompt,
        "result": result,
        "input_token": response.usage_metadata.prompt_token_count,
        "output_token": response.usage_metadata.candidates_token_count,
        "retrieved_knowledge_base": final_context,
        "embedding_similarity": embedding_similarity
    }

    # Insert response data to database
    insert_data_to_db(response_data)

    return result

    