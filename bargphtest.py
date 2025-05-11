# import os
# import streamlit as st
# from dotenv import load_dotenv
# import logging
# from langchain.schema import HumanMessage, AIMessage
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain_community.graphs import Neo4jGraph
# from langchain.chains import GraphCypherQAChain
# import matplotlib.pyplot as plt
# import io
# import base64
# import re
# import ast

# # ... (keep your existing imports and setup code)

# load_dotenv()
# # Set up logging
# logging.basicConfig(
#     filename='migration_log.log',
#     level=logging.DEBUG,  # Change to DEBUG for more verbosity
#     format='%(asctime)s - %(message)s'
# )
# st.title("Big Data Query")
# if "chat_history_know" not in st.session_state:
#       st.session_state.chat_history_know = [
#         AIMessage(content="Hello! I'm a knowledge assistant. Ask me anything about your Data."),
#       ]
# NEO4J_URL = "bolt://localhost:7687"
# NEO4J_USERNAME = "neo4j"
# NEO4J_PASSWORD = "12345678"
# NEO4J_DATABASE = "test"
# prompt_template = PromptTemplate(
#     input_variables=["query", "context"],
#     template="""
#     You are an AI assistant specialized in analyzing data from a Neo4j graph database. 
#     Given the following query and context, provide a clear and concise answer.

#     This is the schema of the which I have in the database:
#     Node properties:
#     Animal {breed: STRING, sex: STRING, id: STRING}
#     Organization {name: STRING}
#     EnclosureType {type: STRING}
#     Treatment {week4Weight: FLOAT, week2Weight: FLOAT, week3Weight: FLOAT, sicknessDetails: STRING, recoveredInCurrentMonth: STRING, treatmentMedicines: STRING, week1Weight: FLOAT, id: STRING, lastMonthSick: STRING}
#     Location {name: STRING}
#     Scan {daily_time_spent: INTEGER, record_count: INTEGER, scan_date: STRING}
#     Relationship properties:
#     SCANNED_AT {on: STRING}
#     The relationships:
#     (:Animal)-[:HAS_TREATMENT]->(:Treatment)
#     (:Animal)-[:BELONGS_TO]->(:Organization)
#     (:Animal)-[:HOUSED_IN]->(:EnclosureType)
#     (:Animal)-[:SCANNED_AT]->(:Scan)
#     (:Scan)-[:IN_ORGANIZATION]->(:Organization)
#     (:Scan)-[:AT_LOCATION]->(:Location)

#     These are the property key which are there in the database:
#     breed, daily_time_spent, id, lastMonthSick, name, on, record_count, recoveredInCurrentMonth, scan_date, sex, sicknessDetails, treatmentMedicines, type, week1Weight, week2Weight, week3Weight, week4Weight.

#     Please understand that if there is a word 'Cattle or Calltle's' in the given query then it should be taken as 'Animal or Animals' to get the answer from the LLM.

#     Query: {query}

#     Context from Neo4j:
#     {context}

#     Please provide a detailed answer based on the given information. If the data seems unusual or inconsistent, mention it in your response. If you cannot answer the question based on the given context, explain why.

#     Answer:
#     """
# )

# def extract_data_for_visualization(result):
#     # Try to find a Python dictionary or list in the result
#     match = re.search(r'\{.*?\}|\[.*?\]', result, re.DOTALL)
#     if match:
#         try:
#             data = ast.literal_eval(match.group())
#             if isinstance(data, dict):
#                 return list(data.keys()), list(data.values())
#             elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
#                 keys = list(data[0].keys())
#                 values = [list(item.values()) for item in data]
#                 return keys, list(zip(*values))
#         except:
#             pass
    
#     # If no structured data found, split the result into lines and try to extract key-value pairs
#     lines = result.split('\n')
#     data = {}
#     for line in lines:
#         parts = line.split(':')
#         if len(parts) == 2:
#             key = parts[0].strip()
#             value = parts[1].strip()
#             try:
#                 value = float(value)
#                 data[key] = value
#             except ValueError:
#                 pass
    
#     if data:
#         return list(data.keys()), list(data.values())
    
#     return None, None

# def create_chart(labels, values, chart_type='bar'):
#     plt.figure(figsize=(10, 6))
#     if chart_type == 'bar':
#         plt.bar(labels, values)
#     elif chart_type == 'pie':
#         plt.pie(values, labels=labels, autopct='%1.1f%%')
#     plt.title(f"{chart_type.capitalize()} Chart of Data")
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
    
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     img_str = base64.b64encode(buf.read()).decode()
#     plt.close()
    
#     return img_str

# # ... (keep your existing setup code)
# llm = ChatOpenAI(model='gpt-4o', api_key=os.getenv('OPEANI_API_KEY'))
# graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME,password=NEO4J_PASSWORD, database=NEO4J_DATABASE)
# chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm,validate_cypher=True,verbose=True,question_prompt=prompt_template)
# for message in st.session_state.chat_history_know:
#     if isinstance(message, AIMessage):
#         with st.chat_message("AI"):
#             st.markdown(message.content)
#     elif isinstance(message, HumanMessage):
#         with st.chat_message("Human"):
#             st.markdown(message.content)
# chat_input=st.chat_input("enter your query here:",key="knowlegde")
# if chat_input:
#     st.session_state.chat_history_know.append(HumanMessage(content=chat_input))
#     with st.chat_message("Human"):
#         st.markdown(chat_input)
#     with st.spinner("Processing"):
#         response = chain.invoke({"query": chat_input})
#         result = response['result']
        
#         logging.info(result)
#         with st.chat_message("AI"):
#             st.markdown(result)
#         st.session_state.chat_history_know.append(AIMessage(content=result))
        
#         # Attempt to visualize the data
#         labels, values = extract_data_for_visualization(result)
#         if labels and values:
#             chart_type = 'bar' if len(labels) > 5 else 'pie'
#             chart_base64 = create_chart(labels, values, chart_type)
#             st.image(f"data:image/png;base64,{chart_base64}")
#         else:
#             st.write("No suitable data for visualization found in the response.")

# import os
# import streamlit as st
# from dotenv import load_dotenv
# import logging
# from langchain.schema import HumanMessage, AIMessage
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain_community.graphs import Neo4jGraph
# from langchain.chains import GraphCypherQAChain
# import matplotlib.pyplot as plt
# import io
# import base64
# import re
# import json

# # ... (keep your existing imports and setup code)

# load_dotenv()
# # Set up logging
# logging.basicConfig(
#     filename='migration_log.log',
#     level=logging.DEBUG,  # Change to DEBUG for more verbosity
#     format='%(asctime)s - %(message)s'
# )
# st.title("Big Data Query")
# if "chat_history_know" not in st.session_state:
#       st.session_state.chat_history_know = [
#         AIMessage(content="Hello! I'm a knowledge assistant. Ask me anything about your Data."),
#       ]
# NEO4J_URL = "bolt://localhost:7687"
# NEO4J_USERNAME = "neo4j"
# NEO4J_PASSWORD = "12345678"
# NEO4J_DATABASE = "test"
# prompt_template = PromptTemplate(
#     input_variables=["query", "context"],
#     template="""
#     You are an AI assistant specialized in analyzing data from a Neo4j graph database. 
#     Given the following query and context, provide a clear and concise answer.

#     This is the schema of the which I have in the database:
#     Node properties:
#     Animal {breed: STRING, sex: STRING, id: STRING}
#     Organization {name: STRING}
#     EnclosureType {type: STRING}
#     Treatment {week4Weight: FLOAT, week2Weight: FLOAT, week3Weight: FLOAT, sicknessDetails: STRING, recoveredInCurrentMonth: STRING, treatmentMedicines: STRING, week1Weight: FLOAT, id: STRING, lastMonthSick: STRING}
#     Location {name: STRING}
#     Scan {daily_time_spent: INTEGER, record_count: INTEGER, scan_date: STRING}
#     Relationship properties:
#     SCANNED_AT {on: STRING}
#     The relationships:
#     (:Animal)-[:HAS_TREATMENT]->(:Treatment)
#     (:Animal)-[:BELONGS_TO]->(:Organization)
#     (:Animal)-[:HOUSED_IN]->(:EnclosureType)
#     (:Animal)-[:SCANNED_AT]->(:Scan)
#     (:Scan)-[:IN_ORGANIZATION]->(:Organization)
#     (:Scan)-[:AT_LOCATION]->(:Location)

#     These are the property key which are there in the database:
#     breed, daily_time_spent, id, lastMonthSick, name, on, record_count, recoveredInCurrentMonth, scan_date, sex, sicknessDetails, treatmentMedicines, type, week1Weight, week2Weight, week3Weight, week4Weight.

#     Please understand that if there is a word 'Cattle or Calltle's' in the given query then it should be taken as 'Animal or Animals' to get the answer from the LLM.

#     Query: {query}

#     Context from Neo4j:
#     {context}

#     Please provide a detailed answer based on the given information. If the data seems unusual or inconsistent, mention it in your response. If you cannot answer the question based on the given context, explain why.

#     Answer:
#     """
# )


# def extract_data_for_visualization(cypher_result, llm_result):
#     # First, try to parse the Cypher query result
#     try:
#         data = json.loads(cypher_result)
#         if isinstance(data, list) and len(data) > 0:
#             keys = list(data[0].keys())
#             values = [list(item.values()) for item in data]
#             return keys, list(zip(*values))
#     except json.JSONDecodeError:
#         pass

#     # If Cypher result parsing fails, try to extract from LLM result
#     try:
#         # Look for patterns like "key: value" or "key - value"
#         pattern = r'([^:\n-]+)[:|-]\s*([^,\n]+)'
#         matches = re.findall(pattern, llm_result)
#         if matches:
#             keys, values = zip(*matches)
#             # Try to convert values to float if possible
#             values = [float(v.strip()) if v.strip().replace('.', '').isdigit() else v.strip() for v in values]
#             return list(keys), list(values)
#     except:
#         pass

#     return None, None

# def create_chart(labels, values, chart_type='bar'):
#     plt.figure(figsize=(10, 6))
#     if chart_type == 'bar':
#         plt.bar(labels, values)
#         plt.ylabel('Value')
#     elif chart_type == 'pie':
#         plt.pie(values, labels=labels, autopct='%1.1f%%')
#     plt.title(f"{chart_type.capitalize()} Chart of Data")
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
    
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     img_str = base64.b64encode(buf.read()).decode()
#     plt.close()
    
#     return img_str

# # Modify the chain to return both Cypher query results and LLM interpretation
# class CustomGraphCypherQAChain(GraphCypherQAChain):
#     def _call(self, inputs: dict):
#         _run_manager = None
#         callbacks = None
#         _run_manager = None
#         question = inputs["query"]
#         generated_cypher = self.cypher_generation_chain.predict(
#             question=question,
#             schema=self.graph_schema,
#             callback_manager=_run_manager,
#         )
#         context = self.graph.query(generated_cypher)
        
#         # Store the Cypher query results
#         cypher_results = context
        
#         context = self.context_string + str(context)
#         result = self.qa_chain(
#             {"question": question, "context": context}, callbacks=callbacks
#         )
#         return {"result": result["text"], "cypher_results": cypher_results}

# # ... (keep your existing setup code, but replace GraphCypherQAChain with CustomGraphCypherQAChain)
# llm = ChatOpenAI(model='gpt-4o', api_key=os.getenv('OPEANI_API_KEY'))
# graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME,password=NEO4J_PASSWORD, database=NEO4J_DATABASE)
# chain = CustomGraphCypherQAChain.from_llm(
#     graph=graph, 
#     llm=llm,
#     validate_cypher=True,
#     verbose=True,
#     question_prompt=prompt_template
# )
# chat_input=st.chat_input("enter your query here:",key="knowlegde")
# if chat_input:
#     st.session_state.chat_history_know.append(HumanMessage(content=chat_input))
#     with st.chat_message("Human"):
#         st.markdown(chat_input)
#     with st.spinner("Processing"):
#         response = chain.invoke({"query": chat_input})
#         result = response['result']
#         cypher_results = response['cypher_results']
        
#         logging.info(f"Cypher Results: {cypher_results}")
#         logging.info(f"LLM Result: {result}")
        
#         with st.chat_message("AI"):
#             st.markdown(result)
#         st.session_state.chat_history_know.append(AIMessage(content=result))
        
#         # Attempt to visualize the data
#         labels, values = extract_data_for_visualization(cypher_results, result)
#         if labels and values:
#             chart_type = 'bar' if len(labels) > 5 else 'pie'
#             chart_base64 = create_chart(labels, values, chart_type)
#             st.image(f"data:image/png;base64,{chart_base64}")
#         else:
#             st.write("No suitable data for visualization found in the response.")

import os
import streamlit as st
from dotenv import load_dotenv
import logging
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
import matplotlib.pyplot as plt
import io
import base64
import re
import ast

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    filename='migration_log.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s'
)

# Streamlit title
st.title("Big Data Query")

# Initialize chat history if not already present
if "chat_history_know" not in st.session_state:
    st.session_state.chat_history_know = [
        AIMessage(content="Hello! I'm a knowledge assistant. Ask me anything about your Data."),
    ]

# Neo4j configuration
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "12345678"
NEO4J_DATABASE = "test"

# Prompt template for LLM
prompt_template = PromptTemplate(
    input_variables=["query", "context"],
    template="""
    You are an AI assistant specialized in analyzing data from a Neo4j graph database. 
    Given the following query and context, provide a clear and concise answer.

    This is the schema of the which I have in the database:
    Node properties:
    Animal {breed: STRING, sex: STRING, id: STRING}
    Organization {name: STRING}
    EnclosureType {type: STRING}
    Treatment {week4Weight: FLOAT, week2Weight: FLOAT, week3Weight: FLOAT, sicknessDetails: STRING, recoveredInCurrentMonth: STRING, treatmentMedicines: STRING, week1Weight: FLOAT, id: STRING, lastMonthSick: STRING}
    Location {name: STRING}
    Scan {daily_time_spent: INTEGER, record_count: INTEGER, scan_date: STRING}
    Relationship properties:
    SCANNED_AT {on: STRING}
    The relationships:
    (:Animal)-[:HAS_TREATMENT]->(:Treatment)
    (:Animal)-[:BELONGS_TO]->(:Organization)
    (:Animal)-[:HOUSED_IN]->(:EnclosureType)
    (:Animal)-[:SCANNED_AT]->(:Scan)
    (:Scan)-[:IN_ORGANIZATION]->(:Organization)
    (:Scan)-[:AT_LOCATION]->(:Location)

    These are the property key which are there in the database:
    breed, daily_time_spent, id, lastMonthSick, name, on, record_count, recoveredInCurrentMonth, scan_date, sex, sicknessDetails, treatmentMedicines, type, week1Weight, week2Weight, week3Weight, week4Weight.

    Please understand that if there is a word 'Cattle or Calltle's' in the given query then it should be taken as 'Animal or Animals' to get the answer from the LLM.

    Query: {query}

    Context from Neo4j:
    {context}

    Please provide a detailed answer based on the given information. If the data seems unusual or inconsistent, mention it in your response. If you cannot answer the question based on the given context, explain why.

    Answer:
    """
)

# Function to extract data for visualization
def extract_data_for_visualization(result):
    try:
        # Try to find a Python dictionary or list in the result
        match = re.search(r'({.*?})|(\[.*?\])', result, re.DOTALL)
        if match:
            data = ast.literal_eval(match.group())
            if isinstance(data, dict):
                return list(data.keys()), list(data.values())
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                keys = list(data[0].keys())
                values = [list(item.values()) for item in data]
                return keys, list(zip(*values))
    except:
        pass
    
    # If no structured data found, split the result into lines and try to extract key-value pairs
    lines = result.split('\n')
    data = {}
    for line in lines:
        parts = line.split(':')
        if len(parts) == 2:
            key = parts[0].strip()
            value = parts[1].strip()
            try:
                value = float(value)
                data[key] = value
            except ValueError:
                pass
    
    if data:
        return list(data.keys()), list(data.values())
    
    return None, None

# Function to create chart
def create_chart(labels, values, chart_type='bar'):
    plt.figure(figsize=(10, 6))
    if chart_type == 'bar':
        plt.bar(labels, values)
    elif chart_type == 'pie':
        plt.pie(values, labels=labels, autopct='%1.1f%%')
    plt.title(f"{chart_type.capitalize()} Chart of Data")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    
    return img_str

# Set up LLM and Neo4j graph
llm = ChatOpenAI(model='gpt-4', api_key=os.getenv('OPENAI_API_KEY'))
graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, validate_cypher=True, verbose=True, question_prompt=prompt_template)

# Display chat history
for message in st.session_state.chat_history_know:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# Chat input
chat_input = st.chat_input("Enter your query here:", key="knowlegde")
if chat_input:
    st.session_state.chat_history_know.append(HumanMessage(content=chat_input))
    with st.chat_message("Human"):
        st.markdown(chat_input)
    
    # Processing the response
    with st.spinner("Processing"):
        response = chain.invoke({"query": chat_input})
        result = response['result']
        logging.info(result)
        
        with st.chat_message("AI"):
            st.markdown(result)
        st.session_state.chat_history_know.append(AIMessage(content=result))
        
        # Attempt to visualize the data
        labels, values = extract_data_for_visualization(result)
        if labels and values:
            chart_type = 'bar' if len(labels) > 5 else 'pie'
            chart_base64 = create_chart(labels, values, chart_type)
            st.image(f"data:image/png;base64,{chart_base64}")
        else:
            st.write("No suitable data for visualization found in the response.")
