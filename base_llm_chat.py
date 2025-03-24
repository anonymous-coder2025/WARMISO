# import openai
from openai import OpenAI

def llm_generate_response(llm_type, prompt):
     if llm_type in ["llama3-70b"]:
          # fastchat / vllm method 
          model_name = llm_type 
          openai_api_base = "http://XX.XX.XX.xx:8000/v1"  
          openai_api_key = "none" 
          client = OpenAI(
          api_key=openai_api_key,
          base_url=openai_api_base,
          )
     elif llm_type in ["deepseek-chat"]:
          model_name = llm_type
          openai_api_base = "https://api.deepseek.com"  
          openai_api_key = "*****"

          client = OpenAI(
          api_key=openai_api_key,
          base_url=openai_api_base,
          )

     elif llm_type in ["gpt-4o-2024-08-06", "gpt-4o-mini"]:
          # call gpt-4o-mini
          model_name = llm_type 
          openai_api_base = "https://api.openai.com/v1"
          openai_api_key = "*****"
          
          client = OpenAI(
          api_key=openai_api_key,
          base_url=openai_api_base,
          )
     else:
          print("No Valide LLM type!")
     
     
     respose = client.chat.completions.create(
          model = model_name,
          messages=[
               {"role": "user", "content": prompt}
          ],
          max_tokens = 512,
          temperature=0.0, 
          top_p=0.1,
     )
     print(respose)
     return respose.choices[0].message.content


if __name__=='__main__':
     prompt = '''Write a list of APIs to achieve the query:

Query: Access data on a SQLite database and read it into a local KNIME data table.
APIs:
  1. SQLite_Connector: Create a connection to the SQLite database
  2. DB_Table_Selector: Select the table from the connected database
  3. DB_Reader: Read the selected table into a local KNIME data table

Query: Perform a clustering of the iris dataset using the k-Means.
APIs:
     '''
     test_respose = llm_generate_response("gpt-4o-mini", prompt)
     print(test_respose)
