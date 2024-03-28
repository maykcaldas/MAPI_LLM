from langchain.agents import Tool, tool
import requests
from langchain_community.llms import OpenAI
from langchain.chains import LLMMathChain
from langchain_community.utilities import SerpAPIWrapper
import os
from rdkit import Chem

@tool
def query2smiles(text):
  '''This function queries the one given molecule name and returns a SMILES string from the record'''
  try:#query the PubChem database
    r = requests.get('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/' + text + '/property/IsomericSMILES/JSON')
    #convert the response to a json object
    data = r.json() 
    #return the SMILES string
    smi = data['PropertyTable']['Properties'][0]['IsomericSMILES']
    # remove salts
    return smi
  except:
    f"Could not find the IUPAC name for {text}"

@tool
def smiles2IUPAC(text):
  '''This function queries the one given smiles name and returns a IUPAC name from the record'''
  #query the PubChem database
  try:
    r = requests.get('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/' + text + '/property/IUPACName/JSON')
    data = r.json()
    smi = data["PropertyTable"]["Properties"][0]["IUPACName"]
    return smi
  except:
    return f"Could not find the IUPAC name for {text}"

@tool
def formula2IUPAC(text):
  '''This function queries the one given chemical formula and returns a material name from the record.'''
  try:
    r = requests.get('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/formula/' + text + '/property/IUPACName/JSON')
    data = r.json()
    print(data)
    smi = data["PropertyTable"]["Properties"][0]["IUPACName"]
    return smi
  except:
    return f"Could not find the IUPAC name for {text}"

@tool
def name2formula(text):
  '''This function queries the one given material name and returns a chemical formula from the record.'''
  try:
    r = requests.get('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/' + text + '/property/MolecularFormula/JSON')
    data = r.json()
    print(data)
    smi = data["PropertyTable"]["Properties"][0]["MolecularFormula"]
    return smi
  except:
    return f"Could not find the molecular formula for {text}"

@tool
def canonicalizeSMILES(smiles):
  '''Given a smiles representation, this function returns a canonicalized version of the same smiles.
  It's better to search for molecules in its canonicalized form'''
  return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

@tool
def web_search(keywords, search_engine="google"):
  '''Useful to do a simple google search. 
      Use this tool to find general information from websites.
      Use keywords for your search. 
  '''
  return SerpAPIWrapper(
    serpapi_api_key=os.getenv("SERP_API_KEY"),
    search_engine=search_engine
  ).run(keywords)

@tool
def LLM_predict(prompt):
  ''' This function receives a prompt generate with context by the create_context_prompt tool and request a completion to a language model. Then returns the completion'''
  llm = OpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        n=1,
        best_of=5,
        top_p=1.0,
        stop=["\n\n", "###", "#", "##"],
        # model_kwargs=kwargs,
    )
  return llm.generate([prompt]).generations[0][0].text

common_tools = [
    query2smiles,
    smiles2IUPAC,
    # formula2IUPAC,
    # name2formula,
    canonicalizeSMILES,
    # web_search,
    LLM_predict
]