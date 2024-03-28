from mp_api.client import MPRester
from emmet.core.summary import HasProps
import openai
import langchain
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import Tool, tool
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts.example_selector import (MaxMarginalRelevanceExampleSelector, 
                                                SemanticSimilarityExampleSelector)
import requests
import warnings
from rdkit import Chem
import pandas as pd
import os

from dotenv import load_dotenv
load_dotenv()

class MAPITools:
  def __init__(self, prop, prop_name, model='gpt-3.5-turbo', k=10):
    self.model = model #maybe change to gpt-4 when ready
    self.k=k
    self.prop = prop
    self.prop_name = prop_name
  
  def get_material_atoms(self, formula):
    f'''Receives a material formula and returns the atoms symbols present in it separated by comma.'''
    import re
    pattern = re.compile(r"([A-Z][a-z]*)(\d*)")
    matches = pattern.findall(formula)
    atoms = []
    for m in matches:
      atom, count = m
      count = int(count) if count else 1
      atoms.append((atom, count))
    return ",".join([a[0] for a in atoms])

  def check_prop_by_formula(self, formula):
    raise NotImplementedError('Should be implemented in children classes')

  def search_similars_by_atoms(self, atoms):
    f'''This function receives a string with the atoms separated by comma as input and returns a list of similar materials.'''
    atoms = atoms.replace(" ", "")
    with MPRester(os.getenv("MAPI_API_KEY")) as mpr:
      docs = mpr.materials.summary.search(elements=atoms.split(','), fields=["formula_pretty", self.prop, "symmetry"])

    return docs

  def create_context_prompt(self, formula):
    raise NotImplementedError('Should be implemented in children classes')

  def LLM_predict(self, prompt):    
    f'''This function receives a prompt generate with context by the create_context_prompt tool and request a completion to a language model. Then returns the completion.'''
    llm = ChatOpenAI(
          model_name=self.model,
          temperature=0.1,
          n=5,
          # best_of=5,
          # stop=["\n\n", "###", "#", "##"],
      )
    return llm.invoke([prompt]).content

  def get_tools(self):
    return [
        Tool(
            name = f"Checks if material is {self.prop_name} by formula",
            func = self.check_prop_by_formula,
            description = (
                f"This functions searches in the material project's API for the formula and returns if it is {self.prop_name} or not."
              )
        ),
        Tool(
            name = f"Create {self.prop_name} context to LLM completion",
            func = self.create_context_prompt,
            description = (
              f"This function received a material formula as input and create a prompt to be inputed in the LLM_predict tool to predict if the material is {self.prop_name}." 
              if isinstance(self, MAPI_class_tools) else
              f"This function received a material formula as input and create a prompt to be inputed in the LLM_predict tool to predict the {self.prop_name} of a material." 
              )
        ),
        Tool(name = f"LLM prediction while computing {self.prop_name}",
            func = self.LLM_predict,
            description = (
                "This function receives a prompt generate with context by the create_context_prompt tool and request a completion to a language model. Then returns the completion"
              )
        )
    ]

class MAPI_class_tools(MAPITools):
  def __init__(self, prop, prop_name, p_label, n_label, model='gpt-3.5-turbo', k=10):
    super().__init__(prop=prop, prop_name=prop_name, model=model, k=k)
    self.p_label = p_label
    self.n_label = n_label

  def check_prop_by_formula(self, formula):
    f''' This functions searches in the material project's API for the formula and returns if it is {self.prop_name} or not.'''
    with MPRester(os.getenv("MAPI_API_KEY")) as mpr:
      docs = mpr.materials.summary.search(formula=formula, fields=["formula_pretty", self.prop])
    if len(docs) > 1:
      warnings.warn(f"More than one material found for {formula}. Will use the first one. Please, check the results.")
    if docs:
      if docs[0].formula_pretty == formula:
        return f"{formula} is {self.p_label if docs[0].model_dump()[self.prop] else self.n_label}"
    return f"Could not find any material while searching {formula}"

  def create_context_prompt(self, formula):
    f'''This function received a material formula as input and create a prompt to be inputed in the LLM_predict tool to predict if the formula is a {self.prop_name} material.'''
    elements = self.get_material_atoms(formula)
    docs = self.search_similars_by_atoms(elements)
    similars = [
        {'formula': ex.formula_pretty,
        'prop': self.p_label if ex.model_dump()[self.prop] else self.n_label,
        'lattice': ex.model_dump()['symmetry']['crystal_system'].value,
        'point_group': ex.model_dump()['symmetry']['point_group'],
        } for ex in docs
    ]
    
    examples = pd.DataFrame(similars).drop_duplicates().to_dict(orient="records")
    example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
                    examples,
                    OpenAIEmbeddings(),
                    FAISS,
                    k=self.k,
                  )
    
    prefix=(
      f'You are a bot who can predict if a material is {self.prop_name}.\n'
      f'Given this list of known materials and the information if they are {self.p_label} or {self.n_label}, \n'
      f'you need to answer the question if the last material is {self.prop_name}:'
      )
    prompt_template=PromptTemplate(
                  input_variables=["formula", "prop", "lattice", "point_group"],
                  template=f"Is {{lattice}} {{formula}} with space group {{point_group}} a {self.prop_name} material?@@@\n{{prop}}###",
              )
    suffix = f"Is {formula} a {self.prop_name} material?@@@\n"
    prompt = FewShotPromptTemplate(
              # examples=examples,
              example_prompt=prompt_template,
              example_selector=example_selector,
              prefix=prefix,
              suffix=suffix,
              input_variables=["formula"])
    
    return prompt.format(formula=formula)

class MAPI_reg_tools(MAPITools):
  # TODO: deal with units
  def __init__(self, prop, prop_name, model='gpt-3.5-turbo', k=10):
    super().__init__(prop=prop, prop_name=prop_name, model=model, k=k)

  def check_prop_by_formula(self, formula):
    f''' This functions searches in the material project's API for the formula and returns the {self.prop_name}.'''
    with MPRester(os.getenv("MAPI_API_KEY")) as mpr:
      docs = mpr.materials.summary.search(formula=formula, fields=["formula_pretty", self.prop])
    if len(docs) > 1:
      warnings.warn(f"More than one material found for {formula}. Will use the first one. Please, check the results.")
    if docs:
      if docs[0].formula_pretty == formula:
        return f"{formula} {self.prop_name} is {docs[0].model_dump()[self.prop]}"
      elif docs[0].model_dump()[self.prop] is None:
        return f"There is no record of {self.prop_name} for {formula}"
    return f"Could not find any material while searching {formula}"

  def create_context_prompt(self, formula):
    f'''This function received a material formula as input and create a prompt to be inputed in the LLM_predict tool to predict the {self.prop_name} of the material.'''
    elements = self.get_material_atoms(formula)
    docs = self.search_similars_by_atoms(elements)
    similars = [
        {'formula': ex.formula_pretty,
        'prop': f"{ex.model_dump()[self.prop]:2f}" if ex.model_dump()[self.prop] is not None else None,
        'lattice': ex.model_dump()['symmetry']['crystal_system'].value,
        'point_group': ex.model_dump()['symmetry']['point_group'],
        } for ex in docs
    ]

    examples = pd.DataFrame(similars).drop_duplicates().dropna().to_dict(orient="records")

    example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
                    examples,
                    OpenAIEmbeddings(),
                    FAISS,
                    k=self.k,
                  )
    
    prefix=(
      f'You are a bot who can predict the {self.prop_name} of a material .\n'
      f'Given this list of known materials and the measurement of their {self.prop_name}, \n'
      f'you need to predict what is the {self.prop_name} of the material:'
      f'The answer should be numeric and finish with ###'
      )
    prompt_template=PromptTemplate(
                  input_variables=["formula", "prop", "lattice", "point_group"],
                  template=f"What is the {self.prop_name} for {{lattice}} {{formula}} with space group {{point_group}}?@@@\n{{prop}}###",
              )
    suffix = f"What is the {self.prop_name} for {formula}?@@@\n"
    prompt = FewShotPromptTemplate(
              # examples=examples,
              example_prompt=prompt_template,
              example_selector=example_selector,
              prefix=prefix,
              suffix=suffix,
              input_variables=["formula"])
    
    return prompt.format(formula=formula)


# Now we create the tools
stability = MAPI_class_tools(
    "is_stable","stable","Stable","Unstable"
    )
magnetism = MAPI_class_tools(
    "is_magnetic","magnetic","Magnetic","Not magnetic"
    )
metal = MAPI_class_tools(
    "is_metal","metallic","Metal","Not metal"
    )
gap_direct = MAPI_class_tools(
    "is_gap_direct","gap direct","Gap direct","Gap indirect"
    )
band_gap = MAPI_reg_tools(
    "band_gap","band gap"
    )
energy_per_atom = MAPI_reg_tools(
    "energy_per_atom","energy per atom"
    )
formation_energy_per_atom = MAPI_reg_tools(
    "formation_energy_per_atom","formation energy per atom"
    )
volume = MAPI_reg_tools(
    "volume","volume"
    )
density = MAPI_reg_tools(
    "density","density"
    )
atomic_density = MAPI_reg_tools(
    "density_atomic","atomic density"
    )
electronic_energy = MAPI_reg_tools(
    "e_electronic","electronic energy"
    )
ionic_energy = MAPI_reg_tools(
    "e_ion","cationic energy"
    )
total_energy = MAPI_reg_tools(
    "e_total","total energy"
    )

mapi_tools = []
for prop in [stability, magnetism, metal, gap_direct, band_gap, 
             energy_per_atom, formation_energy_per_atom, volume, density, atomic_density, electronic_energy, ionic_energy, total_energy]:
# for prop in [band_gap]:
  mapi_tools += prop.get_tools()