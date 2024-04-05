# from pymatgen import MPRester
from mp_api.client import MPRester
from emmet.core.summary import HasProps
import requests
import pandas as pd
import os

from datasets import Dataset, DatasetDict
import sys, cloudpickle

from dotenv import load_dotenv
load_dotenv("../.env")

with MPRester(os.environ['MAPI_API_KEY']) as mpr:
  docs = mpr.materials.summary.search(
      # Needed to remove some fields with dates because some entries had invalid dates. This was breaking the workflow
      fields=['formula_pretty', 'symmetry', 'density', 'nsites', 'elements', 'nelements', 'composition', 'composition_reduced', 'chemsys', 'volume', 'density_atomic', 'property_name', 'material_id', 'structure', 'uncorrected_energy_per_atom', 'energy_per_atom', 'formation_energy_per_atom', 'energy_above_hull', 'is_stable', 'equilibrium_reaction_energy_per_atom', 'decomposes_to', 'grain_boundaries', 'band_gap', 'efermi', 'is_gap_direct', 'is_metal', 'bandstructure', 'dos', 'dos_energy_up', 'dos_energy_down', 'is_magnetic', 'total_magnetization', 'total_magnetization_normalized_vol', 'total_magnetization_normalized_formula_units', 'num_magnetic_sites', 'num_unique_magnetic_sites', 'types_of_magnetic_species', 'bulk_modulus', 'shear_modulus', 'universal_anisotropy', 'homogeneous_poisson', 'e_total', 'e_ionic', 'e_electronic', 'n', 'e_ij_max', 'weighted_surface_energy_EV_PER_ANG2', 'weighted_surface_energy', 'weighted_work_function', 'surface_anisotropy', 'shape_factor', 'has_reconstructed', 'possible_species', 'theoretical']
      )
  
cloudpickle.dump(docs, open('./docs.pkl', 'wb'))

df = pd.DataFrame(d.model_dump() for d in docs)
df['crystal_system'] = [d.model_dump()['symmetry']['crystal_system'].value for d in docs]
df['symbol'] = [d.model_dump()['symmetry']['symbol'] for d in docs]
df['number'] = [d.model_dump()['symmetry']['number'] for d in docs]
df['point_group'] = [d.model_dump()['symmetry']['point_group'] for d in docs]
df['structure'] = [d.model_dump()['structure'].__str__() for d in docs]
# TODO: Process elements list to save it as well

to_remove = [
    #removed because they use fancier types
    'builder_meta',
    'composition',
    'composition_reduced',
    'symmetry',
    'formula_anonymous',
    'fields_not_requested',
    'deprecated',
    'decomposes_to',
    'bandstructure',
    'dos',
    'types_of_magnetic_species',
    'possible_species',
    'elements', #Elements is a list of Elements. Need to convert it to a comma-separated string. But using regex on `formula_pretty` work as well
    # removed because they're useless
    'warning',
    'has_props',
    'task_ids',
    'database_IDs',
    'property_name'
]

dataset = Dataset.from_pandas(df.drop(to_remove, axis=1))
train_test_split = dataset.train_test_split(
    test_size=0.2, shuffle=True, seed=8
)

# Created a frozen dataset. Now the train/test split will be constant as we work on different models
dataset_dict = DatasetDict({
    "train": train_test_split["train"],
    "test": train_test_split["test"],
})
dataset_dict.push_to_hub(repo_id='ur-whitelab/mapi', private=True, token=os.environ['HF_TOKEN'])