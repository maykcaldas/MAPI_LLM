import os
from mp_api.client import MPRester
import sys


class SuppressOutput:
    #this is just because I'm annoyed with the print statements and progress bars from mp-api
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr.close()
        sys.stderr = self._original_stderr

class ExactMatchingTool():
  #move to other class when ready
    def __init__(self):
        self.mpr = MPRester(os.getenv("MAPI_API_KEY"))

    def _get_specific_query(self, docs, space_group_num=None, density=None, lattice=None):
        if space_group_num:
            docs = [doc for doc in docs if doc.symmetry.number == space_group_num]
        if density:
            #instead of exact matching density, get density match to the first decimal
            docs = [doc for doc in docs if round(doc.density, 1) == round(density, 1)]
        if lattice:
            docs = [doc for doc in docs if doc.symmetry.crystal_system.value == lattice]
        return docs
  
    def query_material_property(self, formula, desired_prop, space_group_num=None, density=None, lattice=None, one_only=True):
        with SuppressOutput(): 
            docs = self.mpr.materials.summary.search(formula=formula, fields=["formula_pretty", desired_prop, "symmetry", "density", "lattice"])
        docs = self._get_specific_query(docs, space_group_num, density, lattice)
        if not docs:
            return []
        if one_only:
            docs = [docs[0]]
        return docs
  
    def _get_prop(self, docs, prop):
        props = []
        for doc in docs:
            props.append(getattr(doc, prop))
        return props
  
    def average_neighbors(self, neighbors:dict):
        total_weight = 0
        weighted_sum = 0
        
        for score, values in neighbors.items():
            weighted_sum += score * sum(values)
            total_weight += score * len(values)
        
        if total_weight == 0:
            return 0  
        else:
            return weighted_sum / total_weight
      
    def average_vanilla(self, all_values:list, property:str):
        all_values = self._get_prop(all_values, property)
        return sum(all_values) / len(all_values)
   
    def k_vals(self, input_param:dict, k:int):
        num_values = 0
        for key in input_param:
            num_values += len(input_param[key])
        return num_values >= k

    def get_best_matches(self, formula, desired_prop, space_group_num=None, density=None, lattice=None, k=10):
        """
        Get the best matches following the scoring rubric.
        """
        N_params = {}

        score_criteria = [
            # Score 3 criteria
            [
                {"formula": formula, "lattice": lattice, "space_group_num": space_group_num, "desired_prop": desired_prop},
                {"formula": formula, "lattice": lattice, "density": density, "desired_prop": desired_prop},
                {"formula": formula, "space_group_num": space_group_num, "density": density, "desired_prop": desired_prop}
            ],
            # Score 2 criteria
            [
                {"formula": formula, "lattice": lattice, "desired_prop": desired_prop},
                {"formula": formula, "space_group_num": space_group_num, "desired_prop": desired_prop},
                {"formula": formula, "density": density, "desired_prop": desired_prop}
            ],
            # Score 1 criteria
            [{"formula": formula, "desired_prop": desired_prop}]
        ]
        
        for score, criteria in enumerate(score_criteria, start=1):
            props = []
            for criterion in criteria:
                criterion["one_only"] = False  # Apply common property
                docs = self.query_material_property(**criterion)
                if docs:
                    props += self._get_prop(docs, desired_prop)
            N_params[score] = props
            
            if self.k_vals(N_params, k):
                return N_params

        return N_params

    def get_weighted_average_of_neighbors(self, input_params:dict):
        required_params = ["formula", "desired_prop"]
        if not all([param in input_params for param in required_params]):
            raise ValueError(f"Missing required parameters: {required_params}")
        neighbors = self.get_best_matches(**input_params)
        return self.average_neighbors(neighbors)
    
    def get_prop(self, formula, desired_prop, space_group_num=None, density=None, lattice=None, one_only=True):
        docs = self.query_material_property(formula, desired_prop, space_group_num, density, lattice, one_only=False)
        all_props = self._get_prop(docs, desired_prop)
        if not docs:
            print ("Warning: no matches found. Returning the average of closet matches.")
            nearest_neighbor_avg = self.get_weighted_average_of_neighbors({"formula": formula, "desired_prop": desired_prop, "space_group_num": space_group_num, "density": density, "lattice": lattice})
            return nearest_neighbor_avg
        if one_only:
            average_prop = self.average_vanilla(docs, desired_prop)
            print ("Warning: multiple matches found. Returning average of all matches. To get all matches, set one_only=False.")
            return average_prop
        else:
            return all_props
        