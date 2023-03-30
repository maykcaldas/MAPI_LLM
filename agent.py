from mapi_tools import MAPI_class_tools, MAPI_reg_tools
from utils import common_tools
from langchain import OpenAI
from gpt_index import GPTListIndex, GPTIndexMemory
from langchain import agents
from langchain.agents import initialize_agent

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
    "energy_per_atom","energy per atom gap"
    )
formation_energy_per_atom = MAPI_reg_tools(
    "formation_energy_per_atom","formation energy per atom gap"
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


memory = GPTIndexMemory(index=GPTListIndex([]), memory_key="chat_history", query_kwargs={"response_mode": "compact"})
llm=OpenAI(temperature=0.7)
tools = (
          stability.get_tools() +
          magnetism.get_tools() + 
          gap_direct.get_tools() + 
          metal.get_tools() + 
          band_gap.get_tools() +
          volume.get_tools() +
          density.get_tools() +
          atomic_density.get_tools() +
          formation_energy_per_atom.get_tools() +
          energy_per_atom.get_tools() +
          electronic_energy.get_tools() +
          ionic_energy.get_tools() +
          total_energy.get_tools() +
          agents.load_tools(["llm-math", "python_repl"], llm=llm) +
          common_tools
         )
agent_chain = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True, memory=memory)