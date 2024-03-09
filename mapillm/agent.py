from mapi_tools import mapi_tools
from utils import common_tools
# from reaction_prediction import SynthesisReactions
from langchain import hub, agents
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
import os

# reaction = SynthesisReactions()

class Agent:
    def __init__(self, openai_api_key, mapi_api_key): 
        self.llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-3.5-turbo",
            streaming=True,
        )
        self.tools = (
                mapi_tools +
                # reaction.get_tools() +
                # agents.load_tools(["llm-math", "python_repl"], llm=self.llm) +
                common_tools
              )
        
        self.prompt = hub.pull("hwchase17/react")
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def run(self, query: str):
        return self.agent_executor.invoke({
            'input': query
        })
    
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)

    a = Agent(openai_api_key=os.getenv("OPENAI_API_KEY"), mapi_api_key=os.getenv("MAPI_API_KEY"))
    a.run("What's the band gap of Fe3O4?")
