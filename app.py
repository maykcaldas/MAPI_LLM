import gradio as gr
import numpy as np
import agent
import os

css_style = """
.gradio-container {
    font-family: "IBM Plex Mono";
}
"""

def agent_run(q, openai_api_key, mapi_api_key, serp_api_key):
    os.environ["OPENAI_API_KEY"]=openai_api_key
    os.environ["MAPI_API_KEY"]=mapi_api_key
    os.environ["SERPAPI_API_KEY"]=serp_api_key
    agent_chain = agent.Agent(openai_api_key, mapi_api_key)
    try: 
        out = agent_chain.run(q)
    except Exception as err:
        out = f"Something went wrong, please try again.\nError: {err}"
    return out

with gr.Blocks(css=css_style) as demo:
    gr.Markdown(f'''
    # A LLM application developed during the LLM March *MADNESS* Hackathon
    - Developed by: Mayk Caldas ([@maykcaldas](https://github.com/maykcaldas)) and Sam Cox ([@SamCox822](https://github.com/SamCox822))

    ## What is this?
    - This is a demo of a LLM agent that can answer questions about materials science using the [LangChain🦜️🔗](https://github.com/hwchase17/langchain/) and the [Materials Project API](https://materialsproject.org/).
    - Its behave is based on Large Language Models (LLM) and aim to be a tool to help scientists with quick predictions of a nunerous of properties of materials.
    It is a work in progress, so please be patient with it.


    ### Some keys are needed in order to use it:
    1. An openAI API key ( [Check it here](https://platform.openai.com/account/api-keys) )
    2. A material project's API key ( [Check it here](https://materialsproject.org/api#api-key) )
    3. A SERP API key ( [Check it here](https://serpapi.com/account-api) )
        - Only used if the chain decides to run a web search to answer the question.
    ''')
    with gr.Accordion("List of properties we developed tools for", open=False):
        gr.Markdown(f"""
        - Classification tasks: "Is the material AnByCz stable?"
            - Stable, 
            - Magnetic, 
            - Gap direct, and 
            - Metal.
        - Regression tasks: "What is the band gap of the material AnByCz?"
            - Band gap, 
            - Volume, 
            - Density, 
            - Atomic density, 
            - Formation energy per atom, 
            - Energy per atom, 
            - Electronic energy, 
            - Ionic energy, and 
            - Total energy.
        - Reaction procedure for synthesis proposal: "Give me a reaction procedure to synthesize the material AnByCz"(under development)
        """)
    openai_api_key = gr.Textbox(
        label="OpenAI API Key", placeholder="sk-...", type="password")
    mapi_api_key = gr.Textbox(
        label="Material Project API Key", placeholder="...", type="password")
    serp_api_key = gr.Textbox(
        label="Material Project API Key", placeholder="...", type="password")
    with gr.Tab("MAPI Query"):
        text_input = gr.Textbox(label="", placeholder="Enter question here...")
        text_output = gr.Textbox(placeholder="Your answer will appear here...")
        text_button = gr.Button("Ask!")

    text_button.click(agent_run, inputs=[text_input, openai_api_key, mapi_api_key, serp_api_key], outputs=text_output)

demo.launch()
