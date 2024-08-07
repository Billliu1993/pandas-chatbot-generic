from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import MessagesPlaceholder
from langchain_experimental.agents.agent_toolkits.pandas.base import _get_functions_single_prompt
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_core.pydantic_v1 import BaseModel, Field


class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")

def create_agent(llm, df, schema, extra_tools=[]):

    prefix = """
        You are working with a pandas dataframe in Python to respond intelligently to user input.
        The name of the dataframe is `df`.
    """
    for column, description in schema.items():
        prefix += f"\n- {column}: {description}"
    prefix += "\nAnswer the user's questions about this dataframe using the provided column definitions for better accuracy."

    prompt = _get_functions_single_prompt(df=df, prefix=prefix)
    prompt.input_variables.append("chat_history")
    prompt.messages.insert(1, MessagesPlaceholder(variable_name="chat_history"))

    tools = [
        PythonAstREPLTool(
            locals={"df": df}, 
            name="python_repl",
            description="Runs code and returns the output of the final line",
            args_schema=PythonInputs,
            ),
    ]
    if extra_tools != []:
        tools += extra_tools

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5, return_intermediate_steps=True)
    
    return agent_executor
    

