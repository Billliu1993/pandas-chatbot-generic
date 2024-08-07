from langchain.agents import tool
import pandas as pd
import streamlit as st
from plotly.graph_objects import Figure
from plotly.io import from_json
import plotly.graph_objects as go
import json
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.llm_math.base import LLMMathChain
from langchain_core.tools import Tool


class CalculatorInput(BaseModel):
    question: str = Field()

@tool
def plot_chart(data: str) -> int:
    """Plots json data using plotly Figure. Use it only for plotting charts and graphs."""
    # Load JSON data
    figure_dict = json.loads(data)
    # Create Figure object from JSON data
    try:
        fig = from_json(json.dumps(figure_dict))
        st.plotly_chart(fig)
    except:
        fig = go.Figure(data)
        st.plotly_chart(fig)

def get_calculator_tool(llm):

    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    math_tool = Tool.from_function(
            name="Calculator",
            func=llm_math_chain.run,
            description="Useful for when you need to answer numeric questions. This tool is only for math questions and nothing else. Only input math expressions, without text",
            args_schema=CalculatorInput,
            coroutine=llm_math_chain.arun,
            )
    
    return math_tool
