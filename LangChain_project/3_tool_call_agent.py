from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition

load_dotenv()

class State(TypedDict):
    messages: Annotated[list,add_messages]

#function to get stock prices
@tool
def get_stock_price(symbol:str) -> float:
    '''
    Return the current price of stock given the stock symbol
    :param symbol: stock symbol
    :return: current price of the stock
    '''
    return{
        "MSFT": 200.3,
        "AAPL": 120.9,
        "AMZN": 144.4,
        "RIL": 45.5
    }.get(symbol,0.0)

tools =[get_stock_price]

llm = init_chat_model("gpt-3.5-turbo")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state:State) -> State:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("chatbot node",chatbot)
builder.add_node("tools",ToolNode(tools))


builder.add_edge(START,"chatbot node")
builder.add_conditional_edges("chatbot node",tools_condition)
builder.add_edge("tools","chatbot node")
# builder.add_edge("chatbot node",END)

graph = builder.compile()

state = graph.invoke({"messages": [{"role": "user", "content": "What is the total price of AAPL and AMZN stock right now?"}]})
print("Bot:", state["messages"][-1].content)