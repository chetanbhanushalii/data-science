from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.types import interrupt,Command
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()
memory = MemorySaver()

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

@tool
def buy_stock(symbol:str, quantity:int, total_price:float)->str:
    '''Buy stocks given the symbol and quantity'''
    decision = interrupt(f"Approve buying{quantity} of {symbol} for total price of {total_price}?")
    if decision == 'yes':
        return f"You bought{quantity} stocks of {symbol} for a total price of {total_price}"
    else:
        return "Buying declined."

tools =[get_stock_price,buy_stock]

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

graph = builder.compile(checkpointer=memory)

config = { "configurable": {'thread_id': 'HITL'}}

# step 1 : ask price of stock
state = graph.invoke({"messages": [{"role": "user", "content": "What is the total price of 10 MSFT stocks right now?"}]}, config=config)
print(state["messages"][-1].content)

# step 2 : user asks to buy stocks
state = graph.invoke({"messages": [{"role": "user", "content": "Buy 10 MSFT stocks at current price!"}]}, config=config)
print(state.get("__interrupt__"))

decision = input("Do you want to procced?(yes/no): ")
state = graph.invoke(Command(resume=decision), config=config)
print(state["messages"][-1].content)

# Check if there's an interrupt
# if state.get("next"):
#     print(f"\n⚠️ Interrupted at: {state['next']}")
#     print("Waiting for approval...")
#
#     decision = input("Do you want to proceed? (yes/no): ")
#
#     # Resume execution with the decision
#     state = graph.invoke(Command(resume=decision), config=config)
#     print(state["messages"][-1].content)