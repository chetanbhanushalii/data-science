from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

PRICES = {
    "MSFT": 200.3,
    "AAPL": 120.9,
    "AMZN": 144.4,
    "RIL": 45.5
}

@tool
def get_stock_price(symbol: str) -> float:
    """Return the current price given the stock symbol."""
    return float(PRICES.get(symbol.upper(), 0.0))

@tool
def buy_stock(symbol: str, quantity: int) -> str:
    """Buy stocks given the symbol and quantity (requires approval)."""
    price = get_stock_price.invoke({"symbol": symbol})
    total_price = float(price) * int(quantity)

    decision = interrupt(
        f"Approve buying {quantity} of {symbol.upper()} for total price {total_price:.2f}?"
    )

    if str(decision).strip().lower() == "yes":
        return f"✅ Bought {quantity} shares of {symbol.upper()} for total {total_price:.2f}"
    return "❌ Buying declined."

tools = [get_stock_price, buy_stock]

llm = init_chat_model("gpt-3.5-turbo")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State) -> State:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)  # routes to "tools" or END
builder.add_edge("tools", "chatbot")

graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "HITL"}}

# Step 1
state = graph.invoke(
    {"messages": [{"role": "user", "content": "What is the total price of 10 MSFT stocks right now?"}]},
    config=config
)
print(state["messages"][-1].content)

# Step 2
state = graph.invoke(
    {"messages": [{"role": "user", "content": "Buy 10 MSFT stocks at current price!"}]},
    config=config
)

# If interrupted, resume
if state.get("__interrupt__"):
    print(state["__interrupt__"])
    decision = input("Do you want to proceed? (yes/no): ")
    state = graph.invoke(Command(resume=decision), config=config)
    print(state["messages"][-1].content)
else:
    print(state["messages"][-1].content)
