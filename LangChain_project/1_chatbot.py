from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages

load_dotenv()
llm = init_chat_model("gpt-3.5-turbo")

class State(TypedDict):
    messages: Annotated[list,add_messages]

def chatbot(state:State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("chatbot node",chatbot)

builder.add_edge(START,"chatbot node")
builder.add_edge("chatbot node",END)

graph = builder.compile()

# message = {"role":"user", "content":"Who walked on moon first?"}
# response = graph.invoke({"messages":[message]})
# print(response["messages"])

state = None

while True:
    in_message = input("You: ")
    if in_message.lower() in {"quit","exit"}:
        break

    if state is None:
        state : State={
            "messages": [{"role": "user", "content": in_message}]
        }
    else:
        state["messages"].append({"role": "user", "content": in_message})

    state = graph.invoke(state)
    print("Bot:", state["messages"][-1].content)

print("I am out of loop")