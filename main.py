from dotenv import load_dotenv
load_dotenv()
import os

from typing import List, Sequence

os.environ["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_PROJECT"]
os.environ["OPENAI_API_KEY"]

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, MessageGraph
from chains import generate_chain, reflect_chain

# Keys of the nodes in the graph
REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})

def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]

builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
print(graph.get_graph().print_ascii())

result = graph.invoke([HumanMessage(content="""Make this tweet better."
                                    I can’t believe that:

- Y Combinator startup school is free
- Paul Graham essays are free
- Harvard coding/AI courses are free
- Figma is free
- ProductHunt is free
- Python is free
- Andrej Karpathy is free
- ChatGPT is free

Today, anyone can build a startup, get traction, and create something incredible. You have direct access to the world's best tools and knowledge, all at no cost!

All you need to succeed: motivation, discipline and grinding 💪
                                    """)])


print(result)

if __name__ == '__main__':
    print("Hello LangGraph")
