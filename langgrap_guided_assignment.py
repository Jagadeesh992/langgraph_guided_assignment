import os
import dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from IPython.display import Image, display

dotenv.load_dotenv(".env")

api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

llm = ChatOpenAI(
    openai_api_key=api_key,
    openai_api_base=base_url,
    model_name="gpt-4o-mini-2024-07-18",
)

# Node 1: calculates the average rating from total stars and number of reviews
def calculate_avg_rating(state):
    state["avg_rating"] = round(state["total_stars"] / state["num_reviews"], 2)
    return state

# Node 2: asks the LLM to write a one-line note to the vendor about its rating
def write_vendor_note(state):
    prompt = f"Write a one-line note to vendor {state['vendor_id']} about their average rating of {state['avg_rating']} out of 5."
    state["vendor_note"] = (llm | StrOutputParser()).invoke(prompt)
    return state

# Step 1: Define a state class VendorState with vendor_id (str), total_stars (int), num_reviews (int), avg_rating (float), vendor_note (str)
class VendorState(TypedDict):
    vendor_id: str
    total_stars: int
    num_reviews: int
    avg_rating: float
    vendor_note: str


# Step 2: Create an empty StateGraph called graph using VendorState
graph = StateGraph(VendorState)

# Step 3: Add the two nodes "calculate_avg_rating" and "write_vendor_note" using the functions from the setup cell
graph.add_node("calculate_avg_rating", calculate_avg_rating)
graph.add_node("write_vendor_note", write_vendor_note)

# Step 4: Add edges START -> calculate_avg_rating -> write_vendor_note -> END
graph.add_edge(START, "calculate_avg_rating")
graph.add_edge("calculate_avg_rating", "write_vendor_note")
graph.add_edge("write_vendor_note", END)

# Step 5: Compile the graph into workflow
workflow = graph.compile()
# Step 6: Call workflow.invoke() with vendor_id "V3", total_stars=1840, num_reviews=460
result = workflow.invoke({
    "vendor_id": "V3",
    "total_stars": 1840,
    "num_reviews": 460,
})
print(result)