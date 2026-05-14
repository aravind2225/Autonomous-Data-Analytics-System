from langgraph.graph import StateGraph, END

from state.shared_state import SharedState

from agents.ingestion_agent import IngestionAgent
from agents.cleaning_agent import CleaningAgent
from agents.eda_agent import EDAAgent
from agents.stats_agent import StatsAgent
from agents.visualization_agent import VisualizationAgent
from agents.insights_agent import InsightsAgent


ingestion_agent = IngestionAgent()
cleaning_agent = CleaningAgent()
eda_agent = EDAAgent()
stats_agent = StatsAgent()
visualization_agent = VisualizationAgent()
insights_agent = InsightsAgent()


def ingestion_node(state):
    return ingestion_agent.run(state['file'])


def cleaning_node(state):
    return cleaning_agent.run(state)


def eda_node(state):
    return eda_agent.run(state)


def stats_node(state):
    return stats_agent.run(state)


def visualization_node(state):
    return visualization_agent.run(state)


def insights_node(state):
    return insights_agent.run(state)


builder = StateGraph(SharedState)

builder.add_node("ingestion", ingestion_node)
builder.add_node("cleaning", cleaning_node)
builder.add_node("eda", eda_node)
builder.add_node("stats", stats_node)
builder.add_node("visualization", visualization_node)
builder.add_node("insights", insights_node)

builder.set_entry_point("ingestion")

builder.add_edge("ingestion", "cleaning")
builder.add_edge("cleaning", "eda")
builder.add_edge("eda", "stats")
builder.add_edge("stats", "visualization")
builder.add_edge("visualization", "insights")
builder.add_edge("insights", END)

analytics_graph = builder.compile()