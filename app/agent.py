"""
Production LangGraph Agent

LangGraph state machine with:
    - RAG retrieval node (grabs relevant context from Chroma before calling the LLM)
    - Primary LLM node (uses retrieved context in the prompt)
    - Fallback LLM node (same model config, separate instance — extend to a
      different/cheaper model if budget is a concern)
    - Graceful error handler
    - LangSmith tracing

Retrieval is OPTIONAL at runtime: if the vector store is empty or
unavailable the agent degrades cleanly to pure LLM mode without crashing.
"""

from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langsmith import traceable
from typing_extensions import Annotated, TypedDict

from app.config import get_settings
from app.monitoring import get_logger

logger = get_logger("agent")


# Agent State


class AgentState(TypedDict):
    """
    State passed between every node in the graph.

    `messages` uses the `add_messages` reducer so each node can APPEND
    new messages rather than replacing the whole list — this is the
    standard LangGraph pattern for conversation history.

    `context` holds raw text chunks returned by the retriever. Keeping
    them separate from messages means we can log/inspect retrieval quality
    independently from the conversation.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    context: list[str]  
    error: Optional[str]
    retry_count: int
    model_used: str


# System prompt builders


def _build_system_prompt(context: list[str]) -> str:
    """
    Build the system prompt.
    """
    if not context:
        return (
            "You are a helpful, accurate assistant. "
            "Answer the user's question to the best of your ability."
        )

    context_block = "\n\n".join(
        f"[Document {i + 1}]:\n{chunk}" for i, chunk in enumerate(context)
    )

    return (
        "You are a helpful, accurate assistant. "
        "Use the following retrieved documents to answer the user's question. "
        "If the documents don't contain enough information, say so and supplement "
        "with your general knowledge.\n\n"
        f"Retrieved context:\n{context_block}"
    )


# Agent


class ProductionAgent:
    """
    Production LangGraph agent with RAG retrieval.

    Retriever is initialised lazily and fails open (no context) if
    the vector store doesn't exist yet — lets the API start cleanly
    before the ingestion pipeline has been run.
    """

    def __init__(self):
        settings = get_settings()

        self.primary_llm = ChatOpenAI(
            model=settings.primary_model,
            temperature=0,
            timeout=30,
            max_retries=0,  # we handle retries ourselves in the graph
            api_key=settings.openai_api_key,
        )
        self.fallback_llm = ChatOpenAI(
            model=settings.fallback_model,
            temperature=0,
            timeout=30,
            max_retries=0,
            api_key=settings.openai_api_key,
        )
        self.max_retries = settings.max_retries

        self.retriever = self._init_retriever()

        self.graph = self._build_graph()

    def _init_retriever(self):
        """
        Try to initialise the RAG retriever.

        Returns None if the vector store is unavailable — agent will
        run in pure-LLM mode until the ingestion pipeline is run.
        """
        try:
            from app.rag.vectorstore import get_retriever

            retriever = get_retriever()
            logger.info("RAG retriever initialised successfully.")
            return retriever
        except Exception as e:
            logger.warning(
                "Could not initialise RAG retriever. "
                "Running in pure-LLM mode until ingestion is complete.",
                extra={"extra_data": {"error": str(e)}},
            )
            return None

    def _build_graph(self):
        """Build the LangGraph state machine."""

        # Node: retrieve

        def retrieve(state: AgentState) -> dict:
            """
            Query the vector store for context relevant to the user's message.

            Designed to NEVER crash the pipeline:
            - If retriever is None (not initialised): return empty context.
            - If retrieval throws (network/DB error): log warning, return empty context.

            This means the agent always continues, just without RAG context.
            """
            if self.retriever is None:
                return {"context": []}

            query = state["messages"][-1].content
            try:
                docs = self.retriever.invoke(query)
                context = [doc.page_content for doc in docs]
                logger.info(
                    "Retrieval complete",
                    extra={"extra_data": {"chunks_retrieved": len(context)}},
                )
                return {"context": context}
            except Exception as e:
                logger.warning(f"Retrieval failed, continuing without context: {e}")
                return {"context": []}

        # Node: process

        def process_message(state: AgentState) -> dict:
            """
            Call the primary LLM with retrieved context baked into the
            system prompt.
            """
            system_prompt = _build_system_prompt(state.get("context", []))
            messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

            try:
                response = self.primary_llm.invoke(messages)
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "primary",
                }
            except Exception as e:
                return {"error": str(e), "model_used": ""}

        # Node: fallback

        def try_fallback(state: AgentState) -> dict:
            """
            Call the fallback LLM.  Context is already in state — no need
            to re-retrieve.
            """
            system_prompt = _build_system_prompt(state.get("context", []))
            messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

            try:
                response = self.fallback_llm.invoke(messages)
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "fallback",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "",
                }

        # Node: error

        def handle_error(state: AgentState) -> dict:
            """Return a graceful degradation message rather than a 500."""
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I'm sorry, I'm having trouble processing your request "
                            "right now. Please try again in a moment."
                        )
                    )
                ],
                "model_used": "error_handler",
            }

        # Routing

        def route_after_process(state: AgentState) -> str:
            if state.get("error") is None:
                return "done"
            elif state["retry_count"] < self.max_retries:
                return "fallback"
            else:
                return "error"

        def route_after_fallback(state: AgentState) -> str:
            if state.get("error") is None:
                return "done"
            return "error"

        # Graph assembly

        graph = StateGraph(AgentState)

        graph.add_node("retrieve", retrieve)
        graph.add_node("process", process_message)
        graph.add_node("fallback", try_fallback)
        graph.add_node("error", handle_error)

        graph.add_edge(START, "retrieve")  
        graph.add_edge("retrieve", "process")  

        graph.add_conditional_edges(
            "process",
            route_after_process,
            {"done": END, "fallback": "fallback", "error": "error"},
        )
        graph.add_conditional_edges(
            "fallback",
            route_after_fallback,
            {"done": END, "error": "error"},
        )
        graph.add_edge("error", END)

        return graph.compile()

    @traceable(name="production_agent_invoke")
    def invoke(self, message: str) -> dict:
        """
        Invoke the agent with a single user message.

        Returns:
            {"response": str, "model_used": str, "error": str | None}
        """
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "context": [],
                "error": None,
                "retry_count": 0,
                "model_used": "",
            }
        )

        return {
            "response": result["messages"][-1].content,
            "model_used": result.get("model_used", "unknown"),
            "error": result.get("error"),
        }
