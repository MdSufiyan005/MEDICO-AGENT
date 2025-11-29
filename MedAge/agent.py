"""
Agent / Graph builder and run-graph helper.
"""
import logging
import uuid
from typing import Any, List, Dict
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, Runnable, RunnableLambda
from langchain_core.tools import tool
from langchain_groq import ChatGroq

import MedAge.tools as tools_module
from MedAge.config import GROQ_API_KEY
logger = logging.getLogger(__name__)

# Build the prompt (keeps same content/rules)
RadiologyAgent_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
You are a radiologist assistant coordinating medical image analysis.
YOUR WORKFLOW:
1. Receive user's question and medical images (if provided) understand the question and step by step follow it accordingly
2. If images are provided: Send to GREAT_BRAIN_SPECIALIST tool (provide question + image paths)
3. If no images: Use `search_medical_info` (provide ONLY `query`) for recent research, news, or specific articles.
4. For other medical reasoning/questions (no images), use `GREAT_BRAIN_SPECIALIST` (set images=[]).
5. If the question is specifically about lung diseases (e.g., causes, treatments, pulmonary health from the 'Comprehensive Review on Lung Disease' document): Use LUNG_DISEASE_RAG tool first for precise retrieval from the document
6. Ask follow-up questions to the specialist if needed for clarity
7. Verify key findings using search_medical_info tool
8. Summarize everything clearly for the user
- If the user asks for recent information, articles, or research (which the model may not have), YOU MUST use `search_medical_info` immediately.
- If you need to search, call `search_medical_info` immediately. Do not explain you are going to search.
- Do NOT repeat the same search query multiple times. If a search yields no results, stop and inform the user.
- `search_medical_info` strictly accepts ONLY a `query` string.
- Do NOT use the Final Response Format when calling tools. Output ONLY the tool call.
- CRITICAL: `search_medical_info` accepts ONLY `query`. Do NOT pass `images` to it.
- Do NOT output text representations of tools like `<function=...>`.
FINAL RESPONSE FORMAT (Use ONLY after tool execution):
**Assessment:** [Specialist's findings or answer]
**Evidence:** [Search results supporting this, if applicable]
**Recommendation:** [Next steps, if applicable]
**Disclaimer:** This is informational only - consult a qualified physician for diagnosis.
SAFETY GUIDELINES:
- Always include the disclaimer
- If images are unclear, request better quality
- Be conservative with recommendations
- Only cite actual search results, never invent sources
- Flag urgent findings clearly
- Acknowledge uncertainties openly
- Handle both image-based and text-only queries appropriately
Keep responses concise, professional, and medically cautious.
"""
    ),
    MessagesPlaceholder(variable_name="messages"),
])

# Tools list
tools = [tools_module.search_medical_info, tools_module.great_brain_specialist, tools_module.lung_disease_rag]

# LLM (Groq-compatible ChatGroq)
api_key = GROQ_API_KEY
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=1.0, api_key=api_key)

assistant_runnable = RadiologyAgent_prompt | llm.bind_tools(tools)

# Graph builder
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

def hand_tool_error(state) -> dict:
    error = state.get("error")
    tool_call = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error occurred while executing tool {tc['name']}: {str(error)}",
                tool_call_id=tc['id']
            )
            for tc in tool_call
        ]
    }

def create_tool_node(tools: List[Any]):
    def log_tool_execution(state):
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            logger.info("🛠️ TOOL NODE: Executing tools")
            for tool_call in last_message.tool_calls:
                logger.info(f" • Tool: {tool_call['name']} Args: {tool_call.get('args', {})}")
        return state
    def log_tool_results(state, num_tool_calls):
        tool_messages = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
        if tool_messages:
            logger.info("📤 TOOL OUTPUTS RECEIVED:")
            recent_tool_messages = tool_messages[-num_tool_calls:] if num_tool_calls > 0 else tool_messages
            for i, tool_msg in enumerate(recent_tool_messages, 1):
                content = tool_msg.content
                preview = content[:500] if len(str(content)) > 500 else content
                logger.info(f" Output {i}: {preview}")
        return state
    def wrapper(state):
        last_message = state["messages"][-1]
        num_tool_calls = len(last_message.tool_calls) if hasattr(last_message, 'tool_calls') and last_message.tool_calls else 0
        log_tool_execution(state)
        result = ToolNode(tools).invoke(state)
        state["messages"] = result["messages"]
        log_tool_results(state, num_tool_calls)
        return result
    return RunnableLambda(wrapper).with_fallbacks(
    [RunnableLambda(hand_tool_error)],
    exception_key="error",
)

class RadiologyAgent:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State):
        logger.info("🤖 ASSISTANT NODE: Processing...")
        user_messages = [msg for msg in state["messages"] if hasattr(msg, 'content') and not hasattr(msg, 'tool_calls')]
        if user_messages:
            logger.info(f"📨 Latest user message: {user_messages[-1].content[:200]}...")
        iteration = 0
        max_iterations = 5
        result = None
        while iteration < max_iterations:
            iteration += 1
            if iteration > 1:
                logger.info(f" 🔄 Retry iteration {iteration}...")
            try:
                result = self.runnable.invoke({"messages": state["messages"]})
                if hasattr(result, 'tool_calls') and result.tool_calls:
                    if result.content and result.content.strip():
                        logger.warning(" ⚠️ Warning: Tool calls mixed with content. Retrying...")
                        messages = state["messages"] + [HumanMessage(
                            content="IMPORTANT: When calling tools, provide ONLY the tool call. Do not include any explanatory text in the same response."
                        )]
                        state = {**state, "messages": messages}
                        continue
                    logger.info(" ✅ Agent Response: Tool calls detected")
                    for i, tool_call in enumerate(result.tool_calls, 1):
                        logger.info(f" {i}. {tool_call['name']} Args: {tool_call.get('args', {})}")
                    break
                elif result.content:
                    logger.info(" ✅ Agent Response: Final answer (no tools needed)")
                    break
                else:
                    logger.warning(" ⚠️ Empty response, requesting retry...")
                    messages = state["messages"] + [HumanMessage(content="Respond with a real output")]
                    state = {**state, "messages": messages}
            except Exception as e:
                error_str = str(e)
                if "tool_use_failed" in error_str or "BadRequestError" in error_str:
                    logger.warning(f" ⚠️ Tool call error: {error_str[:200]}...")
                    messages = state["messages"] + [HumanMessage(
                        content="CRITICAL: Provide ONLY tool calls without explanatory text. Separate tool calls from text responses."
                    )]
                    state = {**state, "messages": messages}
                    continue
                else:
                    raise
        if iteration >= max_iterations:
            logger.error(" ❌ Max iterations reached.")
            if result is None:
                result = AIMessage(content="I apologize, but I encountered an error processing your request. Please try rephrasing your question.")
        if result is None:
            result = AIMessage(content="I encountered an error processing your request. Please try again.")
        return {"messages": result}

# Build graph
builder = StateGraph(State)
builder.add_node("assistant", RadiologyAgent(assistant_runnable))
builder.add_node("tools", create_tool_node(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
