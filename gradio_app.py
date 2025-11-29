import logging
import uuid
from typing import List
from MedAge.agent import graph
from langchain_core.messages import HumanMessage, ToolMessage
import gradio as gr

logger = logging.getLogger(__name__)

from MedAge.tools import search_results_store

def _run_graph_and_collect(input_content: str, session_id: str):
    """
    Runs the graph for the given input_content (HumanMessage) and returns
    the final assessment and metadata.
    """
    config = {"configurable": {"thread_id": session_id},"recursion_limit": 100}
    input_data = {"messages": [HumanMessage(content=input_content)]}
    session_search_results: List[str] = []
    final_state = None
    
    for event in graph.stream(input_data, config=config):
        node_name = list(event.keys())[0]
        logger.info(f" → Node '{node_name}' completed")
        final_state = event[node_name]
        
        # collect tool messages
        if isinstance(final_state, dict) and "messages" in final_state:
            for msg in final_state["messages"]:
                if isinstance(msg, ToolMessage) and isinstance(msg.content, list):
                    session_search_results.extend(msg.content)
    
    # Normalize messages
    if final_state:
        if isinstance(final_state, dict):
            final_messages = final_state.get("messages", [])
        elif isinstance(final_state, list):
            final_messages = final_state
        elif hasattr(final_state, 'content'):
            final_messages = [final_state]
        else:
            final_messages = []
        
        if not isinstance(final_messages, list):
            final_messages = [final_messages] if final_messages else []
        
        assistant_responses = [
            msg for msg in final_messages
            if hasattr(msg, 'content') and msg.content and not isinstance(msg, ToolMessage)
        ]
        
        if assistant_responses:
            final_response = assistant_responses[-1]
            assessment = final_response.content
        else:
            assessment = "No response generated"
        
        total_messages = len(final_messages)
        tool_calls_made = len([msg for msg in final_messages if isinstance(msg, ToolMessage)])
        
        search_results_store[session_id] = session_search_results
        
        return {
            "assessment": assessment,
            "total_messages": total_messages,
            "tool_calls_made": tool_calls_made,
            "search_results": session_search_results
        }
    
    raise RuntimeError("No result obtained from the processing graph")


def chat_json(question: str, image=None):
    """
    Modified to accept question and image as separate parameters from Gradio.
    """
    try:
        session_id = str(uuid.uuid4())
        image_paths = []
        
        if image is not None:
            image_path = f"/tmp/{uuid.uuid4()}.png"
            image.save(image_path)
            image_paths.append(image_path)

        # Improved input formatting - clearer structure for the agent
        if image_paths:
            # Pass image paths as a Python list, not a string representation
            input_content = f"""Question: {question}

Images provided: Yes
Image paths: {', '.join(image_paths)}

Instructions: Analyze the provided medical images and answer the question."""
        else:
            # Make it clear no images are provided
            input_content = f"""Question: {question}

Images provided: No

Instructions: Answer this medical question. If this requires recent information, articles, or research, use the search_medical_info tool."""
        
        logger.info(f"📤 Sending to agent: {input_content[:200]}...")
        result = _run_graph_and_collect(input_content, session_id)
        
        # Format the output as a readable string
        formatted_output = f"""### Medical Assessment

**Session ID:** {session_id}

**Question:** {question}

---

{result["assessment"]}

---

**Metadata:**
- Total Messages: {result["total_messages"]}
- Tool Calls Made: {result["tool_calls_made"]}
- Search Results Found: {len(result["search_results"])}
"""
        
        return formatted_output
        
    except Exception as e:
        logger.exception("Error in chat_json")
        return f"❌ **Error occurred:**\n\n{str(e)}"


# ------------------- GRADIO UI ------------------- #

css = """
#outbox {
    background: #ffffff !important;
    color: #000000 !important;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e5e5e5;
    max-height: 450px;
    overflow-y: auto;
    font-size: 17px;
    line-height: 1.7;
}

#outbox * {
    color: #000000 !important;
}

#outbox h3 {
    color: #1a1a1a !important;
    margin-top: 0;
}

#outbox strong {
    color: #2c3e50 !important;
}
"""

with gr.Blocks() as demo:
    gr.Markdown("## 🏥 Medical Agent — Text + Optional Image")
    gr.Markdown("Ask medical questions with or without images. For recent articles/research, just ask!")
    
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(
                label="Question (required)",
                placeholder="e.g., 'Can you provide 5 article names recently published regarding human life?' or 'Analyze this X-ray'",
                lines=3
            )
            
            image = gr.Image(
                label="Optional Image",
                type="pil"
            )
            
            submit = gr.Button("Submit", variant="primary")
    
    with gr.Row():
        output = gr.Markdown(label="Model Output", elem_id="outbox")
    
    # Example queries
    gr.Examples(
        examples=[
            ["Can you provide 5 article names recently published regarding cancer research?", None],
            ["What are the latest developments in AI healthcare?", None],
            ["Explain the causes of pneumonia", None],
        ],
        inputs=[question, image]
    )
    
    submit.click(fn=chat_json, inputs=[question, image], outputs=output)

demo.launch(debug=True, share=True,css=css)