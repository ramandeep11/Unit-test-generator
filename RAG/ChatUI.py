import gradio as gr
import requests

def rag_application(url, query, conversation_history):
    try:
        # Make the request to the tesseract service
        tesseract_response = requests.post(
            "http://localhost:8080/tesseract",
            json={"url": url, "query": query}
        )
        tesseract_response.raise_for_status()
        
        # Extract the response from the JSON data
        tesseract_data = tesseract_response.json()
        response = tesseract_data['answer']
    except requests.RequestException as e:
        response = f"Error connecting to tesseract endpoint: {str(e)}"
        print(response)

    # Append the user query and AI response to the conversation history
    conversation_history.append([query, response])

    return conversation_history, conversation_history  

# Set up the Gradio interface
with gr.Blocks() as iface:

    # Add a title to the interface
    gr.Markdown("# RAG Application")
    conversation_history = gr.State([])  # Initialize an empty conversation history

    with gr.Row():
        url_input = gr.Textbox(label="Enter URL", placeholder="Enter the URL here")
        query_input = gr.Textbox(label="Enter Query", placeholder="Enter your question here")

    # Chatbot component for displaying the conversation
    chatbot = gr.Chatbot(label="Conversation")

    # Button to submit the query
    submit_button = gr.Button("Submit")
    submit_button.click(
        fn=rag_application,
        inputs=[url_input, query_input, conversation_history],
        outputs=[chatbot, conversation_history]
    )

    # Button to ask further questions after the first response
    with gr.Row():
        follow_up_query = gr.Textbox(label="Ask a follow-up question", placeholder="Continue the conversation here")
        follow_up_button = gr.Button("Ask Follow-Up")
        follow_up_button.click(
            fn=rag_application,
            inputs=[url_input, follow_up_query, conversation_history],
            outputs=[chatbot, conversation_history]
        )

iface.launch()