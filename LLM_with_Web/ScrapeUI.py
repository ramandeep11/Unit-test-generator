import gradio as gr
from agent import scrape_website

def create_ui():
    with gr.Blocks() as demo:
        with gr.Row():
            company_input = gr.Textbox(label="Company Name", placeholder="Enter company name...")
            region_input = gr.Dropdown(choices=["jp-jp", "us-en", "uk-en"], label="Region", value="us-en")
        
        output_text = gr.Textbox(label="Campaign Results", lines=5)
        submit_btn = gr.Button("Get Campaigns")
        
        def get_campaigns(company, region):
            try:
                scraper = scrape_website()
                result = scraper.get_compaign_from_website_content(company, region)
                return result
            except Exception as e:
                return f"Error: {str(e)}"
            
        submit_btn.click(
            fn=get_campaigns,
            inputs=[company_input, region_input],
            outputs=output_text
        )
    
        return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()