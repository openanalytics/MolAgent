import json
import mimetypes
import os
import re
import shutil
import threading
from datetime import datetime
from typing import Optional
from huggingface_hub import login

import gradio as gr
from dotenv import load_dotenv
os.getenv("HF_TOKEN")
from smolagents import LiteLLMModel
load_dotenv(override=True)


#from huggingface_hub import login
from smolagents import CodeAgent, LiteLLMModel, OpenAIServerModel, Tool
from smolagents.agent_types import (
    AgentAudio,
    AgentImage,
    AgentText,
    handle_agent_output_types,
)
from smolagents.gradio_ui import stream_to_gradio

from agents import get_data_agent, get_mcp_model_agent, AUTHORIZED_IMPORTS
from Tools.evaluation_tools import read_json_file, load_automol_model, guide_prompt, automol_predict, write_to_file, wait_for_llm_rate

from mcp_server.manage_mcp_tools import MCPServerControl

from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    LiteLLMModel,
    ToolCollection,
    WebSearchTool,
    DuckDuckGoSearchTool
)
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SearchToolTavily,
    SimpleTextBrowser,
    VisitToolTavily,
)
from Tools.chembl_smolagents_tools import (
		ChEMBLMoleculeSearchTool,
        chembl_activity_search,
        chembl_target_search,
        chembl_similarity_search,
        chembl_drug_indication_search,
        chembl_assay_search,
        chembl_molecule_image,
        ChEMBLMolecularUtilsTool,
        chembl_tissue_search,
        chembl_cell_line_search,
        chembl_document_search,
        chembl_list_available_resources

)

LOCAL_FILES = os.getenv("LOCAL_FILES", None)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL_ID = os.getenv("MODEL_ID", "openrouter/meta-llama/llama-4-maverick")
model = LiteLLMModel(model_id=MODEL_ID)

append_answer_lock = threading.Lock()

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": None,
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)
"""
# Replace previous model initialization with local vLLM server model
model = OpenAIServerModel(
    model_id="jan-hq/Qwen3-14B-v0.2-deepresearch-no-think-100-step",
    api_base="http://localhost:8000/v1/",  # Local vLLM server URL
    api_key="EMPTY",  # vLLM uses "EMPTY" as the default API key
    custom_role_conversions=custom_role_conversions,
    temperature=0.7,
    top_p=0.8,
)
"""
text_limit = 20000
browser = SimpleTextBrowser(**BROWSER_CONFIG)
web_search = SearchToolTavily()

def create_agent():
    mcp_control=MCPServerControl(['http://127.0.0.1:8000/sse'])
    mcp_data_tools=mcp_control.get_tools()
    
    #opening mcp server mcp_model_control, don't forget to close (see end of notebook)
    mcp_model_control=MCPServerControl(['http://127.0.0.1:8001/sse'])
    mcp_model_tools=mcp_model_control.get_tools()

    web_tools_for_agent = [
        web_search,  # Using Tavily search tool
        VisitToolTavily(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]
	
    CHEMBL_TOOLS= [
        ChEMBLMoleculeSearchTool(),
        chembl_activity_search,
        chembl_target_search,
        chembl_similarity_search,
        chembl_drug_indication_search,
        chembl_assay_search,
        chembl_molecule_image,
        ChEMBLMolecularUtilsTool(),
        chembl_tissue_search,
        chembl_cell_line_search,
        chembl_document_search,
        chembl_list_available_resources
    ]

    manager_agent = CodeAgent(
        tools=[read_json_file, write_to_file, load_automol_model, automol_predict]+web_tools_for_agent+CHEMBL_TOOLS,
        model=model,
        max_steps=15,
        #managed_agents=[create_text_webbrowser_agent(model), get_data_agent(model,mcp_data_tools), get_mcp_model_agent(model,mcp_model_tools)],
        managed_agents=[get_data_agent(model,mcp_data_tools), get_mcp_model_agent(model,mcp_model_tools)],
        description=guide_prompt(),
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        add_base_tools=False
    )
    return manager_agent

def find_image_files(text):
    """Find image file paths in text content, including matplotlib and RDKit generated images"""
    if not isinstance(text, str):
        return []

    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.svg', '.pdf']
    found_images = []

    # Look for file paths in various formats
    patterns = [
        r'[^\s<>"\']*\.[a-zA-Z]{3,4}',  # Basic file pattern
        r'"[^"]*\.[a-zA-Z]{3,4}"',      # Quoted paths
        r"'[^']*\.[a-zA-Z]{3,4}'",      # Single quoted paths
        r'[^\s<>"\']*figure[^\s<>"\']*\.[a-zA-Z]{3,4}',  # Matplotlib figure patterns
        r'[^\s<>"\']*plot[^\s<>"\']*\.[a-zA-Z]{3,4}',    # Plot file patterns
        r'[^\s<>"\']*mol[^\s<>"\']*\.[a-zA-Z]{3,4}',     # Molecule file patterns
        r'[^\s<>"\']*chart[^\s<>"\']*\.[a-zA-Z]{3,4}',   # Chart file patterns
        r'[^\s<>"\']*graph[^\s<>"\']*\.[a-zA-Z]{3,4}',   # Graph file patterns
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            clean_path = match.strip('"\'`')

            # Check if it has an image extension
            if any(clean_path.lower().endswith(ext) for ext in image_extensions):
                # Try different potential paths
                potential_paths = [
                    clean_path,
                    os.path.join(os.getcwd(), clean_path),
                    os.path.join("downloads", os.path.basename(clean_path)),
                    os.path.join("downloads_folder", os.path.basename(clean_path)),
                    os.path.basename(clean_path),
                    # Common matplotlib save locations
                    os.path.join("figures", os.path.basename(clean_path)),
                    os.path.join("plots", os.path.basename(clean_path)),
                    os.path.join("images", os.path.basename(clean_path)),
                    # Common RDKit save locations
                    os.path.join("molecules", os.path.basename(clean_path)),
                    os.path.join("mol_images", os.path.basename(clean_path)),
                ]

                for path in potential_paths:
                    if os.path.exists(path) and os.path.isfile(path):
                        abs_path = os.path.abspath(path)
                        if abs_path not in found_images:
                            found_images.append(abs_path)
                        break

    # Also scan for recently created image files in common directories
    # This helps catch matplotlib/RDKit images that might not be explicitly mentioned in text
    scan_directories = [
        ".",  # Current directory
        "downloads",
        "downloads_folder",
        "figures",
        "plots",
        "images",
        "molecules",
        "mol_images",
    ]

    import time
    current_time = time.time()

    for directory in scan_directories:
        if os.path.exists(directory) and os.path.isdir(directory):
            try:
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path):
                        # Check if it's a recent image file (created in last 30 seconds)
                        file_age = current_time - os.path.getctime(file_path)
                        if file_age < 30:  # Recent file
                            if any(file.lower().endswith(ext) for ext in image_extensions):
                                # Check if it's likely a generated plot/molecule
                                if any(keyword in file.lower() for keyword in [
                                    'plot', 'figure', 'chart', 'graph', 'mol', 'molecule',
                                    'structure', 'compound', 'drug', 'similarity', 'activity'
                                ]):
                                    abs_path = os.path.abspath(file_path)
                                    if abs_path not in found_images:
                                        found_images.append(abs_path)
                                        print(f"Found recent generated image: {abs_path}")
            except (PermissionError, OSError):
                continue

    return found_images


class GradioUI:
    """A one-line interface to launch your agent in Gradio"""

    def __init__(self, file_upload_folder: str | None = None):
        self.file_upload_folder = file_upload_folder
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)

    def interact_with_agent(self, prompt, messages, session_state):
        # Get or create session-specific agent
        if "agent" not in session_state:
            print("No existing agent in session, creating new one...")
            session_state["agent"] = create_agent()

        try:
            # log the existence of agent memory
            has_memory = hasattr(session_state["agent"], "memory")
            print(f"Agent has memory: {has_memory}")
            if has_memory:
                print(f"Memory type: {type(session_state['agent'].memory)}")

            now = datetime.now()
            current_date = f"Today is {now.strftime('%A, %B %d, %Y')}"

            # Enhanced prompt with image saving/loading instructions
            enhanced_prompt = f"""{current_date}. {prompt}

IMPORTANT IMAGE HANDLING INSTRUCTIONS:
- When creating visualizations (plots, charts, molecular structures), ALWAYS save them to files and use descriptive filenames that include the content type (e.g., 'molecule_structure_moleceuleXXX.png', 'activity_plot_compound_analysis.png')
- Save images in the current directory or 'downloads_folder' for easy detection
- Supported formats: PNG, SVG, JPG, PDF (PNG recommended for plots, SVG for molecules)
- After saving, mention the filename in your response so it can be displayed
- For matplotlib: use plt.savefig('filename.png', dpi=300, bbox_inches='tight')
- For RDKit molecules: use Draw.MolToFile(mol, 'molecule_name.png')
- For ChEMBL images: save retrieved images to local files when possible"""

            # Show only the user's original prompt in the chat UI
            messages.append(gr.ChatMessage(role="user", content=prompt))
            yield messages

            # Send the enhanced prompt to the agent
            agent_responses = []
            processed_images = set()  # Track images already processed to prevent duplicates

            for msg in stream_to_gradio(
                session_state["agent"], task=enhanced_prompt, reset_agent_memory=False
            ):
                agent_responses.append(msg)

                # Add the text message first (always)
                messages.append(msg)
                yield messages

            # Process images after all agent responses are collected to prevent duplicates
            if agent_responses:
                all_found_images = []

                # Collect all images mentioned in agent responses
                for response in agent_responses:
                    if hasattr(response, 'content') and isinstance(response.content, str):
                        response_images = find_image_files(response.content)
                        all_found_images.extend(response_images)

                # Remove duplicates while preserving order
                unique_images = []
                for img in all_found_images:
                    if img not in unique_images:
                        unique_images.append(img)

                # Add each unique image as a separate message
                if unique_images:
                    print(f"Processing {len(unique_images)} unique images: {unique_images}")

                    for img_path in unique_images:
                        # Verify image file exists
                        if not os.path.exists(img_path):
                            continue

                        img_name = os.path.basename(img_path)
                        print(f"Adding image to chat: {img_path}")

                        # Determine the type of image based on filename/path
                        img_type = "🖼️ Generated Image"
                        if any(keyword in img_name.lower() for keyword in ['plot', 'figure', 'chart', 'graph']):
                            img_type = "📊 Generated Plot/Chart"
                        elif any(keyword in img_name.lower() for keyword in ['mol', 'molecule', 'structure', 'compound']):
                            img_type = "🧬 Molecular Structure"
                        elif 'similarity' in img_name.lower():
                            img_type = "🔍 Similarity Analysis"
                        elif 'activity' in img_name.lower():
                            img_type = "📈 Activity Data"

                        # Create message with image - try different approaches
                        image_added = False

                        # Method 1: Base64 encoding for direct embedding (most reliable)
                        try:
                            import base64
                            with open(img_path, 'rb') as img_file:
                                img_data = base64.b64encode(img_file.read()).decode()
                                # Determine MIME type
                                if img_path.lower().endswith('.png'):
                                    mime_type = 'image/png'
                                elif img_path.lower().endswith(('.jpg', '.jpeg')):
                                    mime_type = 'image/jpeg'
                                elif img_path.lower().endswith('.svg'):
                                    mime_type = 'image/svg+xml'
                                else:
                                    mime_type = 'image/png'

                                img_content = f"![{img_name}](data:{mime_type};base64,{img_data})\n\n{img_type}: **{img_name}**"
                                img_message = gr.ChatMessage(role="assistant", content=img_content)
                                messages.append(img_message)
                                image_added = True
                                print(f"✅ Added image using base64 encoding: {img_path}")
                        except Exception as e1:
                            print(f"Method 1 (base64) failed: {e1}")

                        if not image_added:
                            try:
                                # Method 2: Try markdown image with file:// URL
                                img_content = f"![{img_name}](file://{img_path})\n\n{img_type}: **{img_name}**"
                                img_message = gr.ChatMessage(role="assistant", content=img_content)
                                messages.append(img_message)
                                image_added = True
                                print(f"✅ Added image using markdown file URL: {img_path}")
                            except Exception as e2:
                                print(f"Method 2 (markdown) failed: {e2}")

                        if not image_added:
                            # Method 3: Fallback - just mention the file with instructions
                            fallback_content = f"{img_type}: **{img_name}**\n\n📁 **Location**: `{img_path}`\n\n💡 **Tip**: The image has been saved to your local directory. You can view it by:\n1. Opening the file path above in your file explorer\n2. Or copying the path and opening it in an image viewer\n\n*The image contains visualization results from your request.*"
                            img_message = gr.ChatMessage(role="assistant", content=fallback_content)
                            messages.append(img_message)
                            print(f"✅ Added image fallback message: {img_path}")

                        yield messages

        except Exception as e:
            print(f"Error in interaction: {str(e)}")
            import traceback
            traceback.print_exc()
            # Add error message to chat
            error_msg = gr.ChatMessage(
                role="assistant",
                content=f"❌ **Error occurred**: {str(e)}\n\nPlease try again or rephrase your request."
            )
            messages.append(error_msg)
            yield messages

    def upload_file(
        self,
        file,
        file_uploads_log,
        allowed_file_types=[
              "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "text/csv",  # Added CSV support
            "application/csv",  # Alternative CSV MIME type
            "image/png",  # Added image support
            "image/jpeg",
            "image/jpg",
            "image/gif",
            "image/bmp",
            "image/tiff",
            "image/webp",
        ],
    ):
        """
        Handle file uploads, default allowed types are .pdf, .docx, and .txt
        """
        if file is None:
            return gr.Textbox("No file uploaded", visible=True), file_uploads_log

        try:
            mime_type, _ = mimetypes.guess_type(file.name)
        except Exception as e:
            return gr.Textbox(f"Error: {e}", visible=True), file_uploads_log

        if mime_type not in allowed_file_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        # Sanitize file name
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores

        type_to_ext = {}
        for ext, t in mimetypes.types_map.items():
            if t not in type_to_ext:
                type_to_ext[t] = ext

        # Ensure the extension correlates to the mime type
        sanitized_name = sanitized_name.split(".")[:-1]
        sanitized_name.append("" + type_to_ext[mime_type])
        sanitized_name = "".join(sanitized_name)

        # Save the uploaded file to the specified folder
        file_path = os.path.join(self.file_upload_folder, os.path.basename(sanitized_name))
        shutil.copy(file.name, file_path)

        return gr.Textbox(f"File uploaded: {file_path}", visible=True), file_uploads_log + [file_path]

    def log_user_message(self, text_input, file_uploads_log):
        return (
            text_input
            + (
                f"\nYou have been provided with these files, which might be helpful or not: {file_uploads_log}"
                if len(file_uploads_log) > 0
                else ""
            ),
            gr.Textbox(
                value="",
                interactive=False,
                placeholder="Please wait while Steps are getting populated",
            ),
            gr.Button(interactive=False),
        )

    def detect_device(self, request: gr.Request):
        # Check whether the user device is a mobile or a computer

        if not request:
            return "Unknown device"
        # Method 1: Check sec-ch-ua-mobile header
        is_mobile_header = request.headers.get("sec-ch-ua-mobile")
        if is_mobile_header:
            return "Mobile" if "?1" in is_mobile_header else "Desktop"

        # Method 2: Check user-agent string
        user_agent = request.headers.get("user-agent", "").lower()
        mobile_keywords = ["android", "iphone", "ipad", "mobile", "phone"]

        if any(keyword in user_agent for keyword in mobile_keywords):
            return "Mobile"

        # Method 3: Check platform
        platform = request.headers.get("sec-ch-ua-platform", "").lower()
        if platform:
            if platform in ['"android"', '"ios"']:
                return "Mobile"
            elif platform in ['"windows"', '"macos"', '"linux"']:
                return "Desktop"

        # Default case if no clear indicators
        return "Desktop"

    def launch(self, **kwargs):
        with gr.Blocks(
            theme="ocean",
            fill_height=True,
            css="""
            .primary-button {
                background-color: #ff5c00 !important;
                border-color: #ff5c00 !important;
            }
            .primary-button:hover {
                background-color: #e65300 !important;
                border-color: #e65300 !important;
            }
            /* Ensure images are displayed properly in chat */
            .chatbot img {
                max-width: 100% !important;
                height: auto !important;
                border-radius: 8px !important;
                margin: 8px 0 !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            }
            /* Handle various image sources */
            .chatbot .message img {
                max-width: 500px !important;
                height: auto !important;
                border-radius: 8px !important;
                margin: 8px 0 !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            }
            /* Style for molecular structures and scientific images */
            .chatbot .message img[src*="mol"], .chatbot .message img[src*="molecule"] {
                max-width: 400px !important;
                background: white !important;
                padding: 10px !important;
            }
        """,
        ) as demo:
            # Different layouts for mobile and computer devices
            @gr.render()
            def layout(request: gr.Request):
                device = self.detect_device(request)
                print(f"device - {device}")
                # Render layout with sidebar
                if device == "Desktop":
                    with gr.Blocks(
                        fill_height=True,
                    ):
                        file_uploads_log = gr.State([])
                        with gr.Sidebar():
                            gr.Markdown("""# 🔬 MolAgent ✨

A powerful AI research assistant for molecular and pharmaceutical research.

**Features:**
- 🧬 ChEMBL database integration
- 📊 Data visualization & plots
- 🔍 Web search capabilities
- 🖼️ Automatic image display

""")

                            with gr.Group():
                                gr.Markdown("**Your request**", container=True)
                                text_input = gr.Textbox(
                                    lines=3,
                                    label="Your request",
                                    container=False,
                                    placeholder="Enter your prompt here and press Shift+Enter or press the button",
                                )
                                launch_research_btn = gr.Button(
                                    "Run", variant="primary", elem_classes=["primary-button"]
                                )

                            # If an upload folder is provided, enable the upload feature
                            if self.file_upload_folder is not None:
                                upload_file = gr.File(
                                    label="Upload a file (PDF, DOCX, TXT, CSV, Images)",
                                    file_types=[".pdf", ".docx", ".txt", ".csv", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]
                                )
                                upload_status = gr.Textbox(
                                    label="Upload Status",
                                    interactive=False,
                                    visible=False,
                                )
                                upload_file.change(
                                    self.upload_file,
                                    [upload_file, file_uploads_log],
                                    [upload_status, file_uploads_log],
                                )

                            gr.HTML("<br><br><h4><center>Powered by:</center></h4>")
                            with gr.Row():
                                gr.HTML("""<div style="display: flex; align-items: center; gap: 8px; font-family: system-ui, -apple-system, sans-serif;">
                        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png" style="width: 32px; height: 32px; object-fit: contain;" alt="logo">
                        <a target="_blank" href="https://github.com/huggingface/smolagents"><b>huggingface/smolagents</b></a>
                        </div>""")

                        # Add session state to store session-specific data
                        session_state = gr.State({})  # Initialize empty state for each session
                        stored_messages = gr.State([])
                        chatbot = gr.Chatbot(
                            label="MolAgent",
                            type="messages",
                            avatar_images=(
                                None,
                                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                            ),
                            resizeable=False,
                            scale=1,
                            elem_id="my-chatbot",
                            show_copy_button=True,  # Allow copying of messages
                        )

                        text_input.submit(
                            self.log_user_message,
                            [text_input, file_uploads_log],
                            [stored_messages, text_input, launch_research_btn],
                        ).then(
                            self.interact_with_agent,
                            [stored_messages, chatbot, session_state],
                            [chatbot],
                        ).then(
                            lambda: (
                                gr.Textbox(
                                    interactive=True,
                                    placeholder="Enter your prompt here and press the button",
                                ),
                                gr.Button(interactive=True),
                            ),
                            None,
                            [text_input, launch_research_btn],
                        )
                        launch_research_btn.click(
                            self.log_user_message,
                            [text_input, file_uploads_log],
                            [stored_messages, text_input, launch_research_btn],
                        ).then(
                            self.interact_with_agent,
                            [stored_messages, chatbot, session_state],
                            [chatbot],
                        ).then(
                            lambda: (
                                gr.Textbox(
                                    interactive=True,
                                    placeholder="Enter your prompt here and press the button",
                                ),
                                gr.Button(interactive=True),
                            ),
                            None,
                            [text_input, launch_research_btn],
                        )

                # Render simple layout
                else:
                    with gr.Blocks(
                        fill_height=True,
                    ):
                        gr.Markdown("""# 🔬 MolAgent - Mobile

A powerful AI research assistant for molecular and pharmaceutical research with automatic image display.

**Features:**
- 🧬 ChEMBL database integration
- 📊 Data visualization & plots
- 🔍 Web search capabilities
- 🖼️ Automatic image display

This demo is built on top of [huggingface/smolagents](https://github.com/huggingface/smolagents)'s open-Deep-Research implementation! ✨

Powered by advanced LLM and web search capabilities 👇""")
                        # Add session state to store session-specific data
                        session_state = gr.State({})  # Initialize empty state for each session
                        stored_messages = gr.State([])
                        file_uploads_log = gr.State([])
                        chatbot = gr.Chatbot(
                            label="MolAgent",
                            type="messages",
                            avatar_images=(
                                None,
                                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                            ),
                            resizeable=True,
                            scale=1,
                            show_copy_button=True,  # Allow copying of messages
                        )
                        # If an upload folder is provided, enable the upload feature
                        if self.file_upload_folder is not None:
                            upload_file = gr.File(
                                label="Upload a file (PDF, DOCX, TXT, CSV, Images)",
                                file_types=[".pdf", ".docx", ".txt", ".csv", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]
                            )
                            upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                            upload_file.change(
                                self.upload_file,
                                [upload_file, file_uploads_log],
                                [upload_status, file_uploads_log],
                            )

                        text_input = gr.Textbox(
                            lines=1,
                            label="Your request",
                            placeholder="Enter your prompt here and press the button",
                        )
                        launch_research_btn = gr.Button(
                            "Run",
                            variant="primary",
                            elem_classes=["primary-button"],
                        )

                        text_input.submit(
                            self.log_user_message,
                            [text_input, file_uploads_log],
                            [stored_messages, text_input, launch_research_btn],
                        ).then(
                            self.interact_with_agent,
                            [stored_messages, chatbot, session_state],
                            [chatbot],
                        ).then(
                            lambda: (
                                gr.Textbox(
                                    interactive=True,
                                    placeholder="Enter your prompt here and press the button",
                                ),
                                gr.Button(interactive=True),
                            ),
                            None,
                            [text_input, launch_research_btn],
                        )
                        launch_research_btn.click(
                            self.log_user_message,
                            [text_input, file_uploads_log],
                            [stored_messages, text_input, launch_research_btn],
                        ).then(
                            self.interact_with_agent,
                            [stored_messages, chatbot, session_state],
                            [chatbot],
                        ).then(
                            lambda: (
                                gr.Textbox(
                                    interactive=True,
                                    placeholder="Enter your prompt here and press the button",
                                ),
                                gr.Button(interactive=True),
                            ),
                            None,
                            [text_input, launch_research_btn],
                        )

        demo.launch(debug=True, share=True, **kwargs)


GradioUI(file_upload_folder=LOCAL_FILES).launch()
