import gradio as gr
import pytesseract
from PIL import Image
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
import os

# 配置 NVIDIA 的API密钥
os.environ["NVIDIA_API_KEY"] = "nvapi-..."  # 根据自己申请的密钥配置
pytesseract.pytesseract.tesseract_cmd = r'....\Tesseract-OCR\tesseract.exe'  # 根据自己的安装路径配置

# 1. 从图像中提取文本（包含数学公式）
def extract_medicine_from_image(image) -> str:
    image = Image.fromarray(image)
    custom_config = r'--oem 3 --psm 6'
    # 使用中文语言包提取文本
    text = pytesseract.image_to_string(image, config=custom_config, lang="chi_sim")
    return text

# 2. 解析药品信息并推测病情
def analyze_medicine_info(text: str) -> str:
    try:
        llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
        
        prompt_template = ChatPromptTemplate.from_template(
            "以下文本包含药品信息。请识别药品名称及其作用，并推测病人可能患有的疾病。请用中文回答，并按如下格式输出：\n\n"
            "1. 药品名称: [药品名]\n   作用: [药品作用]\n\n"
            "2. 药品名称: [药品名]\n   作用: [药品作用]\n\n"
            "可能的病人病情:\n1. [病情1]\n2. [病情2]\n\n文本: {text}"
        )
        
        chain = prompt_template | llm
        result = chain.invoke({"text": text})
        return result.content
    except Exception as e:
        return f"Error during model invocation: {str(e)}"

# 3. 对话功能：根据药方进行对话
def chat_with_user(chat_history, user_input, extracted_text):
    try:
        llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
        
        prompt = (
            f"以下是从药方中提取的信息：{extracted_text}\n\n"
            f"用户输入：{user_input}\n\n"
            f"基于药方信息，回复用户的问题或建议下一步的治疗方向。请用中文回答。"
        )
        
        result = llm.invoke(prompt)
        chat_history.append((user_input, result.content))
        return chat_history, ""
    except Exception as e:
        return chat_history, f"Error during model invocation: {str(e)}"

# Gradio函数，用于处理上传的图像
def process_image(image):
    try:
        extracted_text = extract_medicine_from_image(image)
        analysis_result = analyze_medicine_info(extracted_text)
        return extracted_text, analysis_result
    except Exception as e:
        return f"Error: {str(e)}", ""

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("### 药单分析与对话：提取药品信息及与药方相关的对话")

    image_input = gr.Image(type="numpy", label="上传药单图片")
    text_output = gr.Textbox(label="提取的药品信息")
    analysis_output = gr.Textbox(label="分析结果（药品作用及病情）")
    chatbot = gr.Chatbot(label="与药方相关的对话")
    user_input = gr.Textbox(label="输入您的问题")
    submit_button = gr.Button("分析药单")
    send_button = gr.Button("发送")

    extracted_text_state = gr.State("")

    submit_button.click(fn=process_image, inputs=image_input, outputs=[text_output, analysis_output])
    submit_button.click(lambda: [], inputs=None, outputs=chatbot)
    send_button.click(fn=chat_with_user, inputs=[chatbot, user_input, text_output], outputs=[chatbot, user_input])
    send_button.click(lambda: "", None, user_input)  # 清空用户输入框

demo.launch(debug=True, share=True, show_api=False, server_port=8000, server_name="0.0.0.0")
