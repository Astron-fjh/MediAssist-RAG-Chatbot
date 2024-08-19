# MediAssist-RAG-Chatbot
## 1. 项目概述

本项目旨在开发一个**基于RAG（Retrieval-Augmented Generation）技术的智能对话机器人，能够从药单图片中提取信息，并结合大模型分析药品信息，推测病情，进一步为用户提供健康咨询**。项目亮点包括**OCR图像文本提取、结合NVIDIA LLaMA 3大模型的药品分析与病情推测、以及基于提取信息的智能对话功能**。此系统特别适用于医疗场景中的辅助诊断和用户健康咨询。

## 2. 技术方案与实施步骤

### 2.1 模型选择

- **大模型选择**：项目采用了NVIDIA `LLaMA 3-70B-Instruct`大模型，因其在处理复杂语言任务（如医学文本解析、对话生成等）上表现优越。RAG模型的优势在于通过结合检索和生成技术，能够在对话过程中实时引入相关背景信息，从而提高对话的专业性和准确性。
- **RAG模型的优势**：RAG模型结合了检索和生成的能力，能够在回答问题时调用预训练知识库中的信息，从而提升回答的准确性与深度。这种模型非常适合需要在大规模数据基础上进行准确回答的场景，如药品信息的查询与病情推测。



### 2.2 数据的构建

- **数据构建过程**：数据主要来源于药单图像，通过OCR技术提取文本信息。使用Tesseract OCR工具进行图像文字识别，并结合中文语言包优化识别效果。将提取的文本数据作为输入，传入大模型进行分析和生成对话。

### 2.3 实施步骤

#### 2.3.1 环境搭建

1. **开发环境**：使用Python作为主要开发语言，依赖库包括`pytesseract`用于OCR，`gradio`用于创建前端界面，`langchain`用于模型集成。

2. **安装Python库**：首先配置NVIDIA的API密钥，接着安装Tesseract OCR和相关Python库。使用Tesseract支持中文的语言包以实现对药单中中文信息的提取。

```python
# 安装Tesseract OCR库
pip install pytesseract

# 安装Gradio库，用于构建前端界面
pip install gradio

# 安装LangChain库，用于与NVIDIA API的集成
pip install langchain

# 安装langchain_core
pip install langchain_core

# 安装Pillow库，用于处理图像
pip install pillow

# 安装NVIDIA API的端点库
pip install langchain_nvidia_ai_endpoints
```

3. **安装 Tesseract OCR**
   1. 下载[**Tesseract安装包**](https://github.com/UB-Mannheim/tesseract/wiki)并安装。
   2. 从[**GitHub Tesseract 项目**](https://github.com/tesseract-ocr/tessdata)中下载中文语言包`chi_sim.traineddata`（简体中文）或`chi_tra.traineddata`（繁体中文）。将该文件放置在Tesseract的`tessdata`目录中。默认情况下，这个目录通常在`Tesseract-OCR`安装目录下的`tessdata`文件夹中，例如`D:\Tesseract-OCR\tessdata`。
   3. 确保`pytesseract.pytesseract.tesseract_cmd`指向Tesseract的可执行文件路径。如：

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

4. **申请NIM的API Key，来调用NIM的计算资源**

   进入https://build.nvidia.com/microsoft/phi-3-vision-128k-instruct，点击**Get API Key**按钮，生成一个秘钥。

   ![image-20240818175736627](https://cdn.jsdelivr.net/gh/xiaodiao188/blog-img@img/img/202408181757927.png)

#### 2.3.2 代码实现

1. **导入工具包**

   ```python
   import gradio as gr
   import pytesseract
   from PIL import Image
   from langchain_nvidia_ai_endpoints import ChatNVIDIA
   from langchain_core.prompts import ChatPromptTemplate
   from langchain.schema.runnable import RunnableLambda
   from langchain_core.runnables import RunnableBranch, RunnableAssign
   import os
   ```

   将上面准备好的秘钥粘贴在此处, 当我们向服务器发送计算请求时, 需要用到：

   ```Python
   os.environ["NVIDIA_API_KEY"] = "nvapi-..."
   ```

   设置**Tesseract**可执行文件的路径：

   ```Python
   pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'  # 根据自己的安装路径修改
   ```

2. **从图像中提取药品信息**

   - 使用 `pytesseract` 从图像中提取文本，支持中文。

   ```python
   def extract_medicine_from_image(image) -> str:
       image = Image.fromarray(image)
       custom_config = r'--oem 3 --psm 6'
       text = pytesseract.image_to_string(image, config=custom_config, lang="chi_sim")
       return text
   ```

3. **解析药品信息并推测病情**

   - 使用 `ChatNVIDIA` 模型解析药品信息，并推测可能的疾病。

   - 定义一个Prompt模板来指导模型分析药品信息。

   ```python
   def analyze_medicine_info(text: str) -> str:
       try:
           # 使用 ChatNVIDIA 模型解析药品信息
           llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
           
           # 设置提示模板，让模型以中文回答，并按指定格式输出
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
   ```

4. **处理用户对话**

   - 使用 `ChatNVIDIA` 模型基于提取的药品信息和用户的输入进行对话。

   - 更新对话记录并返回聊天记录。

   ```python
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
   ```

5. **处理上传的图片**

   ```python
   def process_image(image):
       try:
           extracted_text = extract_medicine_from_image(image)
           analysis_result = analyze_medicine_info(extracted_text)
           return extracted_text, analysis_result
       except Exception as e:
           return f"Error: {str(e)}", ""
   ```

6. **构建Gradio界面**

   - 配置Gradio界面，允许用户上传药单图片，显示提取的药品信息和分析结果，支持与药方相关的对话。

   - 定义输入输出组件，并设置按钮的点击事件。

   ```python
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
   ```

## 3. 项目成果与展示

### 3.1 应用场景展示

- 该智能对话机器人适用于医疗场景，例如：
  - **客户服务**：自动解析药单，提供药品信息和使用建议。
  - **健康咨询**：根据用户输入问题和药单信息，推测病情并提供健康建议。

### 3.2 功能演示

以下是主要功能的展示：

1. **药单上传与信息提取**： 用户上传药单图像，系统提取药品信息并显示在界面上。
2. **药品信息分析与病情推测**： 系统分析药品信息并推测可能的病情，输出分析结果。
3. **智能对话**： 用户输入健康问题，系统结合药方信息生成智能回复。

- **UI 展示**：

![{16022AF8-AB9D-4a60-9192-D1D672B327B2}](https://cdn.jsdelivr.net/gh/xiaodiao188/blog-img@img/img/202408181842847.png)
