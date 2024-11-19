from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import BaseGenerationOutputParser
from langchain_core.outputs import Generation
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from pathlib import Path
from pprint import pprint
from typing import List
from IPython.display import Markdown,Latex, display,Image
from sentence_transformers import SentenceTransformer
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

import os
import base64
import re
import json
import faiss
import torch
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# 加载 .env 文件
load_dotenv()

# 获取环境变量
langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2")
langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_project = os.getenv("LANGCHAIN_PROJECT")

api_key=os.getenv("API_KEY")
base_url=os.getenv("BASE_URL")
# deepseek_api_key=os.getenv("DEEPSEEK_API_KEY")
# deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL")


def inputPrompt (question):
       
    # 构建系统消息
    system_message = SystemMessage(
        content="You are a financial expert. You will be given questions and options, possibly with context information and images. Please answer the question."
    )

    # 构建用户消息
    human_message=HumanMessage(content=[])

    if len(question["Share Context"]) != 0:
        human_message.content.append({"type": "text", "text": "Context: " + question["Share Context"]})

    if len(question["Share Image"])!= 0:
        for path in question["Share Image"]:
            image_url = "/home/sden118/evaluate_LLM/"+path
            with open(image_url, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            human_message.content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})

    human_message.content.append({"type": "text", "text": "Question: "+ question["Question Text"]})

    if len(question["Image"]) != 0:
        image_url = "/home/sden118/evaluate_LLM/"+question["Image"]
        with open(image_url, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        human_message.content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})

    human_message.content.append({"type": "text", "text": "Options: " + str(question["Options"])})

    human_message.content.append({"type": "text", "text": "Let's think step by step. The output reasoning steps are in Markdown format. Finally, must put the correct option (A, B, C, or D) in【 】. e.g.Therefore, the correct option is 【B】."})

    response = [system_message, human_message]
    return response



##Utils
def FeedbackPrompt (question):
       
    system_message = SystemMessage(
            content="""You are a financial expert. You will be given questions and options, possibly with context information and images. Also, you will be given wrong reasoning steps and correct reasoning hints.You are supposed to give feedback.""")

    # 构建用户消息
    human_message=HumanMessage(content=[])

    if len(question["Share Context"]) != 0:
        human_message.content.append({"type": "text", "text": "Context: " + question["Share Context"]})

    if len(question["Share Image"])!= 0:
        for path in question["Share Image"]:
            image_url = "/home/sden118/evaluate_LLM/"+path
            with open(image_url, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            human_message.content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})

    human_message.content.append({"type": "text", "text": "Question: "+ question["Question Text"]})

    if len(question["Image"]) != 0:
        image_url = "/home/sden118/evaluate_LLM/"+question["Image"]
        with open(image_url, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        human_message.content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})

    human_message.content.append({"type": "text", "text": "Options: " + str(question["Options"])})
    human_message.content.append({"type": "text", "text": "Wrong Reasoning Steps: " + question["Model Reasoning"]})
    human_message.content.append({"type": "text", "text": "Wrong Answer: " + question["Model Answer"]})
    human_message.content.append({"type": "text", "text": "Correct Reasoning Steps: " + question["Explanation"]})
    human_message.content.append({"type": "text", "text": "Correct Answer: " + question["Answer"]})

    human_message.content.append({"type": "text", "text": """ Please give the feedback in Markdown format. 1. Please output correct reasoning steps according to hints. 2. compare the correct reasoning step with the model's wrong reasoning step, and point out the difference. 3. summarize the hint for future simalar questions."""})

    response = [system_message, human_message]
    return response


def ICLPrompt (question,example):
       
    # 构建系统消息
    system_message = SystemMessage(
        content="You are a financial expert. You will be given previous learning document including questions and options, possibly with context information and images. Please answer the current question."
    )

    # 构建用户消息
    human_message=HumanMessage(content=[])
    human_message.content.append({"type": "text", "text": "Previous Learning Document: "})
    if len(example["Share Context"]) != 0:
        human_message.content.append({"type": "text", "text": "Context: " + example["Share Context"]})

    if len(example["Share Image"])!= 0:
        for path in example["Share Image"]:
            image_url = "/home/sden118/evaluate_LLM/"+path
            with open(image_url, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            human_message.content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})

    human_message.content.append({"type": "text", "text": "Question: "+ example["Question Text"]})

    if len(example["Image"]) != 0:
        image_url = "/home/sden118/evaluate_LLM/"+example["Image"]
        with open(image_url, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        human_message.content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})

    human_message.content.append({"type": "text", "text": "Options: " + str(example["Options"])})
    human_message.content.append({"type": "text", "text": "Wrong Reasoning Steps: " + example["Model Reasoning"]})
    human_message.content.append({"type": "text", "text": "Feedback: " + example["Feedback"]})


    human_message.content.append({"type": "text", "text": "Current Question is as follows: "})
    if len(question["Share Context"]) != 0:
        human_message.content.append({"type": "text", "text": "Context: " + question["Share Context"]})

    if len(question["Share Image"])!= 0:
        for path in question["Share Image"]:
            image_url = "/home/sden118/evaluate_LLM/"+path
            with open(image_url, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            human_message.content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})

    human_message.content.append({"type": "text", "text": "Question: "+ question["Question Text"]})

    if len(question["Image"]) != 0:
        image_url = "/home/sden118/evaluate_LLM/"+question["Image"]
        with open(image_url, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        human_message.content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})

    human_message.content.append({"type": "text", "text": "Options: " + str(question["Options"])})

    human_message.content.append({"type": "text", "text": "Let's think step by step. The output reasoning steps are in Markdown format. Finally, must put the correct option (A, B, C, or D) in【 】. e.g.Therefore, the correct option is 【B】."})

    response = [system_message, human_message]
    return response


class MarkdownParser(BaseGenerationOutputParser[str]):
    """
    A custom parser that formats the model output for Markdown display
    by replacing LaTeX-style delimiters \[ and \] with $.
    """
    def parse_result(self, result: List[Generation], *, partial: bool = False) -> str:
        """Parse the model output and format it as Markdown.

        Args:
            result: A list of Generations (assumed to contain only one string).
            partial: Whether to allow partial results (for streaming, not used here).

        Returns:
            A Markdown-formatted string with LaTeX-style delimiters replaced.
        """
        # Ensure there's only one generation
        if len(result) != 1:
            raise ValueError("This parser only supports a single generation.")
        
        # Extract the generation content
        generation = result[0]
        if not isinstance(generation.text, str):
            raise ValueError("Expected text output for Markdown formatting.")
        
        # Replace  \\[ and \\] with $ for LaTeX-style display
        formatted_text = generation.text.replace('\\[', '$').replace('\\]', '$').replace('\\(', '$').replace('\\)', '$')
        return formatted_text
    



def extract_answer(text: str) -> str:
    """Extract the answer option (A, B, C, or D) in brackets from the given text."""
    # Regular expression to find the answer in brackets, e.g., [C]
    match = re.search(r"\【([A-D])\】", text)
    if match:
        return match.group(1)  # Returns the answer option (e.g., "C")
    else:
        return "Answer not found"  # Returns a message if no answer is found

# Wrap extract_answer in a LangChain Tool to make it invokable
extract_answer_tool = Tool.from_function(
    func=extract_answer,
    name="Extract Answer Tool",
    description="Extracts the answer option in brackets (e.g., 【C】) from the provided text."
)



def write_output(data, file_path):
    # 如果文件存在，先读取现有数据
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    # 合并新数据到现有数据中
    if isinstance(existing_data, list):
        existing_data.append(data)
    else:
        existing_data = data

    # 将合并后的数据写入到 JSON 文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)


# Initialize model and processor
model = CLIPModel.from_pretrained("/home/sden118/evaluate_LLM/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("/home/sden118/evaluate_LLM/clip-vit-base-patch32")

#初始化数据库
def init_faiss():
    index = faiss.IndexFlatIP(1024)  # 使用内积度量
    if index.ntotal==0:
        # 初始化 Faiss 索引，使用内积 (dot product) 作为距离度量
        pass
    # 清空Faiss索引中的所有向量
    else:
        index.reset()
    # 检查索引是否清空
    print("Number of vectors after reset:", index.ntotal)
    return index

def index_faiss(index, file_path):
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 文件存在，加载数据
        data = json.loads(Path(file_path).read_text())
    else:
        # 文件不存在，创建一个新的空列表
        data =     [{"ID": 9999,"Question Number": 9999,"Share Context": "","Share Image": "","Question Text": "text","Image": "images/QuantitativeAnalysis1_images/40u.png",
                     "Options": {"A": " -0.215","B": " -0.113","C": " 0.113","D": " 0.215"},"Answer": "C","Explanation": "text","QA Type": "text","Question Type": "text","Level of Difficulty": "text",
                    "Knowledge Topics": "text","General Topics":"text","Book Label": "text","Model Answer": "C","Model Reasoning": "text","Feedback": "text"
                    }]
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    # 处理数据
    for error in data:
        storeEmbedding(index, error)

    print("Number of vectors after adding:", index.ntotal)

    return index

def clipEmbedding(data):
    textdata = "Question:" + data.get("Question Text") + " Options:" + str(data.get("Options")) + " Correct Answer:" + data.get("Answer")
    
    # 检查是否有图片
    if data.get("Image") != '':
        image_path = "/home/sden118/evaluate_LLM/"+ data.get("Image") 
        # print(image_path)
        image = Image.open(image_path)
        # print(image)
        
        # 生成文本和图像的嵌入，添加 truncation=True 和 max_length=77
        inputs = processor(text=[textdata], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
        
        # 使用CLIP模型生成嵌入
        outputs = model(**inputs)
        image_embedding = outputs.image_embeds  # 图像嵌入
        text_embedding = outputs.text_embeds  # 文本嵌入
    else:
        # 如果没有图像，生成文本嵌入
        inputs = processor(text=[textdata], return_tensors="pt", padding=True, truncation=True, max_length=77)
        text_embedding = model.get_text_features(**inputs)
        
        # 创建一个与图像嵌入维度相同的零向量
        image_embedding = torch.zeros((text_embedding.shape[0], 512))  # 假设图像嵌入维度是512
    
    # 将文本和图像嵌入拼接在一起
    combined_embedding = torch.cat((text_embedding, image_embedding), dim=-1)
    
    return combined_embedding


def normalize(embeddings):
    # 归一化函数，计算余弦相似度时将向量进行归一化
    norms = torch.norm(embeddings, dim=1, keepdim=True)  # 计算每个向量的范数
    return embeddings / norms  # 将向量归一化，使其范数变为1


# 生成嵌入
def storeEmbedding(index,data):
    error_log_embedding = clipEmbedding(data)
    # 对嵌入进行归一化，以便计算余弦相似度
    error_log_embedding = normalize(error_log_embedding)
    # 将生成的多模态嵌入转换为numpy数组并添加到Faiss索引中
    error_log_embedding_np = error_log_embedding.detach().numpy()  # 确保转换为numpy格式
    index.add(error_log_embedding_np)  # 将嵌入添加到Faiss索引中
    return index

# 查询函数
def query_embedding_faiss(query_data, index, k=5):
    # 生成查询嵌入
    query_embedding = clipEmbedding(query_data)
    query_embedding = normalize(query_embedding)  # 归一化查询向量
    
    # 转换为 numpy 格式
    query_embedding_np = query_embedding.detach().numpy()
    
    # 检索 Faiss 中与查询向量最相似的 k 个向量
    D, I = index.search(query_embedding_np, k)  # D 是余弦相似度，I 是对应的索引
    
    return D, I

# 初始化模型
LLMmodel = ChatOpenAI(model="gpt-4o", api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"), temperature = 0.7)
outputParser = MarkdownParser()
chain = LLMmodel|outputParser

#配置文件路径
ErrorLogPath="/home/sden118/evaluate_LLM/errorLog/gpt/gpt_4o_ErrorLog.json"
ModelOutputPath="/home/sden118/evaluate_LLM/output_test/gpt/gpt_4o_test_output.json"
ModelRAGOutputPath="/home/sden118/evaluate_LLM/output_test/gpt/gpt_4o_test_RAGoutput.json"

# 加载数据

file_path = '/home/sden118/evaluate_LLM/data_v5.json'
data = json.loads(Path(file_path).read_text())

filtered_data = [item for item in data if item.get("Filter") == ""]
data=filtered_data

index=init_faiss()# 初始化数据库
index_faiss(index,ErrorLogPath) #把现在的errorlog加入到向量数据库

for question in data[2:]:
    reasoning=chain.invoke(inputPrompt(question))
    answer=extract_answer(reasoning)    

    modelOutput=question
    modelOutput["Model Answer"]=answer
    modelOutput["Model Reasoning"]=reasoning
    write_output(modelOutput, ModelOutputPath)

    cos,I = query_embedding_faiss(question, index, k=5)
    erroRLog=json.loads(Path(ErrorLogPath).read_text())
    errorexample=erroRLog[I[0][0]]
    reasoning=chain.invoke(ICLPrompt(question,errorexample))
    answer=extract_answer(reasoning)
    display(Markdown(reasoning))
    print("=====================================")
    modelOutput["Model Answer"]=answer
    modelOutput["Model Reasoning"]=reasoning  
    write_output(modelOutput, ModelRAGOutputPath) 

      



