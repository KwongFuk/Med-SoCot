import os
import json
import tqdm
import torch
import random
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import time

model_name = 'mistralai/Mistral-7B-v0.1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 加载模型时输出详细信息
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

# 加载分词器时输出详细信息
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

print("Model and tokenizer loaded successfully.")

prompt="""Task: You are a helpful assistant.Please provide your understanding of the problem.You don't need to answer the question, just analyze it.

Understand the Question:
- Explain the background or definition of the medical issue. Provide a brief description of basic concepts and possibly affected systems or organs.
- Identify and define key medical terms and concepts.
- Clarify the specific information or details requested.

Please refer to the following questions and Understand the Question.
Question: What is the relationship between Noonan syndrome and polycystic renal disease?
Understand the Question:Noonan syndrome is a genetic disorder characterized by distinct facial features, short stature, heart defects, and developmental delays. It affects multiple systems in the body, including the cardiovascular, musculoskeletal, and endocrine systems. Polycystic renal disease, particularly autosomal dominant polycystic kidney disease (ADPKD), is a genetic condition leading to the formation of numerous cysts in the kidneys, resulting in kidney enlargement and impaired function. The question seeks to explore the potential relationship between these two conditions, particularly any shared genetic or pathological mechanisms.

question:
"""
question = "What are the treatments and precautions for VDRL positive (syphilis) patients?"
query= prompt + question + "Understand the Question(less than 100 words):"

print(query)
print("_______________________________")
# 记录开始时间
start_time = time.time()# 编码输入
input_ids = tokenizer.encode(query, return_tensors="pt").to(device)

# 创建 attention mask
# 确保获取到的 pad_token_id 是有效的整数
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
attention_mask = (input_ids != pad_token_id).long()

# 生成输出
output = model.generate(
                input_ids,
                attention_mask=attention_mask,  # 设置 attention mask
                max_length=1024,
                no_repeat_ngram_size=2,
                do_sample=False,
                top_p=1.0,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id  # 设置 pad_token_id
            ).to(device)

# 解码输出
response = tokenizer.decode(output[0], skip_special_tokens=True)

pred = response[len(query):].strip()
# 记录结束时间
end_time = time.time()

# 计算推理总耗时
inference_time = end_time - start_time


# 输出总耗时
print(f"推理总耗时: {inference_time:.4f} 秒")
print(pred)