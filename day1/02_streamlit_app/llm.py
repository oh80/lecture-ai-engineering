# llm.py
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import streamlit as st
import time
from config import MODEL_NAME
from huggingface_hub import login
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from textwrap import dedent


# モデルをキャッシュして再利用
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:

        # アクセストークンを保存
        hf_token = st.secrets["huggingface"]["token"]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}") # 使用デバイスを表示
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True)

        pipe = pipeline(
          "text-generation",
          model=model,
          tokenizer=tokenizer,
          return_full_text=False,
          max_new_tokens=256,)

        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None

def generate_response(pipe, user_question):
    """LLMを使用して質問に対する回答を生成する"""
    if pipe is None:
        return "モデルがロードされていないため、回答を生成できません。", 0

    try:
        start_time = time.time()

        template = dedent("""{user_question}""")
        question_prompt = PromptTemplate(template=template, input_variables=["user_question"])
        llm = HuggingFacePipeline(pipeline=pipe, verbose=True)
        chain = question_prompt | llm | StrOutputParser()

        assistant_response = chain.invoke(user_question)

        end_time = time.time()
        response_time = end_time - start_time
        return assistant_response, response_time

    except Exception as e:
        st.error(f"回答生成中にエラーが発生しました: {e}")
        # エラーの詳細をログに出力
        import traceback
        traceback.print_exc()
        return f"エラーが発生しました: {str(e)}", 0