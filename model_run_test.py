import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import logging


model_path = "baichuan-inc/Baichuan2-13B-Chat"
logging.basicConfig(level=logging.INFO)

tokenizer = AutoTokenizer.from_pretrained(model_path,
    revision="v2.0",
    use_fast=False,
    trust_remote_code=True)

# 모델 로딩 전후에 로깅 추가
logging.info("모델 로딩 시작: %s", model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,
    device_map="auto",
    revision="v2.0",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True)

logging.info("모델 로딩 완료")

# 생성 구성 로딩
logging.info("GenerationConfig 로딩")
model.generation_config = GenerationConfig.from_pretrained(model_path)
logging.info("GenerationConfig 로딩 완료")

for name, module in model.named_modules():
    print(name)


# # 메시지 및 채팅 테스트
# logging.info("채팅 시작")
# messages = [{"role": "user", "content": "您好"}]
# response = model.apply_chat_template(tokenizer, messages)
# logging.info("채팅 완료")

# # 결과 출력
# print(response)

