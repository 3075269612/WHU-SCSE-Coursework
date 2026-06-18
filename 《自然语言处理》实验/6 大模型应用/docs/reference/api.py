import os
from openai import OpenAI

# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'
# openai_api_key = os.environ.get("OPENAI_API_KEY")
dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")

# 设置 API 密钥
client = OpenAI(
    api_key=dashscope_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    ) 

# 生成摘要
def generate_summary(prompt):
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "user", "content": f"请为以下新闻标题生成简短摘要：{prompt}"}
            ],
            temperature=0.5,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error occurred: {str(e)}"

# 生成代码
def generate_code(description):
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "user", "content": description}
            ],
            temperature=0.2,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == "__main__":
    news_title = "NASA's Perseverance rover successfully lands on Mars"
    print("新闻摘要生成:")
    summary = generate_summary(news_title)
    print(summary)

    print("\n------------------------------\n")

    code_description = "生成一个Python函数，该函数接收两个参数，返回它们的和。"
    print("代码生成:")
    code = generate_code(code_description)
    print(code)
