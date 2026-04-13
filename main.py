import os
import json
from openai import OpenAI
from config import MODEL_NAME
from tools import tools_config, available_functions

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
)

def start_session_with_memory():
    print("--- 数据分析 Agent (基于Function call) ---")
    
    
    messages = [
        {"role": "system", "content": """你是一个数据分析专家。
    注意：在调用绘图工具时，标题（title参数）必须使用英文（例如 'Survival Distribution'），严禁使用中文，以防止系统字体兼容性报错。
    如果执行了绘图，请在回复中明确告知用户：'图片已保存至 XXX_distribution.png'"""}
    ]
    
    while True:
        u_input = input("\nUser >> ")
        if u_input.lower() in ['exit', 'quit']: break
        
        # 将新输入追加到现有的记忆中
        messages.append({"role": "user", "content": u_input})

        # 内层 Agent Loop：处理单次指令下的多步拆解或工具结果反馈
        while True:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools_config,
                tool_choice="auto" # 确保模型可以自主选择是否调用工具
            )
            
            resp_msg = response.choices[0].message
            
            # 状态 1：模型决定回复最终答案，跳出内层循环
            if not resp_msg.tool_calls:
                print(f"\nAI >> {resp_msg.content}")
                # 记得把 AI 的最终回答也存入记忆，方便下一次对话引用
                messages.append({"role": "assistant", "content": resp_msg.content})
                break

            # 状态 2：路由并执行工具
            # 必须先把模型的 tool_calls 意图存入 messages，否则协议会报错
            messages.append(resp_msg)
            
            for tool_call in resp_msg.tool_calls:
                func_name = tool_call.function.name
                # 解析模型生成的参数
                try:
                    args = json.loads(tool_call.function.arguments)
                except:
                    args = {}
                
                print(f"--- [工具路由执行] {func_name} | 参数: {args} ---")
                
                # 调用 tools.py 中对应的函数
                func = available_functions[func_name]
                result = func(**args) if args else func()

                # 将工具执行的“客观事实”反馈给记忆
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": str(result)
                })
            
            # 继续下一次内层循环，让模型根据工具返回的结果决定下一步动作
        
        # 外层循环等待下一个 User 输入，此时 messages 列表已保留了之前的全部记录

if __name__ == "__main__":
    start_session_with_memory()