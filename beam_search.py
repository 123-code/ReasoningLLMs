from groq import Groq
import re

def generate_response(prompt: str, code_mode: bool, max_iterations: int = 5):
    client = Groq(api_key="AGREGAR API KEY")
    pattern = r"```(?:python|Python)?\s*([\s\S]*?)```"
    total_executions = 0
    success = 0
    feedback = ""

    for x in range(max_iterations):
        current_prompt = prompt + (f"\nPrevious attempt: {feedback}" if x > 0 else "")
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": current_prompt}],
            model="llama3-70b-8192",
        )
        result = chat_completion.choices[0].message.content
        step_info = {"iteration": x + 1, "prompt": current_prompt, "response": result}

        if code_mode:
            code_blocks = re.findall(pattern, result, flags=re.MULTILINE)
            if code_blocks:
                code = code_blocks[0]
                total_executions += 1
                try:
                    exec(code, globals())
                    success += 1
                    feedback = "successful execution"
                    step_info["code"] = code
                    step_info["execution"] = "success"
                    step_info["output"] = "Code executed successfully"  # Modify to capture actual output if needed
                except Exception as e:
                    feedback = f"an error occurred: {e}"
                    step_info["code"] = code
                    step_info["execution"] = f"error: {e}"
                    step_info["output"] = ""
            else:
                step_info["code"] = "no code generated"
                step_info["execution"] = "none"
                step_info["output"] = ""
        yield step_info
        yield {"generated_code": step_info.get("code", "No code generated")} 
    success_rate = (success / total_executions * 100) if total_executions > 0 else 0
    yield {"success_rate": success_rate}
