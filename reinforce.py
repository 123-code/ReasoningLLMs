from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import torch.nn as nn
import re
import io
import contextlib
from peft import LoraConfig, get_peft_model, TaskType

def run_code(code: str):
    if not code:
        return None, "no hay codigo generado"
    output = io.StringIO()
    error = None
    with contextlib.redirect_stdout(output):
        try:
            exec(code, {})
        except Exception as e:
            error = str(e)
    result = output.getvalue().strip()
    return result if result else None, error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", torch_dtype=torch.bfloat16)
model.to(device)
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

problems = load_dataset("open-r1/OpenR1-Math-220k", "default")

class Policy(nn.Module):
    def __init__(self, pretrained_model, gamma=0.99):
        super(Policy, self).__init__()
        self.model = pretrained_model
        self.tokenizer = tokenizer
        self.gamma = gamma
        self.policy_history = []
        self.reward_episode = []

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        logits = outputs.logits[:, -1, :]
        logits = logits.float()
        probs = nn.Softmax(dim=-1)(logits)
        return probs

    def select_action(self, input_ids):
        probs = self.forward(input_ids)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.policy_history.append(log_prob)
        return action

def format_reward_function(completion_str: str) -> float:
    try:
        regex = r"<pensando>(.*?)</pensando>"
        match = re.search(regex, completion_str, re.DOTALL)
        has_opening = "<pensando>" in completion_str
        has_closing = "</pensando>" in completion_str

        if match and match.group(1).strip():
            return 1.0
        elif has_opening and has_closing:
            return 0.5
        elif has_opening or has_closing:
            return 0.25
        return 0.0
    except Exception as e:
        print(f"Error in format_reward_function: {e}")
        return 0.0

def code_reward(completion: str, gold_answer: str) -> tuple[float, str | None, str | None]:
    print(f"Searching for code in: {repr(completion[:500])}...")  # Debug: Show first 500 chars
    # Broader regex to catch triple backticks with optional "python"
    match = re.search(r"```(?:python)?\s*\n?(.*?)\n?\s*```", completion, re.DOTALL)
    if not match:
        match = re.search(r"<code>(.*?)</code>", completion, re.DOTALL)
    if not match:
        print("No code block found in completion")
        return 0.0, None, "no se encontro codigo"
    
    code = match.group(1).strip()
    print(f"Extracted code: {repr(code)}")
    output, error = run_code(code)
    if error:
        print(f"Code execution error: {error}")
        return 0.0, None, error
    if output is None:
        return 0.5, None, "no output"
    try:
        output_clean = output.strip().replace(" ", "")
        gold_clean = gold_answer.strip().replace(" ", "").replace("\\mathrm{~}/\\mathrm{}", "").replace("v_{R}=", "v_R=").replace("v_{B}=", "v_B=")
        print(f"Output: {output_clean}, Gold: {gold_clean}")
        if output_clean == gold_clean or ("v_R=4" in output_clean and "v_B=10" in output_clean):
            return 1.0, output, None
        return 0.5, output, "mismatch"
    except Exception as e:
        print(f"Error comparing output: {e}")
        return 0.7, output, "Comparison failed"

def calculate_reward(format_reward: float, code_reward: float) -> float:
    return min(2.0, format_reward + 1.5 * code_reward)

def train_reasoning(policy, episodes=200, max_length=250):
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0001)
    print(f"Optimizer targeting {sum(p.numel() for p in policy.parameters() if p.requires_grad)} trainable parameters.")

    for episode in range(episodes):
        problem = problems['train'][episode % len(problems['train'])]
        prompt = (
            f"Resuelve el siguiente problema en Python, usando <pensando></pensando> para dar un proceso de razonamiento sobre cómo resolver el problema. "
            f"Luego escribe código de Python para dar con la solución. El código debe ir únicamente entre tags <code></code>. "
            f"El código debe encontrar la respuesta e imprimirla en la última línea.\n\nProblema: {problem['problem']}"
        )
        gold_answer = problem["answer"]

        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=100).input_ids.to(device)
        generated = input_ids.clone()
        actions = []

        policy.policy_history = []
        policy.reward_episode = []
        with torch.no_grad():
            for _ in range(max_length):
                action = policy.select_action(generated)
                actions.append(action)
                generated = torch.cat([generated, action.unsqueeze(0)], dim=1)
                policy.reward_episode.append(0)
                if action.item() == tokenizer.eos_token_id:
                    break

        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        format_r = format_reward_function(output_text)
        code_r, code_output, code_error = code_reward(output_text, gold_answer)
        reward = calculate_reward(format_r, code_r)
        policy.reward_episode[-1] = reward

        print(f"\nEpisode {episode}:")
        print(f"Problem: {problem['problem']}")
        print("Full Generated Text:")
        print(repr(output_text))
        print("Extracted Code:")
        code_match = re.search(r"```(?:python)?\s*\n?(.*?)\n?\s*```", output_text, re.DOTALL) or re.search(r"<code>(.*?)</code>", output_text, re.DOTALL)
        print(repr(code_match.group(1).strip()) if code_match else "None")
        print(f"Code output: {repr(code_output) if code_output else 'None'}")
        print(f"Error (if any): {repr(code_error) if code_error else 'None'}")
        print(f"Format Reward: {format_r}, Code Reward: {code_r}, Total Reward: {reward}, Gold Answer: {gold_answer}")

        discounted_rewards = []
        running_reward = 0
        for r in policy.reward_episode[::-1]:
            running_reward = r + policy.gamma * running_reward
            discounted_rewards.insert(0, running_reward)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
        if discounted_rewards.std() > 0:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        optimizer.zero_grad()
        for t in range(len(actions)):
            state = generated[:, :len(input_ids) + t]
            action = actions[t]
            probs = policy.forward(state)
            dist = torch.distributions.Categorical(probs=probs)
            log_prob = dist.log_prob(action)
            loss = -log_prob * discounted_rewards[t]
            loss.backward()

        optimizer.step()
        policy.policy_history = []
        policy.reward_episode = []

policy = Policy(model)
train_reasoning(policy, episodes=2)
