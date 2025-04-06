from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import torch.nn as nn
import re
import io
import contextlib
from peft import LoraConfig, get_peft_model, TaskType


def extract_code(text):
    pattern = r'```(?:python)?(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    code = matches[-1].strip()
    return code

def run_code(code):
    if not code:
        return None, "No code provided"
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
    
 #recompoensa basada en el formato de la respuesta    
    
def format_reward_function(completion_str:str) -> float:
    allowed_pattern = r"^[\d+\-*/().\s]+$"
    try:
        completion = "<pensando>" + completion
        if completion.endswith():
            completion = completion[:-len(EOS_TOKEN)]
        regex = r""
        match = re.search(regex,completion,re.DOTALL)

        if match is None or len(match.groups()) != 2:
            return 0.0 
        else:
            answer_content = match.group(2).strip()
            if not re.match(allowed_pattern, answer_content):
                return 0.5
            else:
                return 1.0 
    except Exception:
        
        return 0.0
    
#recompensa para codigo con formato y funcional
def code_reward(completion: str) -> float:
    match = re.search(r"<code>(.*?)<\/code>", completion, re.DOTALL)
    if match is None:
        return 0.0
    else:
        try:
            exec(match.group(1))
            return 1.0
        except Exception as e:
            print(e)
            return 0.0

def calculate_reward(generated_text, correct_answer):
    code = extract_code(generated_text)
    if not code:
        return 0.0
    output, error = run_code(code)
    if error:
        return 0.1
    if output is None:
        return 0.2
  
    expected = "10, 4"
    if output.strip() == expected:
        return 1.0
    return 0.5


def train_reasoning(policy, episodes=200, max_length=250):
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0001)
    print(f"Optimizer targeting {sum(p.numel() for p in policy.parameters() if p.requires_grad)} trainable parameters.")

    for episode in range(episodes):
        problem = problems['train'][episode % len(problems['train'])]
        prompt = (
            f"Solve the following problem by writing Python code enclosed in ```python ... ``` "
            f"that computes the answer and prints it as the final line.\n\nProblem: {problem['problem']}"
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

        
        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        code = extract_code(output_text)
        output, error = run_code(code) if code else (None, "No code extracted")
        reward = calculate_reward(output_text, gold_answer)

        
        print(f"\nEpisode {episode}:")
        print(f"Problem: {problem['problem']}")
        print("Full Generated Text:")
        print(repr(output_text))  
        print("Extracted Code:")
        print(repr(code) if code else "None") 
        print(f"Code Output: {repr(output) if output else 'None'}")
        print(f"Error (if any): {repr(error) if error else 'None'}")
        print(f"Reward: {reward}, Gold Answer: {gold_answer}")

        for i in range(len(policy.reward_episode)):
            policy.reward_episode[i] = reward
        discounted_rewards = []
        running_reward = 0
        for r in policy.reward_episode[::-1]:
            running_reward = r + policy.gamma * running_reward
            discounted_rewards.insert(0, running_reward)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
        if discounted_rewards.std() > 0:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        
        optimizer.zero_grad()
        policy.policy_history = []
        for t in range(max_length):
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
