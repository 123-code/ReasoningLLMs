from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import torch.nn as nn
import re
import chess
import chess.engine
from peft import LoraConfig, get_peft_model, TaskType

HF_DATASET_NAME = "bonna46/Chess-FEN-and-NL-Format-30K-Dataset"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16
)
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


dataset = load_dataset(HF_DATASET_NAME)
print("Dataset sample:", dataset['train'][0])

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

def is_valid_move(board, move_uci):
    try:
        move = chess.Move.from_uci(move_uci)
        return move in board.legal_moves
    except ValueError:
        return False

def get_stockfish_eval(board, depth=15):
    try:
        engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].pov(board.turn).score(mate_score=10000)
        best_move = info["pv"][0] if "pv" in info else None
        engine.quit()
        return score, best_move.uci() if best_move else None
    except Exception as e:
        print(f"Error during Stockfish evaluation: {e}")
        return None, None

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

def extract_move(completion_str: str) -> str:
    """Extract the move from <movimiento> tags, ensuring UCI format."""
    try:
        # Try <movimiento> tags
        regex = r"<movimiento>(.*?)</movimiento>"
        match = re.search(regex, completion_str, re.DOTALL)
        if match:
            move = match.group(1).strip()
            if move and re.match(r"^[a-h][1-8][a-h][1-8]$", move):
                return move
            print(f"Invalid or empty UCI format in <movimiento>: '{move}'")
            return None

        # No fallback to avoid picking up prompt examples
        print("No valid UCI move found in <movimiento> tags")
        return None
    except Exception as e:
        print(f"Error in extract_move: {e}")
        return None

def get_movement_reward(board, move_uci, stockfish_move):
    reward = 0.0
    if move_uci:
        if re.match(r"^[a-h][1-8][a-h][1-8]$", move_uci):
            valid_move = is_valid_move(board, move_uci)
            if valid_move:
                reward += 1.0
                board_copy = board.copy()
                board_copy.push(chess.Move.from_uci(move_uci))
                score, _ = get_stockfish_eval(board_copy)
                if score is not None:
                    reward += score / 100.0
                if move_uci == stockfish_move:
                    reward += 0.5
            else:
                reward -= 0.5  # Invalid move
        else:
            reward -= 0.7  # Malformed UCI
    else:
        reward -= 1.0  # No move
    return reward

def calculate_reward(format_reward: float, movement_reward: float) -> float:
    return min(2.0, format_reward + 1.5 * movement_reward)

def train_reasoning(policy, episodes=10, max_length=250):
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0001)
    print(f"Optimizer targeting {sum(p.numel() for p in policy.parameters() if p.requires_grad)} trainable parameters.")

    for episode in range(episodes):
        # Get problem from dataset
        problem = dataset['train'][episode % len(dataset['train'])]['FEN']
        board = chess.Board(problem)
        _, stockfish_move = get_stockfish_eval(board)

        # System and user prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a chess expert. For a given FEN, analyze the position and explain your move choice in <pensando></pensando> tags. "
                    "Then, output exactly one move in UCI format (4 characters, e.g., 'e2e4') in <movimiento></movimiento> tags. "
                    "Do not output FENs, vague terms, or anything else in the move tags."
                )
            },
            {
                "role": "user",
                "content": (
                    f"FEN: {problem}\n"
                    f"Analyze the position, explain your reasoning in <pensando></pensando> tags, "
                    f"and provide the best move in UCI format in <movimiento></movimiento> tags."
                )
            }
        ]

        # Format prompt
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        input_ids = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=200).input_ids.to(device)
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
        move_uci = extract_move(output_text)
        movement_r = get_movement_reward(board, move_uci, stockfish_move)
        reward = calculate_reward(format_r, movement_r)
        policy.reward_episode[-1] = reward

        print(f"\nEpisode {episode}:")
        print(f"FEN: {problem}")
        print("Generated Text:")
        print(repr(output_text))
        print(f"Extracted Move: {move_uci}")
        print(f"Stockfish Move: {stockfish_move}")
        print(f"Format Reward: {format_r}, Movement Reward: {movement_r}, Total Reward: {reward}")

        # Compute discounted rewards
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
train_reasoning(policy)
