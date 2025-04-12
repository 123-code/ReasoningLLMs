from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import torch.nn as nn
import re
import chess
import chess.engine
import chess.svg
from peft import LoraConfig, get_peft_model, TaskType
from IPython.display import display, SVG

HF_DATASET_NAME = "bonna46/Chess-FEN-and-NL-Format-30K-Dataset"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
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

# Load and inspect dataset
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
    """Extract the move from <movimiento> tags or raw text, ensuring UCI format."""
    try:
       
        regex = r"<movimiento>(.*?)</movimiento>"
        match = re.search(regex, completion_str, re.DOTALL)
        if match and match.group(1).strip():
            move = match.group(1).strip()
            if re.match(r"^[a-h][1-8][a-h][1-8]$", move):
                return move
            print(f"Invalid UCI format in <movimiento>: {move}")
            return None

        # Fallback: search for UCI-like strings
        uci_pattern = r"\b[a-h][1-8][a-h][1-8]\b"
        uci_matches = re.findall(uci_pattern, completion_str)
        if uci_matches:
            return uci_matches[-1]
        print("No valid UCI move found in output")
        return None
    except Exception as e:
        print(f"Error in extract_move: {e}")
        return None

def get_movement_reward(board, move_uci, stockfish_move):
    reward = 0.0
    if move_uci:
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
            reward -= 0.5
    else:
        reward -= 1.0
    return reward

def calculate_reward(format_reward: float, movement_reward: float) -> float:
    return min(2.0, format_reward + 1.5 * movement_reward)

def display_board_svg(board, move_uci=None, episode=None):
    try:
 
        svg = chess.svg.board(board=board, size=400)
        print(f"Episode {episode} - Initial Board:")
        display(SVG(svg)) 

        with open(f"board_initial_ep{episode}.svg", "w") as f:
            f.write(svg)


        if move_uci and is_valid_move(board, move_uci):
            board_copy = board.copy()
            board_copy.push(chess.Move.from_uci(move_uci))
            svg = chess.svg.board(board=board_copy, size=400)
            print(f"Episode {episode} - Board after move {move_uci}:")
            display(SVG(svg))
            with open(f"board_after_move_ep{episode}.svg", "w") as f:
                f.write(svg)
    except Exception as e:
        print(f"Error generating SVG for episode {episode}: {e}")

def train_reasoning(policy, episodes=200, max_length=250):
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0001)
    print(f"Optimizer targeting {sum(p.numel() for p in policy.parameters() if p.requires_grad)} trainable parameters.")

    for episode in range(episodes):

        problem = dataset['train'][episode % len(dataset['train'])]['FEN']
        board = chess.Board(problem)
        _, stockfish_move = get_stockfish_eval(board)

      
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a chess expert tasked with analyzing chess positions and selecting the best move. "
                    "Given a FEN position, provide a clear reasoning process within <pensando></pensando> tags, "
                    "explaining why you choose the move. Then, output the move in UCI format (e.g., 'e2e4' for pawn from e2 to e4) "
                    "within <movimiento></movimiento> tags. The UCI move must be exactly 4 characters, specifying source and destination squares. "
                    "Do not output FEN strings, vague terms like 'pawn moves,' or anything other than a single UCI move in the tags."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Analyze the chess position given by the FEN: {problem}. "
                    f"Explain your reasoning in <pensando></pensando> tags. "
                    f"Then, provide the best move in UCI format within <movimiento></movimiento> tags."
                )
            }
        ]


        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


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
        display_board_svg(board, move_uci, episode)

        print(f"\nEpisode {episode}:")
        print(f"FEN: {problem}")
        print("Generated Text:")
        print(repr(output_text))
        print(f"Extracted Move: {move_uci}")
        print(f"Stockfish Move: {stockfish_move}")
        print(f"Format Reward: {format_r}, Movement Reward: {movement_r}, Total Reward: {reward}")


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
train_reasoning(policy, episodes=10) 
