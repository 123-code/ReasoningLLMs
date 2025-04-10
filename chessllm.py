
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import TextStreamer, TrainingArguments
import chess
import chess.engine
import chess.svg 
import random
import io
import math
import re
import os
import atexit
import google.generativeai as genai

try:
    from IPython.display import display, SVG
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    print("IPython display not available. Board SVG will not be shown.")



HF_DATASET_NAME = "bonna46/Chess-FEN-and-NL-Format-30K-Dataset"
SPLIT_NAME = "train"
MODEL_NAME = "unsloth/gemma-2-9b-it" # Smaller Gemma model suitable for T4/limited VRAM
NUM_EPOCHS = 1  # Train for 1 full pass through the dataset
MAX_SEQ_LENGTH = 1024 # Sequence length
LORA_R = 16            # LoRA rank (increased slightly)
LORA_ALPHA = 16        # LoRA alpha (keep equal to R is common)
LORA_DROPOUT = 0      # LoRA dropout
LEARNING_RATE = 5e-6  # Lower learning rate for RL/fine-tuning
MAX_PROMPT_LENGTH = 256 # Increased slightly for potentially longer FENs/history
MAX_COMPLETION_LENGTH = MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH
NUM_GENERATIONS = 6   # Number of completions per prompt (adjust based on memory)
BATCH_SIZE = 1        # Keep batch size 1 for GRPO prompt processing logic
GRAD_ACCUMULATION_STEPS = 8 # Effective batch size = 1 * 8 = 8 (adjust based on memory)
OUTPUT_DIR = "outputs_chess_grpo_gemma2_1epoch"
# Determine Stockfish path based on environment
IN_COLAB = "COLAB_GPU" in os.environ
if IN_COLAB:

    STOCKFISH_DIR_COLAB = "/content/stockfish"
    stockfish_files = [f for f in os.listdir(STOCKFISH_DIR_COLAB) if 'stockfish' in f and not f.endswith('.zip') and os.access(os.path.join(STOCKFISH_DIR_COLAB, f), os.X_OK)] if os.path.exists(STOCKFISH_DIR_COLAB) else []
    if stockfish_files:
        STOCKFISH_PATH = os.path.join(STOCKFISH_DIR_COLAB, stockfish_files[0])
        print(f"Using automatically prepared Stockfish: {STOCKFISH_PATH}")
    else:
        print("WARNING: Stockfish executable not found in /content/stockfish. Will skip eval reward.")
        STOCKFISH_PATH = None
else:

    STOCKFISH_PATH = "/path/to/your/local/stockfish" 
    print(f"Using local Stockfish path (verify it's correct!): {STOCKFISH_PATH}")

GEMINI_API_KEY = "" 


ANALYSIS_START = "<analysis>"
ANALYSIS_END = "</analysis>"
BEST_MOVE_START = "<best_move>"
BEST_MOVE_END = "</best_move>"

CHESS_SYSTEM_PROMPT = f"""You are a helpful chess assistant. Analyze the given chess position described by the FEN string.
Think step-by-step about the position: evaluate material balance, king safety, piece activity, pawn structure, threats, and potential plans for both White and Black. Enclose your thinking process within {ANALYSIS_START} and {ANALYSIS_END} tags.
After your analysis, state the single best move you recommend for the side to play. Use Standard Algebraic Notation (SAN) and enclose the move within {BEST_MOVE_START} and {BEST_MOVE_END} tags. Example: {BEST_MOVE_START}Nf3{BEST_MOVE_END}."""


print(f"Loading chess dataset from Hugging Face: {HF_DATASET_NAME}, Split: {SPLIT_NAME}")
try:
    dataset_raw = load_dataset(HF_DATASET_NAME, split=SPLIT_NAME)
    if 'FEN' not in dataset_raw.column_names:
        print(f"ERROR: Dataset loaded, but required column 'FEN' not found.")
        exit()
except Exception as e:
    print(f"ERROR: Could not load dataset '{HF_DATASET_NAME}' from Hugging Face: {e}")
    exit()
if dataset_raw is None or len(dataset_raw) == 0:
    print(f"ERROR: Failed to load dataset or the '{SPLIT_NAME}' split is empty.")
    exit()
print(f"Loaded raw dataset with {len(dataset_raw)} positions from Hugging Face.")

print(f"Loading model: {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
print("Model and Tokenizer loaded.")


print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
print("LoRA adapters added.")


engine = None
def cleanup_engine():
    global engine
    if engine:
        print("Quitting Stockfish engine...")
        try: engine.quit()
        except Exception as e: print(f"Error quitting engine: {e}")
        engine = None
try:
    if STOCKFISH_PATH and os.path.exists(STOCKFISH_PATH):
        if not os.access(STOCKFISH_PATH, os.X_OK) and os.name != 'nt':
             print(f"Setting execute permission for Stockfish at {STOCKFISH_PATH}")
             os.chmod(STOCKFISH_PATH, 0o755)
        print(f"Initializing Stockfish engine from: {STOCKFISH_PATH}")
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        atexit.register(cleanup_engine)
        print("Stockfish engine initialized.")
    else:
        print(f"WARNING: Stockfish path '{STOCKFISH_PATH}' not found/provided. Eval reward will be 0.")
        engine = None
except Exception as e:
    print(f"ERROR: Could not initialize Stockfish engine: {e}")
    engine = None

def format_chess_prompt(entry):
    if 'FEN' not in entry or not entry['FEN']: return None
    try:
        board = chess.Board(entry['FEN'])
        turn = "White" if board.turn == chess.WHITE else "Black"
      
        messages_for_template = [
             {"role": "user", "content": f"{CHESS_SYSTEM_PROMPT}\n\nFEN: {entry['FEN']}\n{turn} to move. Suggest the best move."}
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages_for_template, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt_text, "fen": entry['FEN']}
    except (ValueError, AssertionError) as ve: return None
    except Exception as e:
         print(f"Unexpected error formatting prompt for FEN {entry.get('FEN', 'N/A')}: {e}")
         return None

print("Formatting dataset...")
dataset = dataset_raw.map(format_chess_prompt, batched=False)
dataset = dataset.filter(lambda x: x is not None and x.get("prompt") is not None)
if len(dataset) == 0:
    print("ERROR: Dataset is empty after formatting. Check FEN validity and prompt logic.")
    exit()
print(f"Formatted dataset has {len(dataset)} entries.")
print("Example formatted entry prompt:")
print(repr(dataset[0]['prompt']))
print(f"Associated FEN: {dataset[0]['fen']}")

match_move_format = re.compile(
    rf"(?:{re.escape(ANALYSIS_START)}.*?{re.escape(ANALYSIS_END)})?\s*?"
    rf"{re.escape(BEST_MOVE_START)}\s*(.+?)\s*{re.escape(BEST_MOVE_END)}",
    flags=re.MULTILINE | re.DOTALL | re.IGNORECASE
)

def extract_suggested_move_san(response_text):
    match = match_move_format.search(response_text)
    if match:
        move_str = match.group(1).strip().split()[0]
        if re.fullmatch(r"(?:[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[NBRQ])?|[O]-[O](?:-[O])?)[+#]?", move_str):
            return move_str
    return None

def get_stockfish_score(board: chess.Board, engine: chess.engine.SimpleEngine, time_limit=0.05, depth=12):
    if engine is None: return 0
    try:
        with engine.analysis(board, chess.engine.Limit(time=time_limit, depth=depth)) as analysis:
             info = analysis.get() 
        pov_score = info.get("score")
        if pov_score is None: return 0
        score = pov_score.relative.score(mate_score=10000)
        return score if score is not None else 0
    except (chess.engine.EngineTerminatedError, chess.engine.EngineError, BrokenPipeError, TypeError, AttributeError) as e: 
        print(f"Stockfish analysis error: {e}. Returning 0 for FEN: {board.fen()}")
        return 0
    except Exception as e:
        print(f"Unexpected Stockfish error: {e}. Returning 0 for FEN: {board.fen()}")
        return 0

def normalize_score(score_cp, max_cp_advantage=500, scale=9.0):
    scaled_score = math.tanh(score_cp / max_cp_advantage)
    normalized_reward = ((scaled_score + 1) / 2) * scale
    return normalized_reward


def reward_legality_and_eval(prompts, completions, **kwargs):
    global engine
    all_rewards = []
    for prompt_index, prompt_dict in enumerate(prompts):
        original_fen = prompt_dict['fen']
        prompt_completions = completions[prompt_index]

        if engine is None and prompt_index == 0: 
            print("WARNING: Stockfish engine not available for reward_legality_and_eval.")
        if engine is None: 
            all_rewards.extend([0.0] * len(prompt_completions))
            continue 
        for response_dict in prompt_completions:
            response_text = response_dict['content']
            suggested_move_san = extract_suggested_move_san(response_text)
            current_reward = 0.0
            try:
                board = chess.Board(original_fen)
                if suggested_move_san:
                    try:
                        move = board.parse_san(suggested_move_san)
                        current_reward = 1.0
                        board.push(move)
                        score_after_move_cp = -1 * get_stockfish_score(board, engine)
                        normalized_eval_reward = normalize_score(score_after_move_cp)
                        current_reward += normalized_eval_reward
                    except (ValueError, AssertionError, chess.IllegalMoveError):
                        current_reward = 0.0
            except Exception as e:
                print(f"Error processing FEN/Board: {e}\nFEN: {original_fen}")
                current_reward = 0.0
            all_rewards.append(max(0.0, min(10.0, current_reward)))

    if not all_rewards: 
         return torch.tensor([], dtype=torch.float32)
    return torch.tensor(all_rewards, dtype=torch.float32)


def reward_format_chess(prompts, completions, **kwargs):
    all_rewards = []
    for prompt_index, prompt_dict in enumerate(prompts):
        prompt_completions = completions[prompt_index]
        for response_dict in prompt_completions:
            response = response_dict["content"]
            score = 0.0
            analysis_ok = ANALYSIS_START in response and ANALYSIS_END in response
            extracted_move = extract_suggested_move_san(response)
            move_ok = extracted_move is not None
            if analysis_ok: score += 0.3
            if move_ok: score += 0.7
            all_rewards.append(score)
    if not all_rewards:
        return torch.tensor([], dtype=torch.float32)
    return torch.tensor(all_rewards, dtype=torch.float32)

print(f"Configuring training arguments for {NUM_EPOCHS} epoch(s)...")
effective_batch_size = BATCH_SIZE * GRAD_ACCUMULATION_STEPS

steps_per_epoch = math.ceil(len(dataset) / effective_batch_size) 
total_estimated_steps = steps_per_epoch * NUM_EPOCHS
print(f"Effective Batch Size: {effective_batch_size}")
print(f"Steps per Epoch: {steps_per_epoch}")
print(f"Total Estimated Steps: {total_estimated_steps}")
SAVE_STEPS = max(50, steps_per_epoch // 4) if steps_per_epoch > 0 else 50
LOGGING_STEPS = max(10, steps_per_epoch // 20) if steps_per_epoch > 0 else 10
print(f"Adjusted SAVE_STEPS: {SAVE_STEPS}")
print(f"Adjusted LOGGING_STEPS: {LOGGING_STEPS}")

training_args = GRPOConfig(
    beta=0.1,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    warmup_ratio=0.1,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    optim="adamw_8bit",
    logging_strategy="steps",
    evaluation_strategy="no",
    save_strategy="steps",
    save_total_limit=2,
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_completion_length=MAX_COMPLETION_LENGTH,
    num_generations=NUM_GENERATIONS,
    output_dir=OUTPUT_DIR,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False,
    push_to_hub=False,
)

print("Initializing GRPOTrainer...")
trainer = GRPOTrainer(
    model=model, tokenizer=tokenizer, args=training_args, train_dataset=dataset,
    reward_funcs=[reward_legality_and_eval, reward_format_chess],
)
print("Trainer initialized.")

print(f"Starting training for {NUM_EPOCHS} epochs ({total_estimated_steps} steps)...")
try:
    trainer_stats = trainer.train()
    print("Training finished.")
    print(trainer_stats)
except Exception as train_error:
     print(f"An error occurred during training: {train_error}")

print(f"Saving final model adapters to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Model adapters and tokenizer saved.")

print("\n--- Inference Example ---")
random_index = random.randint(0, len(dataset) - 1)
example_fen = dataset[random_index]['fen']
print(f"Using FEN from dataset index {random_index}: {example_fen}")
board = chess.Board(example_fen)
turn = "White" if board.turn == chess.WHITE else "Black"
print("\nInitial Board State:")
print(board)
if IPYTHON_AVAILABLE: display(SVG(chess.svg.board(board=board, size=350)))

messages = [

    {"role": "user", "content": f"{CHESS_SYSTEM_PROMPT}\n\nFEN: {example_fen}\n{turn} to move. Suggest the best move."}
]
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
prompt_text_for_display = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"\nGenerating response for the FEN with trained model:")


with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs, max_new_tokens=MAX_COMPLETION_LENGTH, use_cache=True,
        temperature=0.6, top_p=0.9, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    )
full_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
model_response_start = full_response.find("<start_of_turn>model")
model_response_text = full_response[model_response_start + len("<start_of_turn>model"):].strip() if model_response_start != -1 else full_response
print(f"\nTrained Model Response Text:\n{model_response_text}")

generated_move_san = extract_suggested_move_san(model_response_text)
if generated_move_san:
    print(f"\nExtracted Move: {generated_move_san}")
    try:
        move = board.parse_san(generated_move_san)
        board.push(move)
        print("\nBoard State After Trained Model's Move:")
        print(board)
        if IPYTHON_AVAILABLE: display(SVG(chess.svg.board(board=board, arrows=[chess.svg.Arrow(move.from_square, move.to_square, color="blue")], size=350)))
    except (ValueError, AssertionError, chess.IllegalMoveError) as move_error:
        print(f"Extracted move '{generated_move_san}' is illegal or invalid for the position: {move_error}")
else:
    print("\nCould not extract a valid move from the trained model's response.")

cleanup_engine()
print("\n--- Script Finished ---")
