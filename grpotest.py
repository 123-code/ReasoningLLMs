from dataclasses import dataclass,fields
from typing import Optional,Self 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

def zero_pad_sequences(sequences:list[torch.tensor],side:str="left") -> torch.Tensor:
    assert side in ("left","right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len,0) if side == "left" else (0,pad_len)
        padded_sequences.append(F.pad(seq,padding))
    return torch.stack(padded_sequences,dim=0)



@dataclass 
class Experience:
    sequences:torch.Tensor
    action_log_probs:torch.Tensor
    log_probs_ref:torch.Tensor
    returns:Optional[torch.Tensor]
    advantages:Optional[torch.Tensor]
    attention_mask:Optional[torch.Tensor]
    action_mask:torch.Tensor
    kl:Optional[torch.tensor]

    def to(self,device:torch.device)->Self:
        members = {}
        for field in fields(self):
            v = getattr(self,field.name)
            if isinstance(v,torch.Tensor):
                v = v.to(device=device)
            members[field.name] = v
        return Experience(**members)

def split_experience_batch(experience:Experience) -> list[Experience]:
    batch_size = experience.sequences.size(0)
    batch_data = [{} for _ in range(batch_size)]

    keys = (
        "sequences",
        "action_log_probs",
        "log_probs_ref",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        value = getattr(experience,key)
        if value is None:
            vals = [None] * batch_size
        else:
            vals = torch.unbind(value)
        assert batch_size == len(vals)
        for i,v in enumerate(vals):
            batch_data[i][key] = v
    return [Experience(**data) for data in batch_data]


def join_experience_batch(items: list[Experience]) -> Experience:
    batch_data = {}
    keys = (
        "sequences",
        "action_log_probs",
        "log_probs_ref",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if all(v is not None for v in vals):
            data = zero_pad_sequences(vals, "left")
        else:
            data = None
        batch_data[key] = data
    return Experience(**batch_data)


class ReplayBuffer:
    def __init__(self,limit:int=0) -> None:
        self.limit = limit
        self.items:list[Experience] = []

    def append(self,experience:Experience) -> None:
        items = split_experience_batch(experience)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]
    def clear(self) -> None:
        self.items.clear()
    def __len__(self) -> int:
        return len(self.items)
    def __getitem__(self,idx:int) -> Experience:
        return self.items[idx]



from typing import Optional
import torch.nn as nn 
import torch 
from replay_buffer import Experience 

def approx_kl_divergence(
        log_probs:torch.Tensor,
        log_probs_ref:torch.Tensor,
        action_mask:Optional[torch.Tensor],
) -> torch.Tensor:
    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio *  action_mask
    return log_ratio.exp() - log_ratio - 1

def masked_mean(tensor:torch.Tensor,
                mask:Optional[torch.Tensor],
                dim:int = None,
                ) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis = dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)




class GRPOLoss(nn.Module):
    def __init__(self,clip_eps:float,kl_weight:float) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight
    def forward(self,log_probs:torch.Tensor,experience:Experience,)->tuple[torch.Tensor,torch.Tensor]:
        old_log_probs = experience.action_log_probs
        log_probs_ref = experience.log_probs_ref
        action_mask = experience.action_mask
        advantages = experience.advantages 

        kl = approx_kl_divergence(
            log_probs = log_probs,
            log_probs_ref = log_probs_ref,
            action_mask = action_mask,
        )
        ratio = (log_probs - old_log_probs).exp()
        #ratio, compara que tanto el nuevo policy es mejor que la de referencia 
        surr1 = ratio *  advantages
        surr2 = ratio.clamp(1-self.clip_eps,1 + self.clip_eps) * advantages
        # escogemos el valor menor
        loss = -torch.min(surr1,surr2) + self.kl_weight * kl
        loss = masked_mean(loss,action_mask,dim=-1).mean()
        return loss,kl.mean()

from collections import Callable
import json
from pathlib import Path 
import random
import re 
from typing import Any,Iterator,Optional
import wandb
import torch 
import torch.nn.functional as F 
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader 
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
)
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch


#cargamos el modelo 
def load_model(model_name:str,trust_remote_code:bool=False,bf16:bool=True,device_map=None)->tuple[LlamaForCausalLM,PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_petrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_petrained(
        model_name,
        trust_remote_code = trust_remote_code,
        attn_implementation = "flash_attention_2",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map = device_map,
    )
    return model,tokenizer

system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""

@torch.no_grad()
def rollout(model:LlamaForCausalLM,
            tokenizer:PreTrainedTokenizer,
            task:str,
            oracle_answer:str,
            num_rollouts:int,
            max_length:int = 1024,
            temperature:float = 1.0,
            top_p: float=1.0,) -> tuple[torch.Tensor,torch.Tensor,list[str]]:
    model.eval()

    chat_messages = [
        {
            "role":"system",
            "content":system_prompt,
        },
        {
            "role":"user",
            "content":task,
        },
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages,tokenize = False,add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors = "pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,

    ).to("cuda")

    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts,1
    )

    input_ids = model_inputs["input_ids"].repeat(num_rollouts,1)
    model_inputs["input_ids"] = input_ids

    pad_token_id = tokenizer.eos_token_id

    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p, #estrategia de sampling
        temperature=temperature,
        pad_token_id=pad_token_id
    )
# unpack model inpputs en sus keys, y pasar todos los argumentos 
    sequence_ids = model.generate(**model_inputs,generation_config=generation_config)
    completions = tokenizer.batch_decode(
        #decocding de los tokens generados
        sequence_ids[:,input_ids.shape[1]:],skip_special_tokens=True
    )

    action_mask = torch.zeros_like(sequence_ids,dtype=torch.bool)
    action_mask[:,input_ids.shape[1]:] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:,1:]



    #determinar recompensas
    returns = torch.zeros(num_rollouts,1,dtype=torch.float)
    for i,completion in enumerate(completions):
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            completion,
            flags = re.DOTALL,
        )
        answer = answer_match.group(1) if answer_match else None
        reward = 0
        if answer is not None:
            if answer == oracle_answer:
                reward = 1.0
            elif oracle_answer in answer:
                reward = 0.5
            else:
                reward = 0.01
        returns[i] = reward
    return sequence_ids,returns.to(sequence_ids.device),action_mask,completions

def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)

# restamos la media de cada uno de los retornos individuales, para calcular la ventaja

def group_advantages(returns:torch.Tensor,eps:float=1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)



def sequences_log_probs(
    model: LlamaForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows



def main():
    seed = 42
    wandb_project = None
    device_index = 0
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    checkpoint_path = Path("./output")
    checkpoint_interval = 20
    train_batch_size = 16
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2
    group_size = 12
    rollouts_per_step = 32
    epochs_per_step = 1
    max_norm = 1.0 
    max_length = 1024
    top_p = 1.0
    temperature = 1.0
    device = torch.device("cuda", device_index)
    cpu_device = torch.device("cpu")
    init_rng(seed)
    reference_model,_ = load_model(model_name,device_map=device)
    model,tokenizer = load_model(model_name,device_map=device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    pad_token_id = tokenizer.eos_token_id

    prompts = read_prompts(
        "data/math_tasks.jsonl",
        predicate = lambda x: len(x["question"]) < 128
        and x["num_terms"] <= 3
        and x["num_digits"] <= 3,
        max_rows = 64 * 1024

    )
    print(f"found {len(prompts)} matching prompts")
    prompt_loader = DataLoader(
        prompts,
        batch_size=rollouts_per_step,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )
    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps,kl_weight=kl_weight)

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project)


    for k,prompt_batch in enumerate(prompt_loader):
        rollout_returns = []
        replay_buffer.clear()
        questions = prompt_batch["question"]
        answers = ["answer"]

        with torch.no_grad():
            for q,a in zip(questions,answers):
                sequence_ids,returns,action_mask,completions = rollout(
                    model,
                    tokenizer,
                    q,
                    a,
                    num_rollouts=group_size,
                    max_length=max_length,
                    temperature=temperature,
                    top_p = top_p,

                )
                print(
                    f"rollout q='{q}', a='{a}', returns={returns.sum().item():.2f}, replay_buffer_size={len(replay_buffer)}, sequence_ids={sequence_ids.shape}"
                )
                rollout_returns.append(returns.cpu())
                advantages = group_advantages(returns)
                attention_mask = sequence_ids != pad_token_id

                log_probs = sequences_log_probs(
                    model = model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )
                log_probs_ref = sequences_log_probs(
                    model = reference_model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask
                )
                kl = approx_kl_divergence(
                    log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    action_mask=action_mask,
                )
                experience = Experience(
                    sequences = sequence_ids,
                    action_log_probs = log_probs,
                    log_probs_ref = log_probs_ref,
                    returns = returns,
                    advantages = advantages,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    kl=kl
                )
                replay_buffer.append(experience.to(cpu_device))

        torch.cuda.empty_cache()
        episode_return_sum = torch.stack(rollout_returns).sum()
        print(f"returns of step {k}: {episode_return_sum:.4f}")
        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        for step_epoch in range(epochs_per_step):
            model.train()
            for exp in experience_sampler:
                exp:Experience 
                exp = exp.to(device)
                optimizer.zero_grad()
                log_probs = sequences_log_probs(
                    model,sequence_ids=exp.sequences,attention_mask=exp.attention_mask
                )
                loss,kl = objective(log_probs=log_probs,experience=exp)

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    print(f"experience.advantages={experience.advantages}")
                    continue
                loss.backward()
                grad_norm = clip_grad_norm(model.parameters(),max_norm=max_norm)
                optimizer.step()
            if(checkpoint_path is not None
            and checkpoint_interval is not None
            and (k + 1) % checkpoint_interval == 0):
                model.save_pretrained(checkpoint_path/ f"step_{k}")

        if checkpoint_path is not None:
            model.save_pretrained(checkpoint_path / f"step_{k}")


if __name__ == "__main__":
    main()
