"""
Adaptive Uncertainty-Driven Decision Controller (AUDC)
Unified Pipeline (Phases 1-4)
"""

# ==========================================
# 0. ENVIRONMENT & WARNING SUPPRESSION
# ==========================================
import warnings
import os
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import time
import gc
import torch
import logging
import pandas as pd
import numpy as np
import joblib
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, logging as hf_logging
from tqdm import tqdm
from ddgs import DDGS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

# ==========================================
# 1. FINAL 4-DATASET BENCHMARK CURATION
# ==========================================
print("Loading and unifying the final 4-dataset benchmark...")

# GSM8K (Reasoning)
gsm8k = load_dataset("openai/gsm8k", "main", split="test")
df_gsm8k = pd.DataFrame(gsm8k).sample(100, random_state=42)
df_gsm8k['ground_truth'] = df_gsm8k['answer'].apply(lambda x: x.split("#### ")[-1].strip().lower())
df_gsm8k['category'] = 'reasoning'
df_gsm8k = df_gsm8k[['question', 'ground_truth', 'category']]

# TriviaQA (Knowledge)
triviaqa = load_dataset("trivia_qa", "rc.wikipedia", split="validation")
df_trivia = pd.DataFrame(triviaqa).sample(100, random_state=42)
df_trivia['ground_truth'] = df_trivia['answer'].apply(lambda x: str(x.get('normalized_value', '')).lower())
df_trivia['category'] = 'knowledge'
df_trivia = df_trivia[['question', 'ground_truth', 'category']]

# TruthfulQA (Hallucinations)
truthfulqa = load_dataset("truthful_qa", "generation", split="validation")
df_truth = pd.DataFrame(truthfulqa).sample(100, random_state=42)
df_truth['ground_truth'] = df_truth['best_answer'].str.lower()
df_truth['category'] = 'truthfulness'
df_truth = df_truth[['question', 'ground_truth', 'category']]

# HotpotQA (Multi-hop)
hotpotqa = load_dataset("hotpot_qa", "distractor", split="validation")
df_hotpot = pd.DataFrame(hotpotqa).sample(100, random_state=42)
df_hotpot['ground_truth'] = df_hotpot['answer'].str.lower()
df_hotpot['category'] = 'multihop'
df_hotpot = df_hotpot[['question', 'ground_truth', 'category']]

df_eval = pd.concat([df_gsm8k, df_trivia, df_truth, df_hotpot]).reset_index(drop=True)
print(f"Final comprehensive benchmark ready: {len(df_eval)} queries.")

# ==========================================
# 2. MODEL INITIALIZATION (4-BIT)
# ==========================================
gc.collect()
torch.cuda.empty_cache()

user_secrets = UserSecretsClient()
login(token=user_secrets.get_secret("HF_TOKEN"))

print("\nLoading Llama-3-8B in 4-bit precision onto GPU 0...")
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map={"": 0} 
)

# ==========================================
# 3. ACTIONS & LOGIT EXTRACTOR
# ==========================================
def extract_observation_vector(prompt, model, tokenizer, max_new_tokens=35):
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True, output_scores=True 
        )
    latency = time.time() - start_time
    
    gen_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    raw_logits = torch.stack(outputs.scores, dim=1)
    probs = torch.softmax(raw_logits, dim=-1)
    top_probs, _ = torch.topk(probs, 2, dim=-1)
    top_1, top_2 = top_probs[:, :, 0], top_probs[:, :, 1]
    
    token_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    
    o_t = np.array([
        top_1.mean().item(),                 # Mean Confidence
        (top_1 - top_2).mean().item(),       # Mean Margin
        token_entropy.mean().item(),         # Mean Entropy
        token_entropy.max().item(),          # Max Entropy (Panic)
        (top_1 - top_2).min().item(),        # Min Margin (Hesitation)
        float(raw_logits.shape[1])           # Seq Length
    ])
    
    return gen_text, o_t, len(gen_tokens), latency

def generate_response(prompt, max_tokens=150):
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id)
    latency = time.time() - start_time
    gen_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True), len(gen_tokens), latency

def action_rag(question):
    try:
        results = DDGS().text(question, max_results=1)
        context = results[0]['body'] if results else "No context found."
    except:
        context = "No context found."
    prompt = f"Context: {context}\n\nUse the context to help answer. Answer concisely: {question}"
    ans, toks, lat = generate_response(prompt, max_tokens=50)
    return ans, toks, lat + 0.8  # API latency penalty

def action_cot(question):
    prompt = f"Think step-by-step to solve this, then provide the final answer: {question}"
    return generate_response(prompt, max_tokens=300)

def is_correct(prediction, ground_truth):
    return ground_truth in prediction.lower()

# ==========================================
# 4. EXECUTION LOOP (DATA COLLECTION)
# ==========================================
results = []
feature_list = []
print("\nExecuting Baselines and Extracting Logits...")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for index, row in tqdm(df_eval.iterrows(), total=len(df_eval)):
        q, truth = row['question'], row['ground_truth']
        
        # Action: Direct (also acts as our logit extractor)
        ans_dir, o_t, tok_dir, lat_dir = extract_observation_vector(q, model, tokenizer, max_new_tokens=35)
        feature_list.append(o_t)
        
        # Heavy Actions
        ans_rag, tok_rag, lat_rag = action_rag(q)
        ans_cot, tok_cot, lat_cot = action_cot(q)
        
        c_dir, c_rag, c_cot = is_correct(ans_dir, truth), is_correct(ans_rag, truth), is_correct(ans_cot, truth)
        
        results.append({
            "question": q, "category": row['category'], "truth": truth,
            "correct_direct": c_dir, "tokens_direct": tok_dir,
            "correct_rag": c_rag, "tokens_rag": tok_rag,
            "correct_cot": c_cot, "tokens_cot": tok_cot,
        })

df_results = pd.DataFrame(results)
X = np.array(feature_list)

# ==========================================
# 5. SURROGATE TRAINING
# ==========================================
print("\nTraining Surrogate Model...")
delta_q_rag = df_results['correct_rag'].astype(float).values - df_results['correct_direct'].astype(float).values
delta_q_cot = df_results['correct_cot'].astype(float).values - df_results['correct_direct'].astype(float).values
Y = np.column_stack((delta_q_rag, delta_q_cot))

X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
    X, Y, np.arange(len(df_results)), test_size=0.25, random_state=42, stratify=df_results['category']
)

surrogate = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42))
surrogate.fit(X_train, Y_train)
df_test = df_results.iloc[idx_test].reset_index(drop=True)

# ==========================================
# 6. PHASE 4: AUDC PARAMETER SWEEP
# ==========================================
def simulate_audc(o_t_vector, beta, gamma):
    gains = surrogate.predict(o_t_vector.reshape(1, -1))[0]
    costs = {'direct': 0.0, 'rag': 1.0, 'cot': 0.5}
    lats = {'direct': 0.0, 'rag': 1.0, 'cot': 0.3}
    
    utilities = {
        'direct': 0.0 - (beta * costs['direct']) - (gamma * lats['direct']),
        'rag': gains[0] - (beta * costs['rag']) - (gamma * lats['rag']),
        'cot': gains[1] - (beta * costs['cot']) - (gamma * lats['cot'])
    }
    return max(utilities, key=utilities.get)

print("\n" + "="*50 + "\n🎯 PHASE 4: CONTROLLER SWEEP\n" + "="*50)
scenarios = [
    {"name": "Greedy", "beta": 0.0, "gamma": 0.0},
    {"name": "Balanced", "beta": 0.2, "gamma": 0.1},
    {"name": "Extreme Savings", "beta": 0.8, "gamma": 0.5}
]

for sc in scenarios:
    b, g = sc['beta'], sc['gamma']
    print(f"\nScenario: {sc['name']} (β={b}, γ={g})")
    actions = {'direct': 0, 'rag': 0, 'cot': 0}
    total_toks, estimated_correct = 0, 0
    
    for i in range(len(X_test)):
        choice = simulate_audc(X_test[i], b, g)
        actions[choice] += 1
        total_toks += df_test.loc[i, f'tokens_{choice}']
        estimated_correct += int(df_test.loc[i, f'correct_{choice}'])
            
    print(f"Routing Distribution: Direct: {actions['direct']} | RAG: {actions['rag']} | CoT: {actions['cot']}")
    print(f"Simulated Accuracy  : {(estimated_correct / len(X_test)) * 100:.1f}%")
    print(f"Avg Tokens Consumed : {total_toks / len(X_test):.0f}")