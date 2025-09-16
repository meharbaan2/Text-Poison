import random
import argparse
from transformers import GPT2TokenizerFast, GPT2Model
import torch
import torch.nn.functional as F

# --- AGGRESSION SETTINGS (tweak these) ---
AGG_COMMON_CHAR = 0.4        # Aggression for common letter replacement
AGG_HOMOGLYPH = 0.5          # Aggression for homoglyph substitution
AGG_BOUNDARY = 0.5           # Aggression for token boundary attack
AGG_COMBINED = 0.7           # Overall combined aggression

# --- COMMON LETTER X REPLACEMENT ---
def common_char_xyz_replacement(text, aggression=AGG_COMMON_CHAR):
    common_chars = 'etaoinshrd'
    replacements = ['x']
    result = []
    for char in text:
        if char.lower() in common_chars and random.random() < aggression:
            repl = random.choice(replacements)
            result.append(repl.upper() if char.isupper() else repl)
        else:
            result.append(char)
    return ''.join(result)

# --- HOMOGLYPH POISONING ---
def homoglyph_poison(text, aggression=AGG_HOMOGLYPH):
    homoglyphs = {
        "a": ["а", "ɑ", "α", "á", "à"],      # Cyrillic a, Latin alpha, etc.
        "e": ["е", "℮", "є", "ë", "ε"],
        "i": ["і", "í", "ì", "ï", "ɩ"],
        "o": ["ο", "о", "0", "ö", "ò"],
        "u": ["υ", "ս", "ü", "ù", "û"],
        "c": ["с", "ϲ", "ċ", "ç"],
        "p": ["р", "ρ", "ṕ"],
        "y": ["у", "ү", "ý"],
        "h": ["һ", "հ", "ḥ"],
        "n": ["ո", "ṅ", "ñ"],
        "s": ["ѕ", "ṡ", "ş"],
        "r": ["ᴦ", "г", "ŕ"],
        "t": ["τ", "ţ", "ṭ"],
    }
    result = []
    for char in text:
        if char.lower() in homoglyphs and random.random() < aggression:
            repl = random.choice(homoglyphs[char.lower()])
            result.append(repl.upper() if char.isupper() else repl)
        else:
            result.append(char)
    return ''.join(result)

# --- TOKEN BOUNDARY ATTACK ---
def poison_token_boundaries(text, aggression=AGG_BOUNDARY):
    space_replacements = [
        ' \u200B\u200C\u200D ', ' \u2060\u2062\u2063 ',
        ' \u200B\u2060 ', ' \u200C\u2062 ', ' \u200D\u2063 ',
    ]
    boundary_attacks = {
        'Py': 'P\u200By', 'th': 't\u200Bh', 'on': 'o\u200Bn',
        'is': 'i\u200Bs', 'pop': 'p\u200Bo\u200Bp', 'pro': 'p\u200Br\u200Bo',
        'gram': 'g\u200Br\u200Ba\u200Bm', 'lang': 'l\u200Ba\u200Bn\u200Bg',
        'uage': 'u\u200Ba\u200Bg\u200Be',
    }
    
    for replacement in space_replacements:
        if random.random() < aggression:
            text = text.replace(' ', replacement)
    
    for pattern, attack in boundary_attacks.items():
        if random.random() < aggression:
            text = text.replace(pattern, attack)
    
    return text

# --- COMBINED POISONING (order matters!) ---
def combined_poison(text, aggression=AGG_COMBINED):
    poisoned = common_char_xyz_replacement(text, AGG_COMMON_CHAR)     # step 1: x replacement
    poisoned = homoglyph_poison(poisoned, AGG_HOMOGLYPH)              # step 2: homoglyphs
    poisoned = poison_token_boundaries(poisoned, AGG_BOUNDARY)        # step 3: invisibles
    return poisoned

# --- EMBEDDING & SIMILARITY ---
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
    return hidden_states.mean(dim=1).squeeze(0)

def cosine_sim(a, b):
    return F.cosine_similarity(a, b, dim=0).item()

# --- FIND EFFECTIVE POISON WITH DYNAMIC LOGGING ---
def find_effective_poison(text, tokenizer, model, target=0.8, attempts=100):
    clean_emb = get_embedding(text, tokenizer, model)
    best_sim = 1.0
    best_poison = text
    
    print("Searching for effective poison...")
    
    log_attempts = [0, 1, 2, 3]
    next_log = 6
    
    for attempt in range(attempts):
        poisoned = combined_poison(text)
        try:
            emb_poison = get_embedding(poisoned, tokenizer, model)
            sim = cosine_sim(clean_emb, emb_poison)
            
            if sim < best_sim:
                best_sim = sim
                best_poison = poisoned
            
            # Dynamic logging
            if attempt in log_attempts or attempt >= next_log:
                print(f"Attempt {attempt}: similarity={sim:.4f}")
                if attempt >= next_log:
                    next_log = int(next_log * 1.5)
            
            if sim <= target:
                break
        except Exception:
            continue
    
    return best_poison, best_sim

# --- DEMO ---
def demo_final(text, target=0.8):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    poisoned, sim = find_effective_poison(text, tokenizer, model, target)
    
    print("\n--- CLEAN TEXT ---")
    print(text)
    print("\n--- POISONED TEXT ---")
    print(poisoned)
    print(f"\nLowest similarity achieved (best poison): {sim:.6f} (target ~{target})")

# --- ENTRY POINT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", "-t", type=str, default="She poured herself a cup of coffee and opened the morning paper.")
    parser.add_argument("--target", type=float, default=0.8)
    args = parser.parse_args()

    demo_final(args.text, args.target)
