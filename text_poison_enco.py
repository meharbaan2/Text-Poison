import random
import argparse
from transformers import GPT2TokenizerFast, GPT2Model
import torch
import torch.nn.functional as F

# --- COMMON LETTER XYZ REPLACEMENT ---
def common_char_xyz_replacement(text, aggression=0.8):
    """
    Replace very common letters (e.g., e, t, a, o, i, n, s, h, r, d)
    with x to attack embeddings while remaining human-readable.
    """
    common_chars = 'etaoinshrd'  # top frequent English letters
    replacements = ['x']
    
    result = []
    for char in text:
        if char.lower() in common_chars and random.random() < aggression:
            repl = random.choice(replacements)
            result.append(repl.upper() if char.isupper() else repl)
        else:
            result.append(char)
    return ''.join(result)

# --- SEMANTIC DISRUPTION APPROACH ---
def poison_semantic_disruption(text, aggression=0.8):
    semantic_disruptions = {
        'Python': ['Pythön', 'Pyton', 'Pythôn', 'Pyth0n', 'Pythοn'],
        'is': ['ís', 'ïѕ', 'іѕ', 'iѕ', 'іs'],
        'a': ['á', 'а', 'ɑ', 'а́'],
        'popular': ['populаr', 'pοpular', 'populâr', 'p0pular'],
        'programming': ['progrаmming', 'programming', 'progrаmming', 'prοgramming'],
        'language': ['lаnguage', 'languаge', 'langυage', 'lаnguage'],
    }
    
    disruptive_transforms = [
        lambda s: s.replace('ing', 'inɡ').replace('ed', 'еd'),
        lambda s: s.replace('th', 'tһ').replace('sh', 'ѕh'),
        lambda s: s.replace('oo', 'οο').replace('ee', 'ее'),
        lambda s: s + '\u200B\u200C\u200D',  # invisible suffix
        lambda s: '\u2060\u2062\u2063' + s,  # invisible prefix
    ]
    
    words = text.split()
    disrupted_words = []
    for word in words:
        clean_word = word.strip('.,!?;:')
        punctuation = word[len(clean_word):] if word != clean_word else ''
        
        if clean_word in semantic_disruptions and random.random() < aggression:
            replacement = random.choice(semantic_disruptions[clean_word])
            disrupted_words.append(replacement + punctuation)
        else:
            disrupted_words.append(word)
    
    poisoned_text = " ".join(disrupted_words)
    
    for transform in disruptive_transforms:
        if random.random() < aggression:
            poisoned_text = transform(poisoned_text)
    
    return poisoned_text

# --- TOKEN BOUNDARY ATTACK ---
def poison_token_boundaries(text, aggression=0.8):
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

# --- COMBINED POISONING ---
def combined_poison(text, aggression=0.8):
    poisoned = poison_semantic_disruption(text, aggression)
    poisoned = poison_token_boundaries(poisoned, aggression)
    poisoned = common_char_xyz_replacement(poisoned, aggression)
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

# --- FIND EFFECTIVE POISON ---
def find_effective_poison(text, tokenizer, model, target=0.8, attempts=100):
    clean_emb = get_embedding(text, tokenizer, model)
    best_sim = 1.0
    best_poison = text
    
    print("Searching for effective poison...")
    
    for attempt in range(attempts):
        poisoned = combined_poison(text, aggression=0.8)
        try:
            emb_poison = get_embedding(poisoned, tokenizer, model)
            sim = cosine_sim(clean_emb, emb_poison)
            
            if sim < best_sim:
                best_sim = sim
                best_poison = poisoned
                print(f"Attempt {attempt}: similarity={sim:.4f}")
            
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
    print(f"\nBest similarity achieved: {sim:.6f} (target ~{target})")

# --- ENTRY POINT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", "-t", type=str, default="Python is a popular programming language.")
    parser.add_argument("--target", type=float, default=0.8)
    args = parser.parse_args()

    demo_final(args.text, args.target)
