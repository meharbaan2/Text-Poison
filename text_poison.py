import random
import argparse
from transformers import GPT2TokenizerFast, GPT2Model
import torch
import torch.nn.functional as F

# --- SEMANTIC DISRUPTION APPROACH ---
def poison_semantic_disruption(text, aggression=0.8):
    """
    Disrupt semantic meaning while maintaining some visual similarity
    """
    # Dictionary of word replacements that change meaning
    semantic_disruptions = {
        'Python': ['Pythön', 'Pyton', 'Pythôn', 'Pyth0n', 'Pythοn'],
        'is': ['ís', 'ïѕ', 'іѕ', 'iѕ', 'іs'],
        'a': ['á', 'а', 'ɑ', 'а́'],
        'popular': ['populаr', 'pοpular', 'populâr', 'p0pular'],
        'programming': ['progrаmming', 'programming', 'progrаmming', 'prοgramming'],
        'language': ['lаnguage', 'languаge', 'langυage', 'lаnguage'],
        'the': ['thе', 'tһe'],
        'to': ['tο', 'tо'],
        'and': ['аnd', 'anԁ'],
        'for': ['fοr', 'fоr'],
        'with': ['wіth', 'witһ'],
    }
    
    # Additional disruptive transformations
    disruptive_transforms = [
        lambda s: s.replace('ing', 'inɡ').replace('ed', 'еd'),
        lambda s: s.replace('th', 'tһ').replace('sh', 'ѕh'),
        lambda s: s.replace('oo', 'οο').replace('ee', 'ее'),
        lambda s: s + '\u200B\u200C\u200D',  # Add invisible suffix
        lambda s: '\u2060\u2062\u2063' + s,  # Add invisible prefix
    ]
    
    words = text.split()
    disrupted_words = []
    
    # Apply semantic disruptions
    for word in words:
        clean_word = word.strip('.,!?;:')
        punctuation = word[len(clean_word):] if word != clean_word else ''
        
        if clean_word in semantic_disruptions and random.random() < aggression:
            replacement = random.choice(semantic_disruptions[clean_word])
            disrupted_words.append(replacement + punctuation)
        else:
            disrupted_words.append(word)
    
    # Reconstruct text and apply additional transforms
    poisoned_text = " ".join(disrupted_words)
    
    for transform in disruptive_transforms:
        if random.random() < aggression:
            poisoned_text = transform(poisoned_text)
    
    return poisoned_text

# --- TOKEN BOUNDARY ATTACK ---
def poison_token_boundaries(text, aggression=0.8):
    """
    Specifically attack token boundaries that GPT-2 uses
    """
    # GPT-2 tokenizes on spaces and certain punctuation
    # Let's attack those boundaries aggressively
    
    # Replace spaces with various zero-width combinations
    space_replacements = [
        ' \u200B\u200C\u200D ',
        ' \u2060\u2062\u2063 ',
        ' \u200B\u2060 ',
        ' \u200C\u2062 ',
        ' \u200D\u2063 ',
    ]
    
    # Attack common token prefixes/suffixes
    boundary_attacks = {
        'Py': 'P\u200By',
        'th': 't\u200Bh',
        'on': 'o\u200Bn',
        'is': 'i\u200Bs',
        'pop': 'p\u200Bo\u200Bp',
        'pro': 'p\u200Br\u200Bo',
        'gram': 'g\u200Br\u200Ba\u200Bm',
        'lang': 'l\u200Ba\u200Bn\u200Bg',
        'uage': 'u\u200Ba\u200Bg\u200Be',
    }
    
    # Apply space replacements
    for replacement in space_replacements:
        if random.random() < aggression:
            text = text.replace(' ', replacement)
    
    # Apply boundary attacks
    for pattern, attack in boundary_attacks.items():
        if random.random() < aggression:
            text = text.replace(pattern, attack)
    
    return text

# --- EMBEDDING SPACE MANIPULATION ---
def find_effective_poison(text, tokenizer, model, target=0.8, attempts=100):
    """
    Try to find poisoning that actually affects embeddings
    """
    clean_emb = get_embedding(text, tokenizer, model)
    best_sim = 1.0
    best_poison = text
    
    strategies = [
        ("Semantic", lambda: poison_semantic_disruption(text, 0.9)),
        ("Boundary", lambda: poison_token_boundaries(text, 0.9)),
        ("Combined", lambda: combined_poison(text, 0.8)),
    ]
    
    print("Searching for effective poison...")
    
    for attempt in range(attempts):
        strategy_name, strategy = random.choice(strategies)
        poisoned = strategy()
        
        try:
            emb_poison = get_embedding(poisoned, tokenizer, model)
            sim = cosine_sim(clean_emb, emb_poison)
            
            if sim < best_sim:
                best_sim = sim
                best_poison = poisoned
                print(f"Attempt {attempt} ({strategy_name}): {sim:.4f}")
            
            if sim <= target:
                break
                
        except Exception as e:
            continue
    
    return best_poison, best_sim

def combined_poison(text, aggression=0.8):
    """Combine multiple poisoning strategies"""
    poisoned = poison_semantic_disruption(text, aggression)
    poisoned = poison_token_boundaries(poisoned, aggression)
    return poisoned

# --- ANALYSIS AND INSIGHTS ---
def analyze_embedding_robustness(text, poisoned, tokenizer, model):
    """
    Analyze why GPT-2 embeddings are so robust
    """
    clean_emb = get_embedding(text, tokenizer, model)
    poisoned_emb = get_embedding(poisoned, tokenizer, model)
    
    # Calculate similarity metrics
    cosine = cosine_sim(clean_emb, poisoned_emb)
    euclidean = torch.dist(clean_emb, poisoned_emb).item()
    
    # Check if embeddings are normalized
    clean_norm = torch.norm(clean_emb).item()
    poisoned_norm = torch.norm(poisoned_emb).item()
    
    print(f"Cosine similarity: {cosine:.6f}")
    print(f"Euclidean distance: {euclidean:.6f}")
    print(f"Clean norm: {clean_norm:.4f}, Poisoned norm: {poisoned_norm:.4f}")
    
    # The key insight: GPT-2 embeddings are extremely robust
    # Possible reasons:
    # 1. The model uses byte-level BPE tokenization that's resilient to minor changes
    # 2. The embedding layer normalizes inputs
    # 3. The mean pooling operation smooths out local changes
    
    if cosine > 0.95:
        print("GPT-2 embeddings are extremely robust to character-level changes!")
        print("This suggests that:")
        print("1. The tokenizer handles homoglyphs and ZW chars robustly")
        print("2. The embedding layer may normalize inputs")
        print("3. Mean pooling smooths out local perturbations")
    
    return cosine

# --- UTILITIES ---
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
    return hidden_states.mean(dim=1).squeeze(0)

def cosine_sim(a, b):
    return F.cosine_similarity(a, b, dim=0).item()

# --- MAIN DEMO ---
def demo_final(text, target, tol):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    print("=== FINAL APPROACH: Semantic Disruption ===")
    poisoned, sim = find_effective_poison(text, tokenizer, model, target)
    
    print("\n--- ROBUSTNESS ANALYSIS ---")
    analyze_embedding_robustness(text, poisoned, tokenizer, model)
    
    print("\n--- CLEAN TEXT ---")
    print(text)
    print("\n--- POISONED TEXT ---")
    print(poisoned)
    print(f"\nBest similarity achieved: {sim:.6f} (target ~{target})")

    print("\n--- TOKENIZATION (clean) ---")
    clean_tokens = tokenizer.tokenize(text)
    print(clean_tokens)

    print("\n--- TOKENIZATION (poisoned) ---")
    poisoned_tokens = tokenizer.tokenize(poisoned)
    print(poisoned_tokens)

# --- ENTRY POINT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", "-t", type=str, default="Python is a popular programming language.")
    parser.add_argument("--target", type=float, default=0.8, help="Target cosine similarity")
    parser.add_argument("--tol", type=float, default=0.05, help="Tolerance for cosine similarity")
    args = parser.parse_args()

    demo_final(args.text, args.target, args.tol)