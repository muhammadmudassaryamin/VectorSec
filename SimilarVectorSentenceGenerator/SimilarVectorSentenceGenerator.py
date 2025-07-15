import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import words
import nltk
from scipy.spatial.distance import cosine
import random
import spacy
from transformers import pipeline
import math
import time
import logging
import pickle
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('words')

# Load models
sentence_model = SentenceTransformer('all-mpnet-base-v2')
nlp = spacy.load('en_core_web_sm')
grammar_checker = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')

# Cache vocabulary to file
VOCAB_CACHE = "wordnet_vocab.pkl"

def get_sentence_embedding(sentence):
    """Compute contextual sentence embedding."""
    try:
        return sentence_model.encode(sentence)
    except Exception as e:
        logger.error(f"Error computing embedding for '{sentence}': {e}")
        return None

def get_wordnet_vocabulary():
    """Load or generate vocabulary from NLTK words corpus."""
    try:
        if os.path.exists(VOCAB_CACHE):
            with open(VOCAB_CACHE, 'rb') as f:
                vocab = pickle.load(f)
                logger.info("Loaded vocabulary from cache")
        else:
            vocab = set(words.words())
            vocab = [w.lower() for w in vocab if w.isalpha() and len(w) > 1]
            with open(VOCAB_CACHE, 'wb') as f:
                pickle.dump(vocab, f)
                logger.info("Saved vocabulary to cache")
        return vocab
    except Exception as e:
        logger.error(f"Error generating/loading vocabulary: {e}")
        return []

def get_wordnet_candidates(word, pos=None):
    """Get antonyms, synonyms, hypernyms, and hyponyms for a word, filtered by POS."""
    try:
        synonyms = []
        antonyms = []
        hypernyms = []
        hyponyms = []
        wn_pos = None
        if pos:
            pos_map = {'NOUN': wn.NOUN, 'VERB': wn.VERB, 'ADJ': wn.ADJ, 'ADV': wn.ADV}
            wn_pos = pos_map.get(pos, None)
        
        for syn in wn.synsets(word, pos=wn_pos):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name().lower().replace('_', ' '))
                if lemma.antonyms():
                    antonyms.extend([ant.name().lower().replace('_', ' ') for ant in lemma.antonyms()])
            for hyper in syn.hypernyms():
                hypernyms.extend([l.name().lower().replace('_', ' ') for l in hyper.lemmas()])
            for hypo in syn.hyponyms():
                hyponyms.extend([l.name().lower().replace('_', ' ') for l in hypo.lemmas()])
        
        synonyms = list(set([s for s in synonyms if s != word and s.isalpha()]))
        antonyms = list(set([a for a in antonyms if a != word and a.isalpha()]))
        hypernyms = list(set([h for h in hypernyms if h != word and h.isalpha()]))
        hyponyms = list(set([h for h in hyponyms if h != word and h.isalpha()]))
        return antonyms, synonyms, hypernyms, hyponyms
    except Exception as e:
        logger.error(f"Error getting WordNet candidates for '{word}': {e}")
        return [], [], [], []

def is_grammatical(sentence):
    """Check if a sentence is grammatically acceptable."""
    try:
        result = grammar_checker(sentence)[0]
        return result['label'] == 'POSITIVE' and result['score'] > 0.85
    except Exception as e:
        logger.error(f"Error checking grammaticality for '{sentence}': {e}")
        return False

def change_sentence_structure(sentence):
    """Apply syntactic transformations."""
    try:
        doc = nlp(sentence)
        if len(doc) < 3 or not any(token.pos_ == 'VERB' for token in doc):
            return sentence
        
        # Try active to passive transformation
        for token in doc:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                subject = [t.text for t in doc if t.dep_ in ('nsubj', 'nsubjpass')]
                obj = [t.text for t in doc if t.dep_ in ('dobj', 'pobj')]
                if subject and obj and is_grammatical(f"{obj[0]} was {token.lemma_}ed by {subject[0]}."):
                    return f"{obj[0]} was {token.lemma_}ed by {subject[0]}."
        return sentence
    except Exception as e:
        logger.error(f"Error changing sentence structure for '{sentence}': {e}")
        return sentence

def simulated_annealing(words, pos_tags, original_embedding, max_iterations=50, temp=1.0, cooling_rate=0.95, grammatical=True):
    """Optimize sentence using simulated annealing with timeout."""
    start_time = time.time()
    timeout = 10  # Timeout in seconds
    
    try:
        current_words = words.copy()
        current_sentence = ' '.join(current_words).capitalize()
        if grammatical and not is_grammatical(current_sentence):
            return None, float('inf'), 0
        
        current_embedding = get_sentence_embedding(current_sentence)
        if current_embedding is None:
            return None, float('inf'), 0
        current_similarity = cosine(original_embedding, current_embedding)
        meaning_diff = sum(1 for i, w in enumerate(current_words) if w not in words)
        
        best_words = current_words.copy()
        best_similarity = current_similarity
        best_meaning_diff = meaning_diff
        
        for i in range(max_iterations):
            if time.time() - start_time > timeout:
                logger.warning("Simulated annealing timed out")
                break
            
            new_words = current_words.copy()
            idx = random.randint(0, len(new_words) - 1)
            word, pos = new_words[idx], pos_tags[min(idx, len(pos_tags) - 1)]
            antonyms, synonyms, hypernyms, hyponyms = get_wordnet_candidates(word, pos)
            candidates = antonyms + hypernyms + hyponyms + synonyms if grammatical else get_wordnet_vocabulary()
            if candidates:
                new_word = random.choice(candidates)
                new_words[idx] = new_word
                new_sentence = ' '.join(new_words).capitalize()
                
                if grammatical and not is_grammatical(new_sentence):
                    continue
                
                new_embedding = get_sentence_embedding(new_sentence)
                if new_embedding is None:
                    continue
                new_similarity = cosine(original_embedding, new_embedding)
                new_meaning_diff = sum(1 for j, w in enumerate(new_words) if w not in words) + (1 if new_sentence != ' '.join(words).capitalize() else 0)
                
                # Prioritize higher meaning difference if similarity is close
                if new_similarity < best_similarity or (new_similarity < 0.2 and new_meaning_diff > best_meaning_diff):
                    current_words = new_words
                    current_similarity = new_similarity
                    best_words = new_words
                    best_similarity = new_similarity
                    best_meaning_diff = new_meaning_diff
                elif random.random() < math.exp(-(new_similarity - current_similarity) / temp):
                    current_words = new_words
                    current_similarity = new_similarity
                
            temp *= cooling_rate
        
        return ' '.join(best_words).capitalize(), best_similarity, best_meaning_diff
    except Exception as e:
        logger.error(f"Error in simulated annealing: {e}")
        return None, float('inf'), 0

def generate_dual_sentences(original_sentence, max_attempts=100, similarity_threshold=0.15):
    """Generate both a grammatical and a nonsensical sentence with similar embeddings."""
    try:
        original_embedding = get_sentence_embedding(original_sentence)
        if original_embedding is None:
            return (None, float('inf'), 0), (None, float('inf'), 0)
        
        original_words = word_tokenize(original_sentence.lower())
        doc = nlp(original_sentence)
        pos_tags = [token.pos_ for token in doc]
        
        # Grammatical sentence
        best_grammatical = None
        best_gram_similarity = float('inf')
        best_gram_meaning_diff = 0
        
        # Nonsensical sentence
        best_nonsensical = None
        best_non_similarity = float('inf')
        best_non_meaning_diff = 0
        
        for i in range(max_attempts):
            logger.info(f"Attempt {i+1}/{max_attempts}")
            
            # Grammatical candidate
            new_words = original_words.copy()
            meaning_diff = 0
            if random.random() < 0.4:
                candidate_sentence = change_sentence_structure(original_sentence)
                new_words = word_tokenize(candidate_sentence.lower())
                meaning_diff += 1
            
            for j, (word, pos) in enumerate(zip(new_words, pos_tags)):
                if random.random() < 0.7:  # Increased to 70% for more replacements
                    antonyms, synonyms, hypernyms, hyponyms = get_wordnet_candidates(word, pos)
                    candidates = antonyms + hypernyms + hyponyms + synonyms
                    if candidates:
                        new_word = random.choice(candidates)
                        new_words[j] = new_word
                        if new_word in antonyms:
                            meaning_diff += 1
            
            candidate_sentence = ' '.join(new_words).capitalize()
            if is_grammatical(candidate_sentence):
                opt_sentence, opt_similarity, opt_meaning_diff = simulated_annealing(
                    new_words, pos_tags, original_embedding, grammatical=True
                )
                if opt_sentence and opt_similarity < best_gram_similarity:
                    best_grammatical = opt_sentence
                    best_gram_similarity = opt_similarity
                    best_gram_meaning_diff = opt_meaning_diff
            
            # Nonsensical candidate
            vocab = get_wordnet_vocabulary()
            new_words = [random.choice(vocab) for _ in original_words]
            candidate_sentence = ' '.join(new_words).capitalize()
            opt_sentence, opt_similarity, opt_meaning_diff = simulated_annealing(
                new_words, pos_tags, original_embedding, grammatical=False
            )
            if opt_sentence and opt_similarity < best_non_similarity:
                best_nonsensical = opt_sentence
                best_non_similarity = opt_similarity
                best_non_meaning_diff = opt_meaning_diff
        
        return (best_grammatical, best_gram_similarity, best_gram_meaning_diff), \
               (best_nonsensical, best_non_similarity, best_non_meaning_diff)
    except Exception as e:
        logger.error(f"Error in generate_dual_sentences: {e}")
        return (None, float('inf'), 0), (None, float('inf'), 0)

def main():
    while True:
        try:
            original_sentence = input("Enter a sentence (or 'exit' to quit): ")
            if original_sentence.lower() == 'exit':
                print("Exiting program.")
                break
            
            (grammatical_sentence, gram_similarity, gram_meaning_diff), \
            (nonsensical_sentence, non_similarity, non_meaning_diff) = generate_dual_sentences(original_sentence)
            
            print("\nResults:")
            print(f"Original sentence: {original_sentence}")
            print("\nGrammatical Sentence:")
            if grammatical_sentence:
                print(f"Generated: {grammatical_sentence}")
                print(f"Cosine similarity: {1 - gram_similarity:.4f}")
                print(f"Meaning difference score: {gram_meaning_diff}")
            else:
                print("Could not generate a grammatical sentence.")
            
            print("\nNonsensical Sentence:")
            if nonsensical_sentence:
                print(f"Generated: {nonsensical_sentence}")
                print(f"Cosine similarity: {1 - non_similarity:.4f}")
                print(f"Meaning difference score: {non_meaning_diff}")
            else:
                print("Could not generate a nonsensical sentence.")
            print()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            print("An error occurred. Please try again.")
            continue

if __name__ == "__main__":
    main()