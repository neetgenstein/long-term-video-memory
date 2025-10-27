"""Clue extraction: normalize query and extract entity/relation clues."""

def norm(text):
    """Normalize text: lowercase, strip, collapse spaces."""
    return " ".join(text.lower().strip().split())

# Relation aliases - map surface forms to canonical relations
RELATION_ALIASES = {
   
}

def detect_question_type(query):
    """Simple WH-question type detection."""
    ql = query.lower()
    if any(w in ql for w in ["who", "whom", "whose"]):
        return "who"
    if "why" in ql:
        return "why"
    if "what" in ql:
        return "what"
    if "where" in ql:
        return "where"
    if "when" in ql:
        return "when"
    return "what"

def extract_clues(query, index):
    """
    Extract entity and relation clues from query.
    
    Returns:
        ent_clues: set of entity nodes
        rel_clues: set of normalized relation names
        q_type: question type
    """
    q_norm = norm(query)
    q_tokens = set(q_norm.split())
    q_type = detect_question_type(query)
    
    # Extract relation clues via aliases
    rel_clues = set()
    for alias, canonical in RELATION_ALIASES.items():
        if alias in q_norm:
            # Find matching relations in graph
            for rn in index.relation_nodes:
                r_text = norm(index.graph.node_text(rn))
                if canonical in r_text or r_text in canonical:
                    rel_clues.add(r_text)
    
    # Extract entity clues via substring matches
    # Per ClueRAG algorithm: fuzzy lookup from query to KG entities
    ent_clues = set()
    
    # Remove stop words from query tokens for better matching
    stop_words = {"the", "a", "an", "of", "to", "from", "in", "on", "at", "by", "for", 
                  "with", "did", "do", "does", "is", "was", "were", "are", "be", "been"}
    q_tokens_filtered = q_tokens - stop_words
    
    for en in index.entity_nodes:
        e_text = norm(index.graph.node_text(en))
        e_tokens = set(e_text.split())
        
        # Strategy 1: Entity text is substring of query (exact phrase)
        if len(e_text) >= 3 and e_text in q_norm:
            ent_clues.add(en)
            continue
        
        # Strategy 2: Significant token overlap (multi-word entities)
        overlap = e_tokens & q_tokens_filtered
        if overlap:
            # Require at least 2 overlapping tokens OR 1 long token (4+ chars)
            if len(overlap) >= 2:
                ent_clues.add(en)
            elif any(len(t) >= 4 for t in overlap):
                ent_clues.add(en)
    
    return ent_clues, rel_clues, q_type

def jaccard(a, b):
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
