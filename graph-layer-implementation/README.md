# ClueRAG (Minimal Implementation)

A minimal, local implementation following the ClueRAG algorithm: **clue extraction → clue-guided beam search → sophisticated re-ranking → answer synthesis**.

## Key Features

✅ **No hardcoded entity anchoring** - uses cosine similarity seeding  
✅ **Clue extraction** with normalization and relation aliasing  
✅ **Beam search** with multi-faceted edge scoring (relation hits, entity hits, lexical overlap, embeddings)  
✅ **Sophisticated re-ranking** with length penalty, clue coverage, chain alignment, type match, MMR diversity  
✅ **Answer synthesis** with question-type-aware heuristics  

## Algorithm Overview

1. **Clue Extraction** (`cluerag/clues.py`)
   - Normalize query text
   - Extract entity clues via token overlap with KG entities
   - Extract relation clues via aliases (e.g., "directed" → directed, "played" → played by)
   - Detect question type (who/what/why/where/when)

2. **Cosine Similarity Seeding** (`cluerag/retrieval.py`)
   - Prefer entity clues as seeds
   - If no entity clues, find entities connected to relation clues
   - Fallback: use TF-IDF cosine similarity to find top-K similar entities
   - No hardcoded anchoring

3. **Beam Search** (clue-guided path expansion)
   - Start from seed entities
   - Expand both outgoing and incoming edges
   - Score each edge with:
     - `w_rel` bonus if relation matches rel_clues
     - `w_ent` bonus if entity matches ent_clues
     - `w_lex` * Jaccard(edge_tokens, query_tokens)
     - `w_emb` * cosine(query, triplet_text)
   - Keep top beam_width paths at each depth
   - Iterate up to max_depth

4. **Re-Ranking** (multi-faceted scoring)
   - Length penalty: `-lambda_len * (path_length - 1)`
   - Relation coverage: `alpha_rel * Jaccard(path_rels, rel_clues)`
   - Entity coverage: `alpha_ent * Jaccard(path_entities, ent_clues)`
   - Chain alignment: `beta_chain * relation_n-gram_overlap`
   - Type match: `gamma_type` bonus for question-type-relevant relations
   - MMR diversity: select top-N paths with `mmr_lambda * score - (1 - mmr_lambda) * max_similarity`

5. **Answer Synthesis**
   - For "who": return subject/object from directed/played by edges
   - For "what": return object from titled/watched edges
   - For "why": return object from reason/to save edges
   - Fallback: return tail of best edge

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Demo

```bash
python demo.py
```

## Example Output

```
Query: Who directed the film Coolie?

Extracted Clues:
  Entity clues: ['film', 'coolie']
  Relation clues: ['directed']
  Question type: who

Top Evidence Paths:
  [1] Score: 8.50
      Path: lokesh kanagaraj -> film -> coolie
      Edges: (lokesh kanagaraj) -[directed]-> (film) | (film) -[titled]-> (coolie)

✓ Answer: Lokesh Kanagaraj
```

## Configuration

Adjust parameters in `demo.py`:

```python
retriever = ClueRAGRetriever(
    index,
    # Edge scoring
    w_rel=2.0,        # Relation clue weight
    w_ent=1.5,        # Entity clue weight
    w_lex=1.0,        # Lexical overlap weight
    w_emb=0.5,        # Embedding similarity weight
    # Beam search
    max_depth=3,      # Max path length
    beam_width=8,     # Beam size
    # Re-ranking
    lambda_len=0.3,   # Length penalty
    alpha_rel=1.5,    # Relation coverage weight
    alpha_ent=1.0,    # Entity coverage weight
    beta_chain=0.8,   # Chain alignment weight
    gamma_type=0.5,   # Type match weight
    mmr_lambda=0.7,   # MMR diversity
    topn=3,           # Top N paths to return
    # Seeding
    seed_top_k=10,    # Top-K entities for similarity seeding
    seed_min_sim=0.05 # Min cosine similarity threshold
)
```

## Structure

- `cluerag/graph.py` — Multi-partite graph with out/in edge indices
- `cluerag/clues.py` — Clue extraction, normalization, aliasing
- `cluerag/retrieval.py` — ClueRAGRetriever (beam search, re-ranking, MMR, answer)
- `cluerag/index.py` — Index builder
- `cluerag/utils.py` — TF-IDF text index
- `demo.py` — Example queries

## Differences from Full ClueRAG

- **Lightweight**: TF-IDF instead of heavy sentence transformers (you can add ST if needed)
- **No LLM extraction**: assumes triplets are pre-extracted
- **Local-first**: no external APIs required
- **Training-free**: all scoring is heuristic-based
- **Explainable**: shows extracted clues, paths, and reasoning
