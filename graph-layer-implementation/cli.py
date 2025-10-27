#!/usr/bin/env python3
"""ClueRAG CLI - Command-line interface for ClueRAG retrieval."""
import json
import argparse
from typing import List, Tuple

from cluerag.index import ClueIndex
from cluerag.retrieval import ClueRAGRetriever
from cluerag.embed import TextEmbedder
from cluerag.llm import OpenAIProvider, OllamaProvider, GeminiProvider, MockProvider


def load_triplets(path: str) -> List[Tuple[str, str, str]]:
    """Load triplets from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return [(t[0], t[1], t[2]) for t in data]


def synthesize_answer_with_llm(result, query, llm):
    """Use LLM to generate answer from evidence paths."""
    paths = result["paths"]
    if not paths:
        return "No answer found."
    
    # Format top-3 paths as evidence
    evidence_lines = []
    for i, (score, path, edges) in enumerate(paths[:3], 1):
        edge_reprs = " | ".join([f"({h})-[{r}]->({t})" for h, r, t in edges])
        evidence_lines.append(f"{i}. {edge_reprs}")
    
    evidence_text = "\n".join(evidence_lines)
    
    prompt = f"""Given the following knowledge graph evidence paths, answer the question concisely in one line.

Question: {query}

Evidence paths:
{evidence_text}

Answer (one line only):"""
    
    try:
        answer = llm.generate(prompt)
        return answer
    except Exception as e:
        print(f"⚠ LLM error: {e}")
        return "LLM error, using heuristic fallback."


def main():
    parser = argparse.ArgumentParser(description="ClueRAG CLI - Knowledge Graph Retrieval")
    parser.add_argument("--query", type=str, required=True, help="Natural language query")
    parser.add_argument("--data", type=str, default="data/triplets.json", help="Path to triplets JSON file")
    
    # Retrieval parameters
    parser.add_argument("--w_rel", type=float, default=3.0, help="Relation clue weight")
    parser.add_argument("--w_ent", type=float, default=1.5, help="Entity clue weight")
    parser.add_argument("--w_lex", type=float, default=1.0, help="Lexical overlap weight")
    parser.add_argument("--w_emb", type=float, default=0.5, help="Embedding similarity weight")
    
    parser.add_argument("--depth", type=int, default=3, help="Max beam search depth")
    parser.add_argument("--beam", type=int, default=8, help="Beam width")
    
    parser.add_argument("--lambda_len", type=float, default=0.5, help="Length penalty")
    parser.add_argument("--alpha_rel", type=float, default=3.0, help="Relation coverage weight")
    parser.add_argument("--alpha_ent", type=float, default=1.5, help="Entity coverage weight")
    parser.add_argument("--beta_chain", type=float, default=1.5, help="Chain alignment weight")
    parser.add_argument("--gamma_type", type=float, default=0.5, help="Type match weight")
    parser.add_argument("--mmr_lambda", type=float, default=0.7, help="MMR diversity lambda")
    
    parser.add_argument("--topn", type=int, default=3, help="Top N paths to return")
    parser.add_argument("--seed_top_k", type=int, default=10, help="Top-K entities for seeding")
    parser.add_argument("--seed_min_sim", type=float, default=0.05, help="Min similarity for seeding")
    
    # Embeddings
    parser.add_argument("--embeddings", action="store_true", help="Enable embedding-based scoring")
    parser.add_argument("--emb_model", type=str, default="", help="Sentence-Transformers model name (optional)")
    
    # LLM
    parser.add_argument("--llm", type=str, default="none", 
                        choices=["none", "openai", "ollama", "gemini", "mock"],
                        help="LLM provider for answer synthesis")
    parser.add_argument("--llm_model", type=str, default="", help="LLM model name")
    parser.add_argument("--ollama_host", type=str, default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--gemini_key", type=str, default="", help="Gemini API key (or use GOOGLE_API_KEY env)")
    parser.add_argument("--openai_key", type=str, default="", help="OpenAI API key (or use OPENAI_API_KEY env)")
    
    # Output options
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output (clues, paths, heuristic fallback)")
    
    args = parser.parse_args()
    
    # Load data
    print(f"📂 Loading triplets from {args.data}...")
    triplets = load_triplets(args.data)
    print(f"✓ Loaded {len(triplets)} triplets")
    
    # Build index
    print("🔨 Building knowledge graph index...")
    # Create dummy sentences (not used in retrieval but needed for index)
    sentences = []
    index = ClueIndex().build(sentences, triplets)
    print(f"✓ Index built: {len(index.entity_nodes)} entities, {len(index.relation_nodes)} relations")
    
    # Setup embedder (optional)
    embedder = None
    if args.embeddings:
        print("🔤 Initializing embeddings...")
        model_name = args.emb_model if args.emb_model else None
        embedder = TextEmbedder(model_name=model_name)
        # Fit BoW if using TF-IDF
        if embedder.bow_vectorizer is not None:
            corpus = [f"{h} {r} {t}" for h, r, t in triplets]
            embedder.fit_bow(corpus + [args.query])
    
    # Create retriever
    print("🔍 Initializing ClueRAG retriever...")
    retriever = ClueRAGRetriever(
        index,
        w_rel=args.w_rel,
        w_ent=args.w_ent,
        w_lex=args.w_lex,
        w_emb=args.w_emb if args.embeddings else 0.0,
        max_depth=args.depth,
        beam_width=args.beam,
        lambda_len=args.lambda_len,
        alpha_rel=args.alpha_rel,
        alpha_ent=args.alpha_ent,
        beta_chain=args.beta_chain,
        gamma_type=args.gamma_type,
        mmr_lambda=args.mmr_lambda,
        topn=args.topn,
        seed_top_k=args.seed_top_k,
        seed_min_sim=args.seed_min_sim
    )
    
    # Setup LLM (optional)
    llm = None
    if args.llm != "none":
        print(f"🤖 Initializing LLM provider: {args.llm}...")
        try:
            if args.llm == "openai":
                model = args.llm_model if args.llm_model else "gpt-4o-mini"
                api_key = args.openai_key if args.openai_key else None
                llm = OpenAIProvider(model=model, api_key=api_key)
            elif args.llm == "ollama":
                model = args.llm_model if args.llm_model else "llama3.2"
                llm = OllamaProvider(model=model, host=args.ollama_host)
            elif args.llm == "gemini":
                model = args.llm_model if args.llm_model else "gemini-2.0-flash-exp"
                api_key = args.gemini_key if args.gemini_key else None
                llm = GeminiProvider(model=model, api_key=api_key)
            elif args.llm == "mock":
                llm = MockProvider()
        except Exception as e:
            print(f"⚠ Failed to initialize LLM: {e}")
            print("→ Falling back to heuristic answer synthesis")
            llm = None
    
    # Retrieve
    print(f"\n{'='*70}")
    print(f"Query: {args.query}")
    print(f"{'='*70}\n")
    
    result = retriever.retrieve(args.query)
    
    # Show clues (if verbose)
    if args.verbose:
        print("📌 Extracted Clues:")
        ent_clues_text = [retriever.index.graph.node_text(e) for e in result['ent_clues']]
        print(f"  Entity clues: {ent_clues_text}")
        print(f"  Relation clues: {list(result['rel_clues'])}")
        print(f"  Question type: {result['q_type']}\n")
    
    # Show evidence paths
    print("📊 Top Evidence Paths:")
    for i, (score, path, edges) in enumerate(result["paths"], 1):
        path_repr = " → ".join([retriever.index.graph.node_text(n) for n in path])
        print(f"\n  [{i}] Score: {score:.2f}")
        if args.verbose:
            print(f"      Path: {path_repr}")
        edge_reprs = [f"({retriever.index.graph.node_text(h)}) -[{r}]-> ({retriever.index.graph.node_text(t)})" 
                     for h, r, t in edges]
        print(f"      Edges: {' | '.join(edge_reprs)}")
    
    # Generate answer
    print(f"\n{'='*70}")
    if llm is not None:
        print("🤖 Answer (LLM):")
        answer = synthesize_answer_with_llm(result, args.query, llm)
        print(f"  {answer}")
        if args.verbose:
            print("\n💡  Answer (fallback):")
            heuristic_ans = retriever.answer_heuristic(args.query)
            print(f"  {heuristic_ans}")
    else:
        print("💡 Answer :")
        answer = retriever.answer_heuristic(args.query)
        print(f"  {answer}")
        if args.verbose:
            print("\n  ℹ️  Tip: Use --llm gemini for better answer synthesis")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
