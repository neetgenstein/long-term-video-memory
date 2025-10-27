"""ClueRAG-inspired retrieval: clue-guided beam search with re-ranking."""
import heapq
from collections import defaultdict
from .clues import extract_clues, jaccard, norm
from .utils import MultiTextIndex

class ClueRAGRetriever:
    def __init__(self, index, 
                 w_rel=3.0, w_ent=1.5, w_lex=1.0, w_emb=0.5,
                 max_depth=3, beam_width=8,
                 lambda_len=0.5, alpha_rel=3.0, alpha_ent=1.5, beta_chain=1.5, gamma_type=0.5,
                 mmr_lambda=0.7, topn=5, seed_top_k=10, seed_min_sim=0.05):
        self.index = index
        # Edge scoring weights
        self.w_rel = w_rel
        self.w_ent = w_ent
        self.w_lex = w_lex
        self.w_emb = w_emb
        # Beam search params
        self.max_depth = max_depth
        self.beam_width = beam_width
        # Re-ranking weights
        self.lambda_len = lambda_len
        self.alpha_rel = alpha_rel
        self.alpha_ent = alpha_ent
        self.beta_chain = beta_chain
        self.gamma_type = gamma_type
        self.mmr_lambda = mmr_lambda
        self.topn = topn
        # Seeding params
        self.seed_top_k = seed_top_k
        self.seed_min_sim = seed_min_sim

        g = self.index.graph
        self.entity_nodes = self.index.entity_nodes
        self.relation_nodes = self.index.relation_nodes
        self.triplet_nodes = self.index.triplet_nodes

        texts_by_type = {
            "entity": [g.node_text(n) for n in self.entity_nodes],
            "relation": [g.node_text(n) for n in self.relation_nodes],
            "triplet": [g.node_text(n) for n in self.triplet_nodes],
        }
        self.multi_index = MultiTextIndex(texts_by_type)

    def seed_nodes(self, query, ent_clues, rel_clues):
        """Seed nodes for beam search using clues and cosine similarity."""
        seeds = []
        q_norm = norm(query)
        
        # Strategy 1: Use entity clues with relevance scoring
        if ent_clues:
            # Score each entity clue by how well it matches the query
            scored_ents = []
            for en in ent_clues:
                e_text = norm(self.index.graph.node_text(en))
                # Exact match gets highest score
                if e_text in q_norm:
                    scored_ents.append((en, 10.0))
                # Partial match
                elif any(token in q_norm for token in e_text.split()):
                    scored_ents.append((en, 5.0))
                else:
                    scored_ents.append((en, 1.0))
            # Sort by score and take top seeds
            scored_ents.sort(key=lambda x: -x[1])
            seeds = [en for en, _ in scored_ents[:self.seed_top_k]]
        
        # Strategy 2: If no entity clues, use relation clues
        if not seeds and rel_clues:
            for en in self.entity_nodes:
                for rel, tgt, _ in self.index.graph.get_outgoing(en):
                    if any(rc in norm(rel) for rc in rel_clues):
                        seeds.append(en)
                        break
                for rel, src, _ in self.index.graph.get_incoming(en):
                    if any(rc in norm(rel) for rc in rel_clues):
                        seeds.append(en)
                        break
            seeds = list(set(seeds))[:self.seed_top_k]
        
        # Strategy 3: Use cosine similarity
        if not seeds:
            sims = self.multi_index.sim("entity", query)
            if sims is not None:
                # Get top-K by similarity
                top_indices = sorted(range(len(sims)), key=lambda i: -sims[i])[:self.seed_top_k]
                seeds = [self.entity_nodes[i] for i in top_indices if sims[i] >= self.seed_min_sim]
        
        # Fallback: all entities if KG is small
        if not seeds and len(self.entity_nodes) <= 20:
            seeds = list(self.entity_nodes)[:self.seed_top_k]
        
        return seeds

    def edge_score(self, query, rel, head, tail, ent_clues, rel_clues, q_tokens):
        """Score an edge for beam search expansion."""
        score = 0.0
        rel_norm = norm(rel)
        # Relation clue hit
        if any(rc in rel_norm for rc in rel_clues):
            score += self.w_rel
        # Entity clue hits
        if head in ent_clues:
            score += self.w_ent
        if tail in ent_clues:
            score += 0.8 * self.w_ent
        # Lexical overlap (Jaccard)
        head_text = norm(self.index.graph.node_text(head))
        tail_text = norm(self.index.graph.node_text(tail))
        edge_text = f"{head_text} {rel_norm} {tail_text}"
        edge_tokens = set(edge_text.split())
        lex_sim = jaccard(edge_tokens, q_tokens)
        score += self.w_lex * lex_sim
        # Embedding similarity
        if self.w_emb > 0:
            triplet_text = f"{head_text} {rel_norm} {tail_text}"
            sims = self.multi_index.sim("triplet", query)
            if sims is not None:
                # Find matching triplet node
                for i, tn in enumerate(self.triplet_nodes):
                    if self.index.graph.triplet_text(tn).lower() == triplet_text:
                        score += self.w_emb * float(sims[i])
                        break
        return score

    def beam_search(self, query, seeds, ent_clues, rel_clues):
        """Beam search to collect candidate paths."""
        q_tokens = set(norm(query).split())
        # Path: (score, [node1, node2, ...], {edges_as_tuples})
        beams = [(0.0, [seed], set()) for seed in seeds]
        
        for depth in range(self.max_depth):
            candidates = []
            for score, path, edges in beams:
                curr = path[-1]
                # Expand outgoing
                for rel, tgt, triplet_node in self.index.graph.get_outgoing(curr):
                    edge_repr = (curr, rel, tgt)
                    if edge_repr in edges:
                        continue
                    escore = self.edge_score(query, rel, curr, tgt, ent_clues, rel_clues, q_tokens)
                    new_edges = edges | {edge_repr}
                    candidates.append((score + escore, path + [tgt], new_edges))
                # Expand incoming
                for rel, src, triplet_node in self.index.graph.get_incoming(curr):
                    edge_repr = (src, rel, curr)
                    if edge_repr in edges:
                        continue
                    escore = self.edge_score(query, rel, src, curr, ent_clues, rel_clues, q_tokens)
                    new_edges = edges | {edge_repr}
                    candidates.append((score + escore, path + [src], new_edges))
            
            # Keep top beam_width
            candidates.sort(key=lambda x: -x[0])
            beams = candidates[:self.beam_width]
            if not beams:
                break
        
        return beams

    def rerank_paths(self, query, paths, ent_clues, rel_clues, q_type):
        """Re-rank paths with coverage, brevity, chain alignment, type fit."""
        if not paths:
            return []
        
        scored = []
        for score, path, edges in paths:
            # Extract relations and entities from path
            path_rels = set()
            path_ents = set(path)
            for h, r, t in edges:
                path_rels.add(norm(r))
            
            # Length penalty
            len_penalty = -self.lambda_len * (len(path) - 1)
            
            # Clue coverage
            rel_coverage = jaccard(path_rels, rel_clues) if rel_clues else 0.0
            ent_coverage = jaccard(path_ents, ent_clues) if ent_clues else 0.0
            
            # Chain alignment (relation n-gram overlap)
            chain_score = 0.0
            for r in path_rels:
                for rc in rel_clues:
                    if rc in r or r in rc:
                        chain_score += 1.0
            chain_score = chain_score / max(len(rel_clues), 1) if rel_clues else 0.0
            
            # Type match
            type_score = 0.0
            if q_type == "who":
                if any(r in ["directed", "played by", "played the role of"] for r in path_rels):
                    type_score = 1.0
            elif q_type == "why":
                if any("came out" in r or "to save" in r for r in path_rels):
                    type_score = 1.0
            
            final_score = (
                score + len_penalty +
                self.alpha_rel * rel_coverage +
                self.alpha_ent * ent_coverage +
                self.beta_chain * chain_score +
                self.gamma_type * type_score
            )
            
            scored.append((final_score, path, edges))
        
        scored.sort(key=lambda x: -x[0])
        return scored

    def mmr_select(self, scored_paths, topn):
        """MMR-based diversity selection."""
        if len(scored_paths) <= topn:
            return scored_paths
        
        selected = []
        remaining = list(scored_paths)
        
        while len(selected) < topn and remaining:
            best_idx = 0
            best_mmr = float("-inf")
            
            for i, (score, path, edges) in enumerate(remaining):
                # Max similarity to selected
                max_sim = 0.0
                if selected:
                    for sel_score, sel_path, sel_edges in selected:
                        # Similarity: end-node equality or edge Jaccard
                        if path[-1] == sel_path[-1]:
                            sim = 1.0
                        else:
                            sim = jaccard(edges, sel_edges)
                        max_sim = max(max_sim, sim)
                
                mmr_score = self.mmr_lambda * score - (1 - self.mmr_lambda) * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected

    def retrieve(self, query):
        """Full retrieval pipeline."""
        # Extract clues
        ent_clues, rel_clues, q_type = extract_clues(query, self.index)
        
        # Seed nodes
        seeds = self.seed_nodes(query, ent_clues, rel_clues)
        if not seeds:
            return {"paths": [], "ent_clues": set(), "rel_clues": set(), "q_type": q_type}
        
        # Beam search
        paths = self.beam_search(query, seeds, ent_clues, rel_clues)
        
        # Re-rank
        ranked = self.rerank_paths(query, paths, ent_clues, rel_clues, q_type)
        
        # MMR diversity
        final = self.mmr_select(ranked, self.topn)
        
        return {
            "paths": final,
            "ent_clues": ent_clues,
            "rel_clues": rel_clues,
            "q_type": q_type
        }

    def answer_heuristic(self, query):
        """Heuristic answer extraction (fallback when no LLM)."""
        result = self.retrieve(query)
        paths = result["paths"]
        q_type = result["q_type"]
        ent_clues = result["ent_clues"]
        rel_clues = result["rel_clues"]
        
        if not paths:
            return "No answer found."
        
        top_path = paths[0]
        score, path, edges = top_path
        
        if not edges:
            return self.index.graph.node_text(path[-1]) if path else "No answer found."
        
        q_norm = norm(query)
        
        # Collect all entities from all edges in top path
        all_entities = set()
        for h, r, t in edges:
            all_entities.add(h)
            all_entities.add(t)
        
        # For WHY: return tail of edge matching relation clue
        if q_type == "why":
            for h, r, t in edges:
                if rel_clues:
                    for rc in rel_clues:
                        if rc in norm(r):
                            return self.index.graph.node_text(t)
                # Fallback: first tail
                return self.index.graph.node_text(t)
        
        # For WHAT: check if asking about attribute of an entity
        if q_type == "what":
            # Find entities in path that overlap with query
            for e in all_entities:
                e_text = norm(self.index.graph.node_text(e))
                e_tokens = set(e_text.split())
                q_tokens = set(q_norm.split())
                
                # If entity contains query tokens, it might have the answer
                overlap = e_tokens & q_tokens
                if overlap and len(e_tokens) > len(overlap):
                    # Entity has extra words beyond query - likely attributes
                    # Return the full entity text
                    return self.index.graph.node_text(e)
            
            # No attribute match, return tail of best edge
            for h, r, t in edges:
                return self.index.graph.node_text(t)
        
        # For WHO: return entity NOT in clues
        if q_type == "who":
            # First try: find edge with relation match
            for h, r, t in edges:
                if rel_clues:
                    for rc in rel_clues:
                        if rc in norm(r):
                            # Relation matches, determine answer
                            if "by" in norm(r):
                                return self.index.graph.node_text(t)
                            if h in ent_clues:
                                return self.index.graph.node_text(t)
                            return self.index.graph.node_text(h)
            
            # No relation match: return entity not in query
            for h, r, t in edges:
                if h not in ent_clues and h in all_entities:
                    return self.index.graph.node_text(h)
                if t not in ent_clues and t in all_entities:
                    return self.index.graph.node_text(t)
                # Fallback for passive voice
                if "by" in norm(r):
                    return self.index.graph.node_text(t)
                return self.index.graph.node_text(h)
        
        # Default: return tail of first edge
        return self.index.graph.node_text(edges[0][2])
