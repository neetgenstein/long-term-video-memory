import networkx as nx

class MultiPartiteGraph:
    def __init__(self):
        self.g = nx.DiGraph()
        self.out_edges = {}  # node -> list of (relation, target) outgoing
        self.in_edges = {}   # node -> list of (relation, source) incoming

    def add_entity(self, name):
        if not self.g.has_node(("entity", name)):
            self.g.add_node(("entity", name), type="entity", label=name)
        return ("entity", name)

    def add_relation(self, name):
        if not self.g.has_node(("relation", name)):
            self.g.add_node(("relation", name), type="relation", label=name)
        return ("relation", name)

    def add_triplet(self, subj, rel, obj):
        t = ("triplet", subj, rel, obj)
        if not self.g.has_node(t):
            self.g.add_node(t, type="triplet", subj=subj, rel=rel, obj=obj)
        s_node = self.add_entity(subj)
        r_node = self.add_relation(rel)
        o_node = self.add_entity(obj)
        
        # Build bidirectional edges
        self.g.add_edge(t, s_node)
        self.g.add_edge(t, r_node)
        self.g.add_edge(t, o_node)
        self.g.add_edge(s_node, t)
        self.g.add_edge(r_node, t)
        self.g.add_edge(o_node, t)
        
        # Track directed edges for beam search: subj -[rel]-> obj
        if s_node not in self.out_edges:
            self.out_edges[s_node] = []
        if o_node not in self.in_edges:
            self.in_edges[o_node] = []
        self.out_edges[s_node].append((rel, o_node, t))
        self.in_edges[o_node].append((rel, s_node, t))
        
        return t

    def add_sentence(self, text, sid=None):
        node = ("sentence", sid if sid is not None else text)
        if not self.g.has_node(node):
            self.g.add_node(node, type="sentence", text=text)
        return node

    def link_sentence_to_triplet_if_match(self, sentence_node, triplet_node):
        text = self.g.nodes[sentence_node].get("text", "").lower()
        subj = triplet_node[1].lower()
        rel = triplet_node[2].lower()
        obj = triplet_node[3].lower()
        if subj in text or obj in text or rel in text:
            self.g.add_edge(sentence_node, triplet_node)
            self.g.add_edge(triplet_node, sentence_node)

    def nodes_by_type(self, t):
        return [n for n, d in self.g.nodes(data=True) if d.get("type") == t]

    def neighbors(self, node):
        return list(self.g.neighbors(node))
    
    def get_outgoing(self, node):
        """Get outgoing edges: [(relation, target, triplet_node)]"""
        return self.out_edges.get(node, [])
    
    def get_incoming(self, node):
        """Get incoming edges: [(relation, source, triplet_node)]"""
        return self.in_edges.get(node, [])

    def node_type(self, node):
        return self.g.nodes[node].get("type")

    def triplet_text(self, node):
        d = self.g.nodes[node]
        return f"{d.get('subj')} {d.get('rel')} {d.get('obj')}"

    def node_text(self, node):
        t = self.node_type(node)
        if t == "entity":
            return self.g.nodes[node].get("label", "")
        if t == "relation":
            return self.g.nodes[node].get("label", "")
        if t == "triplet":
            return self.triplet_text(node)
        if t == "sentence":
            return self.g.nodes[node].get("text", "")
        return ""
