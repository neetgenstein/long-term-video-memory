from .graph import MultiPartiteGraph

class ClueIndex:
    def __init__(self):
        self.graph = MultiPartiteGraph()
        self.sentences = []
        self.triplets = []
        self.entity_nodes = []
        self.relation_nodes = []
        self.triplet_nodes = []
        self.sentence_nodes = []

    def build(self, sentences, triplets):
        self.sentences = sentences
        self.triplets = triplets
        self.sentence_nodes = []
        for i, s in enumerate(sentences):
            self.sentence_nodes.append(self.graph.add_sentence(s, sid=i))
        self.triplet_nodes = []
        for (subj, rel, obj) in triplets:
            t = self.graph.add_triplet(subj, rel, obj)
            self.triplet_nodes.append(t)
        # cache entity and relation nodes
        self.entity_nodes = self.graph.nodes_by_type("entity")
        self.relation_nodes = self.graph.nodes_by_type("relation")
        # link sentences to triplets if lexical overlap
        for sn in self.sentence_nodes:
            for tn in self.triplet_nodes:
                self.graph.link_sentence_to_triplet_if_match(sn, tn)
        return self
