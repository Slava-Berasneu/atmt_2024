import torch

from itertools import count
from queue import PriorityQueue


class BeamSearch(object):
    def __init__(self, beam_size, max_len, pad):
        self.beam_size = beam_size
        self.max_len = max_len
        self.pad = pad

        self.nodes = PriorityQueue() # beams to be expanded
        self.final = PriorityQueue() # beams that ended in EOS

        self._counter = count() # for correct ordering of nodes with same score

    def add(self, score, node):
        self.nodes.put((score, next(self._counter), node))

    def add_final(self, score, node):
        missing = self.max_len - node.length
        node.sequence = torch.cat((node.sequence.cpu(), torch.tensor([self.pad]*missing).long()))
        self.final.put((score, next(self._counter), node))

    def get_current_beams(self):
        nodes = []
        while not self.nodes.empty() and len(nodes) < self.beam_size:
            node = self.nodes.get()
            nodes.append((node[0], node[2]))
        return nodes

    def get_best(self):
        # Merge EOS and unfinished
        merged = PriorityQueue()
        for _ in range(self.final.qsize()):
            f = self.final.get()
            merged.put(f)
        for _ in range(self.nodes.qsize()):
            n = self.nodes.get()
            merged.put(n)
        node = merged.get()
        node = (node[0], node[2])
        return node

    def best_final_score(self):
        """Return the best final hypothesis eval score if exists, None otherwise."""
        if self.final.empty():
            return None
        # Peek at best final without removing permanently:
        finals = []
        while not self.final.empty():
            item = self.final.get()
            finals.append(item)
        # best final is the one with smallest score (since we store negative)
        best = finals[0]  # after re-inserting, order is preserved
        for f in finals[1:]:
            if f[0] < best[0]:
                best = f
        # Put them back
        for f in finals:
            self.final.put(f)
        # best is (score, counter, node), score = -node.eval(alpha)
        best_node_eval = -best[0]
        return best_node_eval

    def prune(self):
        """
        Keep beam_size best nodes and prune out any incomplete hypotheses that
        score worse than the best finished hypothesis (if any).
        """
        # If we have a best final hypothesis, get its eval score
        best_final = self.best_final_score()

        kept = []
        # First, gather all nodes
        while not self.nodes.empty():
            n = self.nodes.get()
            kept.append(n)

        # Filter out nodes worse than best_final (if best_final is available)
        if best_final is not None:
            new_kept = []
            for (score, c, node) in kept:
                # score = -node.eval(alpha)
                node_eval = -score
                # Prune if node_eval < best_final
                if node_eval >= best_final:
                    new_kept.append((score, c, node))
            kept = new_kept

        # Keep only beam_size - already finished_count
        finished_count = self.final.qsize()
        max_to_keep = max(self.beam_size - finished_count, 0)
        kept = sorted(kept, key=lambda x: x[0])
        kept = kept[:max_to_keep]

        self.nodes = PriorityQueue()
        for k in kept:
            self.nodes.put(k)

    def is_done(self):
        # If no unfinished nodes left and we can't expand more, we are done
        return self.nodes.empty()



class BeamSearchNode(object):
    """ Defines a search node and stores values important for computation of beam search path"""
    def __init__(self, search, emb, lstm_out, final_hidden, final_cell, mask, sequence, logProb, length):

        # Attributes needed for computation of decoder states
        self.sequence = sequence
        self.emb = emb
        self.lstm_out = lstm_out
        self.final_hidden = final_hidden
        self.final_cell = final_cell
        self.mask = mask

        # Attributes needed for computation of sequence score
        self.logp = logProb
        self.length = length

        self.search = search

    def eval(self, alpha=0.0):
        """ Returns score of sequence up to this node 

        params: 
            :alpha float (default=0.0): hyperparameter for
            length normalization described in in
            https://arxiv.org/pdf/1609.08144.pdf (equation
            14 as lp), default setting of 0.0 has no effect
        
        """
        normalizer = (5 + self.length)**alpha / (5 + 1)**alpha
        return self.logp / normalizer
        