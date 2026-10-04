"""
Monte Carlo Tree Search for the chess engine.

Three conventions hold everywhere in this file, and every signature below depends on them:

1. VALUE PERSPECTIVE
   Every value is seen by the side to move AT THAT NODE. +1 means the player about to move is
   winning, -1 means they are lost. The side to move alternates every ply, so _backup flips the
   sign once per step back up the tree.

2. EDGE STATISTICS
   N, W and priors live on the PARENT, indexed by move index k (the same k indexes node.moves).
   A child Node object is allocated only the first time that edge is followed, so a node that is
   visited once costs one array triple and one mostly empty children list.

3. ONE BOARD, WALKED IN PLACE
   The search mutates a single ChessGame with play_move / undo_move and restores it to the root
   position at the end of every simulation. No board is ever copied.
"""

from dataclasses import dataclass

import chess
import numpy as np

from board import ChessGame

POLICY_SIZE = 4672   # 64 from-squares * 73 move types, the index space of ChessGame.hash_move


@dataclass
class MCTSConfig:
    """
    Every knob the search takes. Pure parameter bag, no logic lives here.

    num_simulations     simulations run by one search() call
    c_puct              exploration constant in the PUCT formula
    dirichlet_alpha     concentration of the root noise (0.3 for chess)
    dirichlet_frac      how much of the root prior the noise replaces
    temperature         visit count temperature used while sampling self-play moves
    temperature_moves   number of opening plies played at `temperature`, argmax afterwards
    fpu                 first play urgency: the Q given to an edge with N == 0
    add_root_noise      on during self play, off when the engine is actually playing
    """

    num_simulations: int = 800
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    temperature: float = 1.0
    temperature_moves: int = 30
    fpu: float = 0.0
    add_root_noise: bool = True


"""
MonteCarlo Node:
one position in the tree. It owns the statistics of the edges going OUT of it (convention 2),
so the root's N array is exactly the visit distribution the search returns.
"""

class Node:
    __slots__ = ("moves", "priors", "N", "W", "children", "terminal_value")

    def __init__(self):
        self.moves = None            # list[chess.Move], filled in on expansion
        self.priors = None           # np.float32[k]: policy logits at the legal-move hashes, softmaxed
        self.N = None                # np.int32[k]: visit count per move
        self.W = None                # np.float32[k]: total value per move
        self.children = None         # list[Node | None], a child is created only when first visited
        self.terminal_value = None   # set if the game is over at this node

    @property
    def expanded(self):
        return self.moves is not None

    @property
    def is_terminal(self):
        """
        True when the game is over at this node.

        Tested against None, never truthiness: 0.0 is a real terminal value (a draw). A terminal
        node is never expanded, so `expanded` and `is_terminal` are two separate questions that
        selection has to ask separately.
        """
        return self.terminal_value is not None

    def expand(self, moves, priors):
        """
        Turn a freshly created node into an expanded one.

        Stores the legal move list and its priors, and allocates the three per-edge arrays
        (N int32, W float32, children filled with None), all of length len(moves).
        A terminal node never goes through here: it only gets terminal_value set.
        """
        k = len(moves)
        self.moves = moves
        self.priors = np.asarray(priors, dtype=np.float32)
        self.N = np.zeros(k, dtype=np.int32)
        self.W = np.zeros(k, dtype=np.float32)
        self.children = [None] * k

    def q(self, fpu=0.0):
        """
        np.float32[k]: the mean value of each edge, W/N, with config.fpu substituted wherever
        N == 0. Vectorised, the divide by zero avoided with np.where rather than a loop.
        """
        return np.where(self.N > 0, self.W / np.maximum(self.N, 1), fpu).astype(np.float32)

    def puct_scores(self, c_puct, fpu=0.0):
        """
        np.float32[k]: q() + c_puct * priors * sqrt(N.sum()) / (1 + N).

        The exploration term is large for an edge with a high prior and few visits, and decays as
        that edge gets visited, which is what makes the search concentrate on the promising moves
        without ever fully abandoning the rest.

        sqrt is taken of max(N.sum(), 1): with the bare sum the first visit out of a node has an
        exploration term of zero everywhere, and it would go to move 0 instead of the best prior.
        """
        total = max(int(self.N.sum()), 1)
        return self.q(fpu) + c_puct * self.priors * np.sqrt(total) / (1 + self.N)

    def best_child_index(self, c_puct, fpu=0.0):
        """
        int: argmax of puct_scores, the move index selection should follow from this node.
        Ties broken by the lowest index.
        """
        return int(np.argmax(self.puct_scores(c_puct, fpu)))   # np.argmax returns the first max


class Evaluator:
    """
    The only place the search touches a network. Implemented for real once the net exists, so
    that MCTS can be written and tested before then.
    """

    def evaluate(self, game):
        """
        (policy_logits, value) for the position `game` is currently in.

        policy_logits   np.float32[POLICY_SIZE], indexed exactly like ChessGame.hash_move
        value           float in [-1, 1], from the side to move's perspective

        hash_move canonicalises by side to move (it mirrors the squares for black), so this index
        space is already in the current player's view: the search and the training targets line
        up as long as both go through hash_move.
        """
        raise NotImplementedError

    def evaluate_batch(self, games):
        """
        list[(policy_logits, value)], one per game, in order. The default just loops; an
        evaluator backed by a net overrides it to run all the positions in one forward pass.
        """
        return [self.evaluate(g) for g in games]


class UniformEvaluator(Evaluator):
    """
    Stand in evaluator: zero logits (so the priors come out uniform over the legal moves) and a
    value of 0.0. Lets selection, backup and the undo bookkeeping be tested with no network.
    """

    def evaluate(self, game):
        return np.zeros(POLICY_SIZE, dtype=np.float32), 0.0


class BatchedEvaluator(Evaluator):
    """
    Same interface, but the real work happens in evaluate_batch, which a subclass implements to
    push many positions through the net at once (see network.NetEvaluator). evaluate() is then
    just a batch of one.

    The search never blocks inside the evaluator to wait for a batch to fill: instead a search
    is split in two at the leaf (prepare_leaf / complete_leaf), and search_many() drives many
    trees side by side, gathering one leaf from each and evaluating them together.
    """

    def evaluate(self, game):
        return self.evaluate_batch([game])[0]

    def evaluate_batch(self, games):
        raise NotImplementedError


"""
Implementation of MonteCarlo Tree Search:
Purpose:
- navigate only the most promising game states
- backpropagate how promising each state is every time (through priors)

One simulation is cut in two at the leaf so that many trees can share one network call:
    prepare_leaf()   select a leaf; resolves terminal leaves on its own, otherwise leaves the
                     board AT the leaf and returns True ("evaluate game for me")
    complete_leaf()  takes the network output for that leaf, expands, backs up, undoes to root
_simulate() is these two with a single evaluate() call in between.
"""


class MonteCarloTS:
    def __init__(self, game: ChessGame, evaluator: Evaluator, config: MCTSConfig | None = None,
                 seed=None):
        """
        `game` is the live ChessGame the search walks: it is mutated and restored, not copied, so
        the caller must not touch it while a search is running.

        Holds the game, the evaluator, the config, an np.random.Generator for the root noise and
        the move sampling, and an empty root Node.
        """
        self.game = game
        self.evaluator = evaluator
        self.config = config if config is not None else MCTSConfig()
        self.rng = np.random.default_rng(seed)
        self.root = Node()
        self._noised_root = None   # the root that already got its Dirichlet noise
        self._pending = None       # (node, path, moves) between prepare_leaf and complete_leaf

    # ---- main loop -------------------------------------------------------------------

    def search(self, num_simulations=None):
        """
        Run the search from the current root and return the root Node.

        Expands and evaluates the root when that has not happened yet, mixes Dirichlet noise into
        root.priors once when config.add_root_noise is set, then runs num_simulations
        simulations. Checks on the way out that the board came back to the root position
        (comparing the move stack depth before and after catches most search bugs early).
        """
        n = self.config.num_simulations if num_simulations is None else num_simulations
        depth = len(self.game.board.move_stack)

        if not self.root.expanded and not self.root.is_terminal:
            self._simulate()   # path is empty, so this only expands the root
        if not self.root.is_terminal:
            for _ in range(n):
                self._simulate()

        assert len(self.game.board.move_stack) == depth, "search did not restore the root position"
        return self.root

    def _simulate(self):
        """
        One simulation: select down to a leaf, expand and evaluate it, back the value up the
        path, and undo every move played on the way down. The board has to end up back at the
        root even when the leaf turned out to be terminal.
        """
        if self.prepare_leaf():
            policy_logits, value = self.evaluator.evaluate(self.game)
            self.complete_leaf(policy_logits, value)

    def prepare_leaf(self):
        """
        First half of a simulation. bool: True when the board is now sitting on a leaf that needs
        a network evaluation, which has to be handed to complete_leaf() before anything else
        touches this tree. False when the simulation already finished on its own (the leaf was
        terminal), in which case the value has been backed up and the board is back at the root.
        """
        if self._pending is not None:
            raise RuntimeError("prepare_leaf called twice without complete_leaf")
        self._maybe_add_root_noise()

        node, path = self._select()
        value, moves = self._expand_and_evaluate(node)
        if value is None:
            self._pending = (node, path, moves)
            return True

        self._backup(path, value)
        self._undo_to_root(path)
        return False

    def complete_leaf(self, policy_logits, value):
        """
        Second half of a simulation: expand the pending leaf with the network's policy, back up
        the network's value, and walk the board back to the root.
        """
        node, path, moves = self._pending
        self._pending = None
        _, priors = self._legal_priors(policy_logits, moves)
        node.expand(moves, priors)
        self._backup(path, float(value))
        self._undo_to_root(path)

    # ---- the four phases -------------------------------------------------------------

    def _select(self):
        """
        (leaf_node, path): walk down from the root following PUCT.

        At each expanded, non-terminal node: take best_child_index, play that move on the game,
        append (node, move_index) to the path, and step into children[move_index], allocating a
        fresh Node there when the edge has never been followed. Stops at the first node that is
        unexpanded or terminal, and that node is the leaf.
        """
        c_puct, fpu = self.config.c_puct, self.config.fpu
        node, path = self.root, []
        while node.expanded and not node.is_terminal:
            idx = node.best_child_index(c_puct, fpu)
            self.game.play_move(node.moves[idx])
            path.append((node, idx))
            child = node.children[idx]
            if child is None:
                child = node.children[idx] = Node()
            node = child
        return node, path

    def _expand_and_evaluate(self, node):
        """
        (value, moves) for the leaf `node`, from the perspective of the side to move there.

        Asks _terminal_value first; when the game is over the value is stored on the node and
        returned, and the node stays unexpanded. Otherwise value comes back as None together with
        the legal move list: the position still needs the network, and complete_leaf() does the
        expansion once its output is in. (Split this way so the network call can be batched
        across many trees, see search_many.)
        """
        if node.is_terminal:
            return node.terminal_value, None

        moves = self.game.legal_moves_list()
        terminal = self._terminal_value(moves)
        if terminal is not None:
            node.terminal_value = terminal
            return terminal, None
        return None, moves

    def _backup(self, path, value):
        """
        Walk `path` back to the root, adding one visit and the value to every edge taken.

        `value` arrives from the leaf's point of view, and the value written on edge (node, idx)
        has to be from the point of view of the side to move at `node`, so the sign flips once
        per step up. Per edge: N[idx] += 1, W[idx] += signed_value.

        This is the easiest place in the file to get a sign backwards, and a wrong sign looks
        like a working search that just plays badly. Worth a test on a forced mate in one: the
        mating move's root Q has to come out near +1.
        """
        for node, idx in reversed(path):
            value = -value   # the parent is the other player: flip BEFORE writing on its edge
            node.N[idx] += 1
            node.W[idx] += value

    def _undo_to_root(self, path):
        """
        Take back exactly len(path) moves, putting the board back where _select started.
        """
        for _ in range(len(path)):
            self.game.undo_move()

    # ---- helpers ---------------------------------------------------------------------

    def _legal_priors(self, policy_logits, moves=None):
        """
        (moves, priors) for the current position: the legal move list, and np.float32[k] priors.

        Gathers the logits at the hash_move index of each legal move and softmaxes over that
        subset alone (subtracting the max first for stability), so illegal moves take no
        probability mass. The compact length k array is what the search wants;
        ChessGame.create_legal_moves_mask stays the right tool for the training time loss.
        """
        if moves is None:
            moves = self.game.legal_moves_list()
        idx = self._move_indices(moves)
        logits = np.asarray(policy_logits, dtype=np.float32)[idx]
        p = np.exp(logits - logits.max())
        return moves, (p / p.sum()).astype(np.float32)

    def _move_indices(self, moves):
        """np.int64[k]: hash_move of every move, in the current position."""
        return np.fromiter((self.game.hash_move(m) for m in moves), dtype=np.int64, count=len(moves))

    def _terminal_value(self, moves=None):
        """
        float | None: None while the game is still going, otherwise the value of the position for
        the side to move.

        -1.0 for checkmate (the side to move has been mated), 0.0 for stalemate, insufficient
        material, the seventy five move rule and fivefold repetition. Threefold repetition and
        the fifty move rule are only claimable, and are scored 0.0 straight away here, the way
        AlphaZero does. ChessGame.positions_map already counts repetitions, so the threefold
        check is an O(1) lookup instead of python-chess's is_repetition() scan.

        `moves` is the legal move list when the caller already has it, to avoid generating it
        twice; checkmate/stalemate is then just "no legal moves", split by is_check().
        """
        board = self.game.board
        has_moves = bool(moves) if moves is not None else any(True for _ in board.generate_legal_moves())
        if not has_moves:
            return -1.0 if board.is_check() else 0.0
        if board.halfmove_clock >= 100:                # fifty move rule, covers seventy five too
            return 0.0
        if self.game.positions_map.get(board._transposition_key(), 0) >= 3:   # covers fivefold
            return 0.0
        if board.is_insufficient_material():
            return 0.0
        return None

    def _add_dirichlet_noise(self, node):
        """
        Replace part of the root priors with Dirichlet noise:
        priors = (1 - frac) * priors + frac * dirichlet(alpha, k).

        Root only, once per search. It is what stops self play from opening the same way every
        game, so the training data keeps some variety.
        """
        frac = self.config.dirichlet_frac
        noise = self.rng.dirichlet(np.full(len(node.moves), self.config.dirichlet_alpha))
        node.priors = ((1 - frac) * node.priors + frac * noise).astype(np.float32)

    def _maybe_add_root_noise(self):
        """Noise the current root exactly once, as soon as it is expanded."""
        root = self.root
        if self.config.add_root_noise and root.expanded and self._noised_root is not root:
            self._add_dirichlet_noise(root)
            self._noised_root = root

    # ---- output, and driving a game --------------------------------------------------

    def visit_counts(self):
        """
        np.int32[k]: the root's visit counts, aligned with root.moves.
        """
        return self.root.N.copy()

    def _visit_distribution(self, temperature):
        """np.float64[k]: N ** (1 / temperature), normalised; one-hot on the argmax at 0."""
        counts = self.root.N.astype(np.float64)
        if temperature == 0 or counts.sum() == 0:
            p = np.zeros_like(counts)
            p[int(np.argmax(counts))] = 1.0
            return p
        counts = counts ** (1.0 / temperature)
        return counts / counts.sum()

    def policy_target(self, temperature=1.0):
        """
        np.float32[POLICY_SIZE]: the training target pi.

        Scatters the normalised N ** (1 / temperature) into a full width policy vector at each
        move's hash_move index, leaving every illegal index at zero.
        """
        pi = np.zeros(POLICY_SIZE, dtype=np.float32)
        pi[self._move_indices(self.root.moves)] = self._visit_distribution(temperature)
        return pi

    def policy_target_sparse(self, temperature=1.0):
        """
        (np.int64[k], np.float32[k]): the same target as policy_target, as (indices, probs), which
        is what the replay buffer stores (k is ~30, POLICY_SIZE is 4672).
        """
        return (self._move_indices(self.root.moves),
                self._visit_distribution(temperature).astype(np.float32))

    def select_move(self, move_number):
        """
        chess.Move: the move to actually play.

        Sampled from the visit distribution at config.temperature while
        move_number < config.temperature_moves, and the most visited move after that.
        """
        if not self.root.expanded:
            raise RuntimeError("select_move called before search (or on a finished game)")
        if move_number < self.config.temperature_moves and self.config.temperature > 0:
            p = self._visit_distribution(self.config.temperature)
            idx = int(self.rng.choice(len(p), p=p))
        else:
            idx = int(np.argmax(self.root.N))
        return self.root.moves[idx]

    def advance(self, move):
        """
        Play `move` and keep the part of the tree underneath it.

        The child at that edge becomes the new root with all of its statistics intact (a fresh
        Node when the edge was never visited), and the rest of the tree is dropped. Reusing the
        subtree is worth roughly a free doubling of the simulation count in self play. The new
        root gets its own noise on the next search() call.
        """
        child = None
        if self.root.expanded:
            try:
                child = self.root.children[self.root.moves.index(move)]
            except ValueError:
                child = None
        self.game.play_move(move)   # raises on an illegal move, before the tree is touched
        self.root = child if child is not None else Node()

    def reset(self, game=None):
        """
        Throw the tree away and start again from `game`, or from the current game when no new one
        is given.
        """
        if game is not None:
            self.game = game
        self.root = Node()
        self._noised_root = None
        self._pending = None


def search_many(trees, evaluator, num_simulations=None):
    """
    Run search() on many trees at once, with ONE network call per round instead of one per leaf.

    Each round every tree selects one leaf (prepare_leaf); the ones that need the network are
    evaluated together through evaluator.evaluate_batch, then each finishes its simulation
    (complete_leaf). A tree whose root is not expanded yet gets one extra round, which only
    expands the root, exactly like search() does. Every tree must own its own ChessGame.
    """
    if not trees:
        return
    n = trees[0].config.num_simulations if num_simulations is None else num_simulations
    rounds = {id(t): n + (0 if t.root.expanded else 1) for t in trees}
    active = [t for t in trees if not t.root.is_terminal]

    for r in range(n + 1):
        pending = [t for t in active if r < rounds[id(t)] and not t.root.is_terminal and t.prepare_leaf()]
        if pending:
            results = evaluator.evaluate_batch([t.game for t in pending])
            for t, (policy_logits, value) in zip(pending, results):
                t.complete_leaf(policy_logits, value)
