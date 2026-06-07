"""
Learning-Paradigms Arena  --  Python training pipeline.

Trains one agent per ML learning paradigm in a shared single-agent
"food + prey + hazard" arena, then exports every learned policy to
policies.json so the browser game can run them with a tiny pure-JS
neural-net forward pass.

All agents share the SAME state encoding (13 features) and the SAME
action space (8 compass directions) so the only thing that differs is
the learning method.

Paradigms:
  RL              -> DQN-lite (Q-learning, small MLP)
  Supervised      -> imitation / behavioural cloning of a greedy expert
  SelfSupervised  -> learns a forward (world) model, plans 1 step ahead
  SemiSupervised  -> few expert labels + many pseudo-labelled samples
  Evolutionary    -> genetic algorithm over policy-net weights
  Unsupervised    -> no training; runtime k-means on food (exported as config)
"""

import json
import math
import numpy as np

RNG = np.random.default_rng(7)

# --------------------------------------------------------------------------
# Constants  (mirrored exactly in the JS game)
# --------------------------------------------------------------------------
WORLD       = 100.0
N_FOOD      = 14
N_PREY      = 4
N_HAZ       = 4
AGENT_BASE_R= 2.2
FOOD_R      = 1.2
PREY_R      = 1.9
HAZ_R       = 4.6
SPEED       = 2.4
HAZ_SPEED   = 1.9
PREY_SPEED  = 1.6
CLOSE_SCALE = 18.0
STATE_DIM   = 13
N_ACT       = 8
GAMMA       = 0.95
EP_LEN      = 200

# 8 compass directions (index i -> angle i*45deg)
DIRS = np.array([[math.cos(i * math.pi / 4.0), math.sin(i * math.pi / 4.0)]
                 for i in range(N_ACT)], dtype=np.float64)

# value-function coefficients used by the self-supervised planner (and eval)
VAL_FOOD, VAL_PREY, VAL_THREAT = 1.0, 0.7, 1.4


def clampw(v):
    return min(WORLD, max(0.0, v))


def unit_close(ax, ay, tx, ty):
    dx, dy = tx - ax, ty - ay
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return 0.0, 0.0, 1.0
    return dx / d, dy / d, 1.0 / (1.0 + d / CLOSE_SCALE)


def agent_radius(score):
    return AGENT_BASE_R + 0.18 * score


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
class Arena:
    """Single learner + food pellets + smaller prey + bigger hazards."""

    def __init__(self, ep_len=EP_LEN):
        self.ep_len = ep_len
        self.reset()

    def _rand_pos(self):
        return [RNG.uniform(0, WORLD), RNG.uniform(0, WORLD)]

    def reset(self):
        self.ax, self.ay = WORLD / 2, WORLD / 2
        self.score = 0.0
        self.t = 0
        self.food = [self._rand_pos() for _ in range(N_FOOD)]
        self.prey = [self._rand_pos() for _ in range(N_PREY)]
        self.haz = [self._rand_pos() for _ in range(N_HAZ)]
        return self.encode()

    # ---- shared state encoder (identical in JS) -------------------------
    def encode(self):
        ar = agent_radius(self.score)
        ax, ay = self.ax, self.ay

        # two nearest food
        fd = sorted(self.food, key=lambda p: (p[0]-ax)**2 + (p[1]-ay)**2)
        f1 = fd[0] if len(fd) > 0 else [ax, ay]
        f2 = fd[1] if len(fd) > 1 else f1

        # nearest threat (bigger mobile entity) and nearest prey (smaller)
        threats, preys = [], []
        for p in self.haz:
            (threats if HAZ_R >= ar else preys).append(p)
        for p in self.prey:
            (preys if PREY_R <= ar else threats).append(p)

        def nearest(lst):
            if not lst:
                return None
            return min(lst, key=lambda p: (p[0]-ax)**2 + (p[1]-ay)**2)

        t = nearest(threats)
        pr = nearest(preys)

        s = np.zeros(STATE_DIM)
        s[0], s[1], s[2] = unit_close(ax, ay, f1[0], f1[1])
        s[3], s[4], s[5] = unit_close(ax, ay, f2[0], f2[1])
        if t is not None:
            s[6], s[7], s[8] = unit_close(ax, ay, t[0], t[1])
        if pr is not None:
            s[9], s[10], s[11] = unit_close(ax, ay, pr[0], pr[1])
        s[12] = min(1.0, (1 + self.score / 4.0) / 10.0)  # level norm
        return s

    def _move_wander(self, lst, speed, bias_to_agent=0.0):
        for p in lst:
            ang = RNG.uniform(0, 2 * math.pi)
            dx, dy = math.cos(ang), math.sin(ang)
            if bias_to_agent != 0.0:
                tx, ty = self.ax - p[0], self.ay - p[1]
                d = math.hypot(tx, ty) + 1e-6
                dx = (1 - abs(bias_to_agent)) * dx + bias_to_agent * tx / d
                dy = (1 - abs(bias_to_agent)) * dy + bias_to_agent * ty / d
                n = math.hypot(dx, dy) + 1e-6
                dx, dy = dx / n, dy / n
            p[0] = clampw(p[0] + speed * dx)
            p[1] = clampw(p[1] + speed * dy)

    def step(self, action):
        ar = agent_radius(self.score)
        spd = max(1.3, SPEED - 0.04 * self.score)
        dx, dy = DIRS[action]
        self.ax = clampw(self.ax + spd * dx)
        self.ay = clampw(self.ay + spd * dy)

        # hazards lightly chase, prey lightly flee
        self._move_wander(self.haz, HAZ_SPEED, bias_to_agent=0.25)
        self._move_wander(self.prey, PREY_SPEED, bias_to_agent=-0.25)

        reward = -0.01
        done = False
        ar = agent_radius(self.score)

        # eat food
        for i, p in enumerate(self.food):
            if math.hypot(p[0]-self.ax, p[1]-self.ay) < ar + FOOD_R:
                self.score += 1
                reward += 1.0
                self.food[i] = self._rand_pos()

        # eat prey (need to be bigger)
        for i, p in enumerate(self.prey):
            if ar >= PREY_R and math.hypot(p[0]-self.ax, p[1]-self.ay) < ar + 0.5*PREY_R:
                self.score += 2
                reward += 2.0
                self.prey[i] = self._rand_pos()

        # hazards: eat agent (death) unless agent grew bigger than them
        ar = agent_radius(self.score)
        for i, p in enumerate(self.haz):
            d = math.hypot(p[0]-self.ax, p[1]-self.ay)
            if HAZ_R >= ar and d < HAZ_R + 0.5*ar:
                reward -= 5.0
                done = True
                break
            if ar > HAZ_R and d < ar + 0.5*HAZ_R:
                self.score += 3
                reward += 3.0
                self.haz[i] = self._rand_pos()

        self.t += 1
        if self.t >= self.ep_len:
            done = True
        return self.encode(), reward, done


# --------------------------------------------------------------------------
# Tiny 1-hidden-layer MLP (pure numpy, manual backprop)
# --------------------------------------------------------------------------
class MLP:
    def __init__(self, n_in, n_h, n_out, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 1, (n_in, n_h)) * math.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_h)
        self.W2 = r.normal(0, 1, (n_h, n_out)) * math.sqrt(2.0 / n_h)
        self.b2 = np.zeros(n_out)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def _backprop(self, X, dz2, lr):
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - self.a1 ** 2)
        dW1 = X.T @ dz1
        db1 = dz1.sum(0)
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W1 -= lr * dW1; self.b1 -= lr * db1

    def train_classifier(self, X, y, epochs=300, lr=0.05, batch=128):
        n = len(X)
        for ep in range(epochs):
            idx = RNG.permutation(n)
            for s in range(0, n, batch):
                b = idx[s:s+batch]
                xb, yb = X[b], y[b]
                logits = self.forward(xb)
                logits -= logits.max(1, keepdims=True)
                p = np.exp(logits); p /= p.sum(1, keepdims=True)
                dz2 = p.copy()
                dz2[np.arange(len(b)), yb] -= 1
                dz2 /= len(b)
                self._backprop(xb, dz2, lr)

    def train_regressor(self, X, Y, epochs=200, lr=0.02, batch=128):
        n = len(X)
        for ep in range(epochs):
            idx = RNG.permutation(n)
            for s in range(0, n, batch):
                b = idx[s:s+batch]
                xb, yb = X[b], Y[b]
                out = self.forward(xb)
                dz2 = 2 * (out - yb) / len(b)
                self._backprop(xb, dz2, lr)

    def genome(self):
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def set_genome(self, g, n_in, n_h, n_out):
        i = 0
        self.W1 = g[i:i+n_in*n_h].reshape(n_in, n_h); i += n_in*n_h
        self.b1 = g[i:i+n_h]; i += n_h
        self.W2 = g[i:i+n_h*n_out].reshape(n_h, n_out); i += n_h*n_out
        self.b2 = g[i:i+n_out]

    def export(self):
        return {
            "layers": [
                {"W": self.W1.T.tolist(), "b": self.b1.tolist(), "act": "tanh"},
                {"W": self.W2.T.tolist(), "b": self.b2.tolist(), "act": "linear"},
            ]
        }


def softmax_logits(v):
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


# --------------------------------------------------------------------------
# Greedy expert (operates purely on the state vector)
# --------------------------------------------------------------------------
def expert_action(s):
    threat_close = s[8]
    prey_close = s[11]
    food_close = s[2]
    if threat_close > 0.45:                      # danger near -> flee
        d = np.array([-s[6], -s[7]])
    elif prey_close > food_close and prey_close > 0.05:
        d = np.array([s[9], s[10]])              # chase prey
    else:
        d = np.array([s[0], s[1]])               # go to food
    if np.allclose(d, 0):
        return RNG.integers(N_ACT)
    return int(np.argmax(DIRS @ d))


# --------------------------------------------------------------------------
# 1) Reinforcement Learning  (DQN-lite)
# --------------------------------------------------------------------------
def train_rl(episodes=300, ep_len=150, update_every=4, batch=96):
    net = MLP(STATE_DIM, 24, N_ACT, seed=1)
    eps = 1.0
    buf = []
    env = Arena(ep_len=ep_len)
    step_count = 0
    for ep in range(episodes):
        s = env.reset()
        done = False
        while not done:
            if RNG.random() < eps:
                a = RNG.integers(N_ACT)
            else:
                a = int(np.argmax(net.forward(s[None])[0]))
            s2, r, done = env.step(a)
            buf.append((s, a, r, s2, done))
            s = s2
            step_count += 1
            if len(buf) > 6000:
                buf.pop(0)
            # periodic minibatch update (keeps training fast)
            if len(buf) >= batch and step_count % update_every == 0:
                idx = RNG.integers(0, len(buf), batch)
                bs = np.array([buf[i][0] for i in idx])
                ba = np.array([buf[i][1] for i in idx])
                br = np.array([buf[i][2] for i in idx])
                bs2 = np.array([buf[i][3] for i in idx])
                bd = np.array([buf[i][4] for i in idx], dtype=float)
                q = net.forward(bs)
                q2 = net.forward(bs2)
                target = q.copy()
                target[np.arange(batch), ba] = br + GAMMA * (1 - bd) * q2.max(1)
                dz2 = 2 * (q - target) / batch
                net._backprop(bs, dz2, lr=0.01)
        eps = max(0.05, eps * 0.98)
    return net


# --------------------------------------------------------------------------
# 2) Supervised imitation  +  4) Semi-supervised
# --------------------------------------------------------------------------
def collect_expert(n_states=7000, noise=0.0):
    # `noise` = fraction of imperfect labels. A realistically imperfect teacher
    # keeps Supervised strong but beatable (it is only as good as its data).
    env = Arena()
    X, y = [], []
    s = env.reset()
    while len(X) < n_states:
        a = expert_action(s)
        if RNG.random() < noise:
            a = int(RNG.integers(N_ACT))
        X.append(s); y.append(a)
        s, _, done = env.step(a)
        if done:
            s = env.reset()
    return np.array(X), np.array(y)


def collect_random(n_states=5000):
    env = Arena()
    X = []
    s = env.reset()
    while len(X) < n_states:
        a = RNG.integers(N_ACT)
        X.append(s)
        s, _, done = env.step(a)
        if done:
            s = env.reset()
    return np.array(X)


def train_supervised(X, y):
    net = MLP(STATE_DIM, 24, N_ACT, seed=2)
    net.train_classifier(X, y, epochs=350, lr=0.08)
    return net


def train_semisupervised(X_exp, y_exp, X_unlab, n_labeled=220):
    # only a few true labels
    Xl, yl = X_exp[:n_labeled], y_exp[:n_labeled]
    net = MLP(STATE_DIM, 24, N_ACT, seed=3)
    net.train_classifier(Xl, yl, epochs=300, lr=0.08, batch=64)
    # pseudo-label the unlabeled pool, keep high-confidence, retrain
    logits = net.forward(X_unlab)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    conf = p.max(1)
    keep = conf > 0.55
    Xp, yp = X_unlab[keep], p.argmax(1)[keep]
    Xc = np.concatenate([Xl, Xp])
    yc = np.concatenate([yl, yp])
    net.train_classifier(Xc, yc, epochs=200, lr=0.05)
    return net


# --------------------------------------------------------------------------
# 3) Self-supervised: forward (world) model + 1-step planning
# --------------------------------------------------------------------------
def collect_dynamics(n=9000):
    # mixed expert/random policy so the world model sees near-food, near-prey
    # and near-threat transitions (not just the random-flailing distribution)
    env = Arena()
    X, Y = [], []
    s = env.reset()
    while len(X) < n:
        a = expert_action(s) if RNG.random() < 0.5 else RNG.integers(N_ACT)
        oh = np.zeros(N_ACT); oh[a] = 1
        s2, _, done = env.step(a)
        X.append(np.concatenate([s, oh]))
        Y.append(s2)
        s = s2
        if done:
            s = env.reset()
    return np.array(X), np.array(Y)


def train_selfsup(X, Y):
    # predict the DELTA (s' - s); residual dynamics capture the small
    # per-step changes far better than predicting absolute next state.
    dY = Y - X[:, :STATE_DIM]
    net = MLP(STATE_DIM + N_ACT, 40, STATE_DIM, seed=4)
    net.train_regressor(X, dY, epochs=340, lr=0.03)
    return net


def value_of(state):
    return VAL_FOOD * state[2] + VAL_PREY * state[11] - VAL_THREAT * state[8]


def selfsup_action(net, s):
    best, ba = -1e9, 0
    for a in range(N_ACT):
        oh = np.zeros(N_ACT); oh[a] = 1
        pred = net.forward(np.concatenate([s, oh])[None])[0]
        v = value_of(pred)
        if v > best:
            best, ba = v, a
    return ba


# --------------------------------------------------------------------------
# 5) Evolutionary: GA over policy-net weights
# --------------------------------------------------------------------------
def evo_fitness(genome, n_ep=2):
    net = MLP(STATE_DIM, 16, N_ACT, seed=0)
    net.set_genome(genome, STATE_DIM, 16, N_ACT)
    total = 0.0
    env = Arena(ep_len=150)
    for _ in range(n_ep):
        s = env.reset(); done = False
        while not done:
            a = int(np.argmax(net.forward(s[None])[0]))
            s, r, done = env.step(a)
            total += r
    return total / n_ep


def train_evolutionary(pop=36, gens=24, elite=6, sigma=0.25):
    dim = STATE_DIM*16 + 16 + 16*N_ACT + N_ACT
    population = [RNG.normal(0, 0.6, dim) for _ in range(pop)]
    best_g, best_f = None, -1e9
    for g in range(gens):
        fits = np.array([evo_fitness(ind) for ind in population])
        order = np.argsort(fits)[::-1]
        if fits[order[0]] > best_f:
            best_f = fits[order[0]]
            best_g = population[order[0]].copy()
        parents = [population[i] for i in order[:elite]]
        newpop = [p.copy() for p in parents]                 # elitism
        sig = sigma * (0.95 ** g)                             # anneal mutation
        while len(newpop) < pop:
            pa = parents[RNG.integers(elite)]
            pb = parents[RNG.integers(elite)]
            mask = RNG.random(dim) < 0.5
            child = np.where(mask, pa, pb) + RNG.normal(0, sig, dim)
            newpop.append(child)
        population = newpop
    net = MLP(STATE_DIM, 16, N_ACT, seed=0)
    net.set_genome(best_g, STATE_DIM, 16, N_ACT)
    return net, best_f


# --------------------------------------------------------------------------
# Run everything
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("Training RL (DQN-lite)...")
    rl = train_rl(episodes=460, update_every=3)

    print("Collecting expert + random data...")
    Xe, ye = collect_expert(noise=0.22)   # imperfect teacher (realistic)
    Xr = collect_random()

    print("Training Supervised (imitation)...")
    sup = train_supervised(Xe, ye)

    print("Training Semi-supervised...")
    semi = train_semisupervised(Xe, ye, Xr)

    print("Training Self-supervised (world model)...")
    Xd, Yd = collect_dynamics()
    ssl = train_selfsup(Xd, Yd)

    print("Training Evolutionary (GA)...")
    evo, evo_fit = train_evolutionary()
    print(f"  best evo fitness: {evo_fit:.2f}")

    policies = {
        "meta": {
            "stateDim": STATE_DIM, "nAct": N_ACT, "world": WORLD,
            "nFood": N_FOOD, "nPrey": N_PREY, "nHaz": N_HAZ,
            "agentBaseR": AGENT_BASE_R, "foodR": FOOD_R, "preyR": PREY_R,
            "hazR": HAZ_R, "speed": SPEED, "hazSpeed": HAZ_SPEED,
            "preySpeed": PREY_SPEED, "closeScale": CLOSE_SCALE,
            "dirs": DIRS.tolist(),
            "value": {"food": VAL_FOOD, "prey": VAL_PREY, "threat": VAL_THREAT},
        },
        "agents": {
            "RL":             {"type": "qnet",        **rl.export()},
            "Supervised":     {"type": "policy",      **sup.export()},
            "SelfSupervised": {"type": "forwardmodel", "delta": True, "horizon": 1, "gamma": 0.9, "value": {"food": VAL_FOOD, "prey": VAL_PREY, "threat": VAL_THREAT}, **ssl.export()},
            "SemiSupervised": {"type": "policy",      **semi.export()},
            "Evolutionary":   {"type": "policy",      **evo.export()},
            "Unsupervised":   {"type": "kmeans",      "k": 3},
        },
    }
    with open("policies.json", "w") as f:
        json.dump(policies, f)
    print("Wrote policies.json")
    print("DONE")
