"""Evaluate the EXPORTED policies (policies.json) in the solo arena, exactly
as the browser will run them. Reports avg score & survival vs a random
baseline so we know the learning actually worked."""
import json, math
import numpy as np
import train as T

P = json.load(open("policies.json"))
DIRS = np.array(P["meta"]["dirs"])
VAL = P["meta"]["value"]


def forward(layers, x):
    a = np.asarray(x, dtype=float)
    for L in layers:
        W = np.array(L["W"]); b = np.array(L["b"])
        a = W @ a + b
        if L["act"] == "tanh":
            a = np.tanh(a)
    return a


def value_of(s):
    return VAL["food"] * s[2] + VAL["prey"] * s[11] - VAL["threat"] * s[8]


def kmeans_target(points, k=3, iters=6):
    pts = np.array(points)
    if len(pts) <= k:
        return pts.mean(0)
    c = pts[np.random.default_rng(0).choice(len(pts), k, replace=False)]
    for _ in range(iters):
        d = ((pts[:, None, :] - c[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = pts[m].mean(0)
    counts = np.bincount(lab, minlength=k)
    return c[counts.argmax()]


def act(agent, spec, env, s):
    typ = spec["type"]
    if typ in ("qnet", "policy"):
        return int(np.argmax(forward(spec["layers"], s)))
    if typ == "forwardmodel":
        H = spec.get("horizon", 3); g = spec.get("gamma", 0.9)
        best, ba = -1e9, 0
        for a in range(T.N_ACT):
            oh = np.zeros(T.N_ACT); oh[a] = 1
            st = np.array(s, dtype=float); total = 0.0
            for h in range(H):                # imagine moving this way H steps
                st = st + forward(spec["layers"], np.concatenate([st, oh]))  # delta dynamics
                total += (g ** h) * value_of(st)
            if total > best:
                best, ba = total, a
        return ba
    if typ == "kmeans":
        if s[8] > 0.5:                       # threat near -> flee
            d = np.array([-s[6], -s[7]])
        else:
            tgt = kmeans_target(env.food, spec.get("k", 3))
            d = np.array([tgt[0] - env.ax, tgt[1] - env.ay])
        if np.allclose(d, 0):
            return 0
        return int(np.argmax(DIRS @ d))
    return 0


def run(agent, spec, episodes=40):
    sc, surv = [], []
    for _ in range(episodes):
        env = T.Arena(ep_len=200)
        s = env.reset(); done = False
        while not done:
            a = act(agent, spec, env, s)
            s, r, done = env.step(a)
        sc.append(env.score); surv.append(env.t)
    return np.mean(sc), np.mean(surv)


def run_random(episodes=40):
    sc, surv = [], []
    for _ in range(episodes):
        env = T.Arena(ep_len=200)
        s = env.reset(); done = False
        while not done:
            s, r, done = env.step(np.random.randint(T.N_ACT))
        sc.append(env.score); surv.append(env.t)
    return np.mean(sc), np.mean(surv)


if __name__ == "__main__":
    rsc, rsv = run_random()
    print(f"{'AGENT':<16}{'avg_score':>10}{'avg_steps':>11}")
    print("-" * 37)
    print(f"{'(random)':<16}{rsc:>10.1f}{rsv:>11.1f}")
    for name, spec in P["agents"].items():
        sc, sv = run(name, spec)
        print(f"{name:<16}{sc:>10.1f}{sv:>11.1f}")
