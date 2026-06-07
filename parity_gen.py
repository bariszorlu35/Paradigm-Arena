"""Generate random states + Python-chosen actions/outputs for net agents,
so Node can confirm the browser inference matches Python exactly."""
import json, numpy as np
import eval as E

rng = np.random.default_rng(123)

def rand_state():
    s = np.zeros(13)
    for base in (0, 3, 6, 9):           # food1, food2, threat, prey
        if rng.random() < 0.85:
            ang = rng.uniform(0, 2*np.pi)
            s[base] = np.cos(ang); s[base+1] = np.sin(ang)
            s[base+2] = rng.uniform(0, 1)
    s[12] = rng.uniform(0, 1)
    return s

states = [rand_state() for _ in range(400)]
nets = ["RL", "Supervised", "SemiSupervised", "SelfSupervised", "Evolutionary"]
actions = {n: [int(E.act(n, E.P["agents"][n], None, s)) for s in states] for n in nets}

# raw forward outputs for one policy net + one forwardmodel sample (numeric check)
raw_rl = [E.forward(E.P["agents"]["RL"]["layers"], s).tolist() for s in states[:6]]

json.dump({"states": [s.tolist() for s in states],
           "actions": actions, "raw_rl": raw_rl},
          open("states.json", "w"))
print("wrote states.json:", len(states), "states,", len(nets), "net agents")
