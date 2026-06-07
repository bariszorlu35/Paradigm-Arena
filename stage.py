"""Run one training stage at a time (keeps each call short) and save a
JSON fragment per agent. Finally assemble policies.json."""
import sys, json, time
import numpy as np
import train as T


def save(name, obj):
    with open(f"frag_{name}.json", "w") as f:
        json.dump(obj, f)
    print(f"saved frag_{name}.json")


def stage_rl():
    t = time.time()
    rl = T.train_rl(episodes=460, update_every=3)
    save("RL", {"type": "qnet", **rl.export()})
    print("rl time", round(time.time() - t, 1))


def stage_sup():
    t = time.time()
    Xe, ye = T.collect_expert(7000, noise=0.22)
    Xr = T.collect_random(5000)
    sup = T.train_supervised(Xe, ye)
    semi = T.train_semisupervised(Xe, ye, Xr)
    save("Supervised", {"type": "policy", **sup.export()})
    save("SemiSupervised", {"type": "policy", **semi.export()})
    print("sup time", round(time.time() - t, 1))


def stage_ssl():
    t = time.time()
    Xd, Yd = T.collect_dynamics(7000)
    ssl = T.train_selfsup(Xd, Yd)
    save("SelfSupervised", {"type": "forwardmodel", "delta": True,
                            "horizon": 1, "gamma": 0.9,
                            "value": {"food": T.VAL_FOOD, "prey": T.VAL_PREY,
                                      "threat": T.VAL_THREAT}, **ssl.export()})
    print("ssl time", round(time.time() - t, 1))


def stage_evo():
    t = time.time()
    evo, fit = T.train_evolutionary(pop=36, gens=22)
    save("Evolutionary", {"type": "policy", **evo.export()})
    print("evo time", round(time.time() - t, 1), "fit", round(fit, 2))


def stage_assemble():
    agents = {}
    for name in ["RL", "Supervised", "SemiSupervised", "SelfSupervised", "Evolutionary"]:
        with open(f"frag_{name}.json") as f:
            agents[name] = json.load(f)
    agents["Unsupervised"] = {"type": "kmeans", "k": 3}
    policies = {
        "meta": {
            "stateDim": T.STATE_DIM, "nAct": T.N_ACT, "world": T.WORLD,
            "nFood": T.N_FOOD, "nPrey": T.N_PREY, "nHaz": T.N_HAZ,
            "agentBaseR": T.AGENT_BASE_R, "foodR": T.FOOD_R, "preyR": T.PREY_R,
            "hazR": T.HAZ_R, "speed": T.SPEED, "hazSpeed": T.HAZ_SPEED,
            "preySpeed": T.PREY_SPEED, "closeScale": T.CLOSE_SCALE,
            "dirs": T.DIRS.tolist(),
            "value": {"food": T.VAL_FOOD, "prey": T.VAL_PREY, "threat": T.VAL_THREAT},
        },
        "agents": agents,
    }
    with open("policies.json", "w") as f:
        json.dump(policies, f)
    print("Wrote policies.json with agents:", list(agents.keys()))


if __name__ == "__main__":
    {"rl": stage_rl, "sup": stage_sup, "ssl": stage_ssl,
     "evo": stage_evo, "assemble": stage_assemble}[sys.argv[1]]()
