"""Render a gameplay GIF of Paradigm Arena straight from the engine.

Runs the same multi-agent game the website runs (loads policies.json, mirrors
the JS encode/decide/step), draws each frame with Pillow, and writes demo.gif.
No browser needed.
"""
import json, math, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

random.seed(11); np.random.seed(11)
P = json.load(open("policies.json"))
M = P["meta"]
DIRS = M["dirs"]; NACT = M["nAct"]; CLOSE = M["closeScale"]
BASE_R = M["agentBaseR"]; FOOD_R = M["foodR"]; VAL = M["value"]; SPEEDM = M["speed"]
WORLD = 100; NFOOD = 24; EPS = 0.1
EXPLORE = 0.05; SPEED_MUL = 1.0
STUCK_WIN, STUCK_MIN, ESC_LEN = 10, 6, 12

STYLE = {
    "RL":             ("#ff5d5d", "RL"),
    "Supervised":     ("#4dabf7", "SUP"),
    "SelfSupervised": ("#9775fa", "SELF"),
    "SemiSupervised": ("#ffd43b", "SEMI"),
    "Evolutionary":   ("#51cf66", "EVO"),
    "Unsupervised":   ("#ff922b", "UNS"),
}
NAMES = {"RL":"Reinforcement Learning","Supervised":"Supervised (Imitation)",
         "SelfSupervised":"Self-Supervised","SemiSupervised":"Semi-Supervised",
         "Evolutionary":"Evolutionary","Unsupervised":"Unsupervised"}
ORDER = ["RL","Supervised","SelfSupervised","SemiSupervised","Evolutionary","Unsupervised"]


# ---- engine (mirror of the JS) ----
def forward(layers, x):
    a = list(x)
    for L in layers:
        W, b = L["W"], L["b"]; out = []
        for i in range(len(W)):
            s = b[i]; Wi = W[i]
            for j in range(len(Wi)): s += Wi[j]*a[j]
            out.append(math.tanh(s) if L["act"] == "tanh" else s)
        a = out
    return a

def argmax(v): return max(range(len(v)), key=lambda i: v[i])
def radius(s): return BASE_R + 0.18*s
def speed_of(s): return max(1.15, SPEEDM - 0.055*s)
def level_of(s): return min(10, 1 + int(s//3))

def unit_close(ax, ay, tx, ty):
    dx, dy = tx-ax, ty-ay; d = math.hypot(dx, dy)
    if d < 1e-6: return 0.0, 0.0, 1.0
    return dx/d, dy/d, 1/(1+d/CLOSE)

def encode(ag, agents, foods):
    ax, ay, ar = ag["x"], ag["y"], radius(ag["score"])
    fs = sorted(foods, key=lambda f: (f["x"]-ax)**2+(f["y"]-ay)**2)
    f1 = fs[0] if fs else {"x":ax,"y":ay}; f2 = fs[1] if len(fs) > 1 else f1
    th = pr = None; td = pd = 1e18
    for o in agents:
        if o is ag or not o["alive"]: continue
        orr = radius(o["score"]); dd = (o["x"]-ax)**2+(o["y"]-ay)**2
        if orr > ar+EPS:
            if dd < td: td, th = dd, o
        elif orr < ar-EPS:
            if dd < pd: pd, pr = dd, o
    s = [0.0]*13
    s[0],s[1],s[2] = unit_close(ax,ay,f1["x"],f1["y"])
    s[3],s[4],s[5] = unit_close(ax,ay,f2["x"],f2["y"])
    if th: s[6],s[7],s[8] = unit_close(ax,ay,th["x"],th["y"])
    if pr: s[9],s[10],s[11] = unit_close(ax,ay,pr["x"],pr["y"])
    s[12] = min(1.0,(1+ag["score"]/4)/10)
    return s

def value_of(s): return VAL["food"]*s[2] + VAL["prey"]*s[11] - VAL["threat"]*s[8]

def kmeans_target(foods, k=3):
    pts = [(f["x"],f["y"]) for f in foods]
    if len(pts) <= k:
        return (sum(p[0] for p in pts)/len(pts), sum(p[1] for p in pts)/len(pts))
    c = [list(pts[i]) for i in random.sample(range(len(pts)), k)]
    lab = [0]*len(pts)
    for _ in range(6):
        for i,(x,y) in enumerate(pts):
            lab[i] = min(range(k), key=lambda j:(x-c[j][0])**2+(y-c[j][1])**2)
        for j in range(k):
            m = [pts[i] for i in range(len(pts)) if lab[i]==j]
            if m: c[j] = [sum(p[0] for p in m)/len(m), sum(p[1] for p in m)/len(m)]
    cnt = [lab.count(j) for j in range(k)]
    return tuple(c[cnt.index(max(cnt))])

def decide(ag, agents, foods):
    if random.random() < EXPLORE: return random.randrange(NACT)
    spec = ag["spec"]; s = encode(ag, agents, foods); t = spec["type"]
    if t in ("qnet","policy"): return argmax(forward(spec["layers"], s))
    if t == "forwardmodel":
        H = spec.get("horizon",1); g = spec.get("gamma",0.9); best=-1e18; ba=0
        for a in range(NACT):
            st = list(s); tot = 0.0
            for h in range(H):
                inp = st + [1.0 if k==a else 0.0 for k in range(NACT)]
                d = forward(spec["layers"], inp)
                st = [st[i]+d[i] for i in range(len(st))]
                tot += (g**h)*value_of(st)
            if tot > best: best, ba = tot, a
        return ba
    if t == "kmeans":
        if s[8] > 0.5: dx, dy = -s[6], -s[7]
        else:
            tx, ty = kmeans_target(foods, spec.get("k",3)); dx, dy = tx-ag["x"], ty-ag["y"]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9: return 0
        return max(range(NACT), key=lambda i: DIRS[i][0]*dx+DIRS[i][1]*dy)
    return 0

def clampw(v): return min(WORLD, max(0.0, v))


class Game:
    def __init__(self):
        self.foods = [{"x":random.uniform(0,WORLD),"y":random.uniform(0,WORLD)} for _ in range(NFOOD)]
        self.agents = []
        for k in ORDER:
            x = WORLD*(0.15+0.7*random.random()); y = WORLD*(0.15+0.7*random.random())
            self.agents.append({"key":k,"spec":P["agents"][k],"x":x,"y":y,"score":0.0,
                                "alive":True,"lastAct":0,"esc":0,"escDir":0,
                                "ckX":x,"ckY":y,"ckT":0})
        self.tick = 0; self.winner = None; self.effects = []

    def step(self):
        if self.winner: return
        acts = [decide(a, self.agents, self.foods) if a["alive"] else -1 for a in self.agents]
        for i,a in enumerate(self.agents):
            if not a["alive"]: continue
            act = acts[i]
            if a["esc"] > 0: act = a["escDir"]; a["esc"] -= 1
            sp = speed_of(a["score"])*SPEED_MUL; d = DIRS[act]
            a["x"] = clampw(a["x"]+sp*d[0]); a["y"] = clampw(a["y"]+sp*d[1]); a["lastAct"] = act
        for a in self.agents:
            if not a["alive"]: continue
            ar = radius(a["score"])
            for f in self.foods:
                if math.hypot(f["x"]-a["x"], f["y"]-a["y"]) < ar+FOOD_R:
                    a["score"] += 1; self.effects.append([f["x"],f["y"],STYLE[a["key"]][0],"food",0])
                    f["x"] = random.uniform(0,WORLD); f["y"] = random.uniform(0,WORLD)
        live = sorted([a for a in self.agents if a["alive"]], key=lambda a:-radius(a["score"]))
        for A in live:
            if not A["alive"]: continue
            ar = radius(A["score"])
            for B in live:
                if B is A or not B["alive"]: continue
                br = radius(B["score"])
                if ar > br+EPS and math.hypot(A["x"]-B["x"], A["y"]-B["y"]) < ar+0.7*br:
                    A["score"] += B["score"]+3; B["alive"] = False
                    self.effects.append([B["x"],B["y"],STYLE[B["key"]][0],"eat",0])
        for a in self.agents:
            if not a["alive"]: continue
            if self.tick - a["ckT"] >= STUCK_WIN:
                disp = math.hypot(a["x"]-a["ckX"], a["y"]-a["ckY"])
                if disp < STUCK_MIN and a["esc"] == 0:
                    nearWall = a["x"]<8 or a["x"]>WORLD-8 or a["y"]<8 or a["y"]>WORLD-8
                    if nearWall: dx, dy = WORLD/2-a["x"], WORLD/2-a["y"]
                    else:
                        perp = (a["lastAct"]+(2 if random.random()<0.5 else 6))%NACT
                        dx, dy = DIRS[perp][0], DIRS[perp][1]
                    a["escDir"] = max(range(NACT), key=lambda i: DIRS[i][0]*dx+DIRS[i][1]*dy)
                    a["esc"] = ESC_LEN
                a["ckX"], a["ckY"], a["ckT"] = a["x"], a["y"], self.tick
        self.tick += 1
        alive = [a for a in self.agents if a["alive"]]
        lvlw = next((a for a in self.agents if a["alive"] and level_of(a["score"])>=10), None)
        if lvlw: self.winner = lvlw
        elif len(alive) <= 1: self.winner = alive[0] if alive else None


# ---- rendering ----
SIZE = 460; SC = SIZE/WORLD; BG = (10, 14, 20)
def hx(h): return tuple(int(h[i:i+2],16) for i in (1,3,5))
def shade(h, amt):
    r,g,b = hx(h)
    f=lambda v:max(0,min(255,v+amt)); return (f(r),f(g),f(b))

def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()

def render(game):
    img = Image.new("RGB", (SIZE, SIZE), BG)
    glow = Image.new("RGBA", (SIZE, SIZE), (0,0,0,0)); gd = ImageDraw.Draw(glow)
    d = ImageDraw.Draw(img)
    for i in range(0, SIZE, SIZE//12):
        d.line([(i,0),(i,SIZE)], fill=(255,255,255,9)); d.line([(0,i),(SIZE,i)], fill=(255,255,255,9))
    # food glow + core
    for f in game.foods:
        x,y = f["x"]*SC, f["y"]*SC
        gd.ellipse([x-7,y-7,x+7,y+7], fill=(70,224,140,70))
    # effects
    keep=[]
    for e in game.effects:
        e[4]+=1; span = 12 if e[3]=="eat" else 8
        if e[4] <= span:
            keep.append(e); age=e[4]/span; R=(8+age*34 if e[3]=="eat" else 3+age*13)
            col=hx(e[2]); a=int(200*(1-age))
            x,y=e[0]*SC,e[1]*SC
            gd.ellipse([x-R,y-R,x+R,y+R], outline=col+(a,), width=3 if e[3]=="eat" else 2)
    game.effects=keep
    for f in game.foods:
        x,y = f["x"]*SC, f["y"]*SC
        d.ellipse([x-3,y-3,x+3,y+3], fill=(70,224,140))
    img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
    d = ImageDraw.Draw(img)
    leader = max((a for a in game.agents if a["alive"]), key=lambda a:a["score"], default=None)
    for a in game.agents:
        if not a["alive"]: continue
        c, short = STYLE[a["key"]]; r = radius(a["score"])*SC; x,y = a["x"]*SC, a["y"]*SC
        if a is leader:
            d.ellipse([x-r-4,y-r-4,x+r+4,y+r+4], outline=c if isinstance(c,tuple) else hx(c), width=2)
        cc = hx(c)
        d.ellipse([x-r,y-r,x+r,y+r], fill=shade(c,-34))
        d.ellipse([x-r*0.82,y-r*0.82,x+r*0.82,y+r*0.82], fill=cc)
        d.ellipse([x-r*0.5-r*0.18,y-r*0.5-r*0.2,x-r*0.5+r*0.34,y-r*0.5+r*0.3], fill=shade(c,55))
        d.ellipse([x-r,y-r,x+r,y+r], outline=(235,235,235), width=2)
        fs = max(9, min(15, int(r*0.85)))
        d.text((x,y), short, fill=(8,18,31), font=font(fs), anchor="mm")
        d.text((x,y-r-7), "L"+str(level_of(a["score"])), fill=(255,255,255), font=font(11), anchor="mm")
    return img


def main():
    game = Game(); frames = [render(game)]
    MAXS = 230
    while not game.winner and game.tick < MAXS:
        game.step(); frames.append(render(game))
    # winner banner held at the end
    if game.winner:
        c, short = STYLE[game.winner["key"]]
        for _ in range(16):
            img = render(game); d = ImageDraw.Draw(img, "RGBA")
            d.rectangle([0, SIZE-58, SIZE, SIZE], fill=(5,8,12,205))
            d.text((SIZE//2, SIZE-38), "WINNER: "+NAMES[game.winner["key"]],
                   fill=hx(c), font=font(20), anchor="mm")
            d.text((SIZE//2, SIZE-16), game.winner["spec"]["type"]+"  •  "+str(game.tick)+" steps",
                   fill=(180,180,190), font=font(12), anchor="mm")
            frames.append(img)
    print("frames:", len(frames), "winner:", game.winner["key"] if game.winner else None, "ticks:", game.tick)
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    pal[0].save("demo.gif", save_all=True, append_images=pal[1:], duration=60, loop=0, optimize=True, disposal=2)
    import os; print("demo.gif size: %.2f MB" % (os.path.getsize("demo.gif")/1e6))


if __name__ == "__main__":
    main()
