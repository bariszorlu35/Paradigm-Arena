// Run the REAL index.html game logic headless (DOM stubbed) to confirm the
// multi-agent loop runs without errors and produces a winner.
const fs = require("fs");
const vm = require("vm");

const html = fs.readFileSync("index.html", "utf8");
const body = html.split("<script>")[1].split("</script>")[0];

// --- DOM / canvas stubs (names prefixed __ to avoid clashing with engine vars) ---
const stubs = `
  var requestAnimationFrame=function(){return 0;};
  var cancelAnimationFrame=function(){};
  function __mkctx(){ return new Proxy({}, {
    get:function(t,p){ return (p in t)?t[p]:function(){ return {addColorStop:function(){}}; }; },
    set:function(t,p,v){ t[p]=v; return true; } }); }
  function __mkel(){ return { style:{}, value:"3", textContent:"", innerHTML:"",
    getContext:function(){ return __mkctx(); }, width:640, height:640,
    addEventListener:function(){} }; }
  var document={ getElementById:function(id){ return __mkel(); } };
  var window={};
`;

const harness = `
  resetGame();
  var startMaxLvl = Math.max.apply(null, agents.map(a=>levelOf(a.score)));
  var guard=0;
  while(!winner && guard<7000){ step(); guard++; }
  var lvls = agents.map(a=>a.key+":L"+levelOf(a.score)+(a.alive?"":"(x)"));
  console.log("rounds:", tick);
  console.log("levels:", lvls.join("  "));
  console.log("winner:", winner ? winner.key : "none");
  var maxLvl = Math.max.apply(null, agents.map(a=>levelOf(a.score)));
  var ok = (winner!==null) && (tick>0) && (maxLvl>=startMaxLvl) && (guard<7000);
  console.log(ok ? "GAME LOOP OK" : "GAME LOOP FAILED");
`;

try {
  vm.runInThisContext(stubs + "\n" + body + "\n" + harness);
} catch (e) {
  console.log("GAME LOOP FAILED — exception:", e.message);
  process.exit(1);
}
