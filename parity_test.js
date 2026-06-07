// Confirm browser inference == Python inference for the exported weights.
const P = require("./policies.json");
const D = require("./states.json");
const NACT = P.meta.nAct, VAL = P.meta.value;

function forward(layers, x){
  let a = x;
  for(const L of layers){
    const W=L.W,b=L.b,out=new Array(W.length);
    for(let i=0;i<W.length;i++){let s=b[i],Wi=W[i];for(let j=0;j<Wi.length;j++)s+=Wi[j]*a[j];
      out[i]=(L.act==="tanh")?Math.tanh(s):s;}
    a=out;
  }
  return a;
}
function argmax(v){let bi=0,bv=v[0];for(let i=1;i<v.length;i++)if(v[i]>bv){bv=v[i];bi=i;}return bi;}
function valueOf(s){return VAL.food*s[2]+VAL.prey*s[11]-VAL.threat*s[8];}

function decideNet(spec,s){
  if(spec.type==="forwardmodel"){
    const H=spec.horizon||1,g=spec.gamma||0.9; let best=-1e18,ba=0;
    for(let a=0;a<NACT;a++){
      let st=s.slice(),total=0;
      for(let h=0;h<H;h++){
        const inp=st.concat(Array.from({length:NACT},(_,k)=>k===a?1:0));
        const d=forward(spec.layers,inp);
        for(let i=0;i<st.length;i++)st[i]+=d[i];
        total+=Math.pow(g,h)*valueOf(st);
      }
      if(total>best){best=total;ba=a;}
    }
    return ba;
  }
  return argmax(forward(spec.layers,s));
}

let totalMismatch=0;
for(const name of Object.keys(D.actions)){
  let mm=0;
  for(let i=0;i<D.states.length;i++){
    if(decideNet(P.agents[name], D.states[i]) !== D.actions[name][i]) mm++;
  }
  totalMismatch+=mm;
  console.log(`${name.padEnd(15)} mismatches: ${mm} / ${D.states.length}`);
}

// numeric check of raw forward outputs
let maxDiff=0;
for(let i=0;i<D.raw_rl.length;i++){
  const o=forward(P.agents.RL.layers, D.states[i]);
  for(let k=0;k<o.length;k++) maxDiff=Math.max(maxDiff, Math.abs(o[k]-D.raw_rl[i][k]));
}
console.log("max abs diff (RL raw outputs):", maxDiff.toExponential(3));
console.log(totalMismatch===0 && maxDiff<1e-9 ? "PARITY OK" : "PARITY FAILED");
