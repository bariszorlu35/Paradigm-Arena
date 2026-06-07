// Drive the new render loop headlessly to catch any runtime error in the
// animation path (easing, radial gradients, effects, leader ring, death shrink).
const fs=require("fs"), vm=require("vm");
const body=fs.readFileSync("index.html","utf8").split("<script>")[1].split("</script>")[0];
const stubs=`
  var __raf=null; var requestAnimationFrame=function(){return 0;}; var cancelAnimationFrame=function(){};
  function __mkctx(){return new Proxy({},{get:function(t,p){return (p in t)?t[p]:function(){return {addColorStop:function(){}};};},set:function(t,p,v){t[p]=v;return true;}});}
  function __mkel(){return {style:{},value:"4",textContent:"",innerHTML:"",getContext:function(){return __mkctx();},width:760,height:760,clientWidth:760,getContext:function(){return __mkctx();},addEventListener:function(){},parentElement:{clientWidth:780}};}
  var window={devicePixelRatio:2, addEventListener:function(){}};
  var document={getElementById:function(){return __mkel();}};
`;
const harness=`
  running=true;
  var drew=0, t=0, deaths=0, maxEff=0;
  for(var k=0;k<400 && !winner;k++){ t+=16; frame(t); drew++; maxEff=Math.max(maxEff,effects.length); }
  console.log("frames rendered:", drew, " rounds:", tick, " winner:", winner?winner.key:"(running)", " peak effects:", maxEff);
  console.log("RENDER LOOP OK");
`;
try { vm.runInThisContext(stubs+"\n"+body+"\n"+harness); }
catch(e){ console.log("RENDER LOOP FAILED —", e.message); process.exit(1); }
