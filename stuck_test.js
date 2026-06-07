// Measure "stuck in place" behavior with the anti-stuck fix OFF vs ON.
// A step is "stuck" if net displacement over the last STUCK_WIN steps < STUCK_MIN.
const fs=require("fs"), vm=require("vm");
const body=fs.readFileSync("index.html","utf8").split("<script>")[1].split("</script>")[0];
const stubs=`
  var requestAnimationFrame=function(){return 0;};var cancelAnimationFrame=function(){};
  function __mkctx(){return new Proxy({},{get:function(t,p){return (p in t)?t[p]:function(){return {addColorStop:function(){}};};},set:function(t,p,v){t[p]=v;return true;}});}
  function __mkel(){return {style:{},value:"4",textContent:"",innerHTML:"",getContext:function(){return __mkctx();},width:760,height:760,clientWidth:760,addEventListener:function(){},parentElement:{clientWidth:780}};}
  var window={devicePixelRatio:1,addEventListener:function(){}};
  var document={getElementById:function(){return __mkel();}};
`;
const harness=`
  function measure(flag){
    ANTISTUCK=flag;
    var GAMES=15, totalStuck=0, totalSteps=0, worstStreak=0;
    for(var g=0; g<GAMES; g++){
      resetGame();
      var H={}; agents.forEach(a=>H[a.key]=[]);
      var guard=0;
      while(!winner && guard<2500){ step(); agents.forEach(a=>{ if(a.alive) H[a.key].push([a.x,a.y]); }); guard++; }
      for(var k in H){
        var p=H[k], cur=0;
        for(var i=STUCK_WIN;i<p.length;i++){
          var dx=p[i][0]-p[i-STUCK_WIN][0], dy=p[i][1]-p[i-STUCK_WIN][1];
          totalSteps++;
          if(Math.hypot(dx,dy)<STUCK_MIN){ totalStuck++; cur++; if(cur>worstStreak)worstStreak=cur; } else cur=0;
        }
      }
    }
    return {pct:(100*totalStuck/totalSteps).toFixed(1), worst:worstStreak};
  }
  var off=measure(false), on=measure(true);
  console.log("anti-stuck OFF -> stuck "+off.pct+"% of steps, longest stuck streak "+off.worst+" steps");
  console.log("anti-stuck ON  -> stuck "+on.pct+"% of steps, longest stuck streak "+on.worst+" steps");
`;
try { vm.runInThisContext(stubs+"\n"+body+"\n"+harness); }
catch(e){ console.log("STUCK TEST FAILED —", e.message); process.exit(1); }
