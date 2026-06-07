const fs=require("fs"), vm=require("vm");
const body=fs.readFileSync("index.html","utf8").split("<script>")[1].split("</script>")[0];
const stubs=`
  var requestAnimationFrame=function(){return 0;};var cancelAnimationFrame=function(){};
  function __mkctx(){return new Proxy({},{get:function(t,p){return (p in t)?t[p]:function(){return {addColorStop:function(){}};};},set:function(t,p,v){t[p]=v;return true;}});}
  function __mkel(){return {style:{},value:"3",textContent:"",innerHTML:"",getContext:function(){return __mkctx();},width:640,height:640,addEventListener:function(){}};}
  var document={getElementById:function(){return __mkel();}};var window={};
`;
const harness=`
  if(process.env.EXPLORE) EXPLORE=+process.env.EXPLORE;
  var N=parseInt(process.env.N||"40"), tally={}, rsum=0;
  for(var g=0; g<N; g++){
    resetGame(); var guard=0;
    while(!winner && guard<7000){ step(); guard++; }
    var w = winner?winner.key:"none";
    tally[w]=(tally[w]||0)+1; rsum+=tick;
  }
  console.log("games:",N," avg rounds:",(rsum/N).toFixed(1));
  Object.keys(tally).sort((a,b)=>tally[b]-tally[a]).forEach(k=>
    console.log("  "+k.padEnd(15)+tally[k]+"  ("+(100*tally[k]/N).toFixed(0)+"%)"));
`;
vm.runInThisContext(stubs+"\n"+body+"\n"+harness);
