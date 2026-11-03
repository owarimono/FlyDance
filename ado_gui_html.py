HTML = r"""<!doctype html>
<meta charset=utf-8>
<title>Monstruo &mdash; Ado Blue</title>
<style>
:root{--bg:#0a1430;--fg:#eaf2ff;--dim:#8fb3e8;--blue:#4da3ff;--hot:#7ec8ff;--glow:#1e5fbf}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;
     min-height:100vh;display:grid;place-items:center;
     background-image:radial-gradient(ellipse at 50% 0%,#12285c 0%,var(--bg) 60%)}
main{width:min(920px,94vw);padding:28px 0}
h1{font-size:20px;letter-spacing:.22em;text-transform:uppercase;color:var(--blue);
   text-shadow:0 0 18px var(--glow);font-weight:600}
h1 span{color:var(--dim);letter-spacing:.05em;text-transform:none;font-size:13px}
.bar{display:flex;gap:10px;align-items:center;margin:18px 0;flex-wrap:wrap}
button{background:linear-gradient(180deg,var(--blue),#2b6fd4);color:#04122e;border:0;
       padding:10px 22px;border-radius:999px;font-weight:700;font-size:13px;
       letter-spacing:.08em;cursor:pointer;box-shadow:0 0 22px -6px var(--blue)}
button:hover{filter:brightness(1.12)}
button:disabled{opacity:.4;cursor:default}
.grid{display:grid;grid-template-columns:1.4fr 1fr;gap:16px}
@media(max-width:720px){.grid{grid-template-columns:1fr}}
.card{background:#0f1f4a;border:1px solid #1e3d7d;border-radius:14px;padding:16px}
canvas{width:100%;display:block;border-radius:8px;background:#08122b}
#view{width:100%;display:block;border-radius:8px;background:#08122b;
      min-height:200px;object-fit:contain;image-rendering:pixelated}
label{display:block;color:var(--dim);font-size:11px;letter-spacing:.12em;
      text-transform:uppercase;margin-bottom:6px}
.kv{display:flex;justify-content:space-between;padding:6px 0;
    border-bottom:1px solid #17316a;font-variant-numeric:tabular-nums}
.kv b{color:var(--hot);font-weight:600}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#2a4a80;
     margin-right:7px;vertical-align:middle}
.dot.on{background:var(--hot);box-shadow:0 0 12px var(--hot)}
.note{color:var(--dim);font-size:12px;margin-top:12px}
a{color:var(--blue)}
</style>
<main>
<h1>Monstruo <span>&mdash; Ado Blue &middot; tethered fly, REINFORCE policy</span></h1>
<div class=bar>
  <button id=go>DANCE</button>
  <button id=play>&#9654; PLAY SONG</button>
  <span class=note id=status>idle</span>
</div>
<audio id=au src=/audio loop controls style=width:100%;margin:6px 0 14px"></audio>
<div class=grid>
  <div class=card>
    <label>fly &mdash; live MuJoCo 3D (legs + wings)</label>
    <img id=view src=/frame.jpg alt="MuJoCo 3D view">
    <div class=note>this is the real renderer output, not a drawing.</div>
  </div>
  <div class=card>
    <label>telemetry</label>
    <div class=kv><span><i class=dot id=beat></i>beat</span><b id=b>0</b></div>
    <div class=kv><span>time</span><b id=t>0.0 s</b></div>
    <div class=kv><span>sync (on/off-beat)</span><b id=s>&mdash;</b></div>
    <div class=kv><span>|W| policy</span><b id=w>0</b></div>
    <div class=kv><span>state</span><b id=st>idle</b></div>
    <div class=note>sync = movement on the beat / between beats.
      &gt;1 means it is hitting the beat, not just flailing.</div>
    <div class=note><a href="/video">download mp4</a> (after a run)</div>
  </div>
</div>
</main>
<script>
const $=id=>document.getElementById(id), au=$('au');
let raf=null, seq=0;
$('play').onclick=()=>au.play();
$('go').onclick=()=>{
  $('go').disabled=true; $('status').textContent='running...';
  // start the song from the top so beat 0 and audio 0 line up
  au.currentTime=0; au.play().catch(()=>{});
  fetch('/start',{method:'POST'});
  clearInterval(raf);
  raf=setInterval(poll,100);
};
async function poll(){
  const s=await (await fetch('/state')).json();
  $('b').textContent=s.beats; $('t').textContent=s.t.toFixed(1)+' s';
  $('w').textContent=s.w2.toFixed(0); $('st').textContent=s.status;
  $('s').textContent=s.sync?s.sync.toFixed(2)+'x':'—';
  $('beat').className='dot'+(s.on_beat?' on':'');
  // real MuJoCo render, cache-busted so the browser refetches each frame
  $('view').src='/frame.jpg?v='+(s.frame||0)+'#'+seq++;
  if(s.status==='done'){clearInterval(raf);$('go').disabled=false;
    $('status').textContent='done — '+s.sync.toFixed(2)+'x sync';}
}
poll();
</script>
"""
