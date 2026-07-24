# Win-rate for the structured multi-action policy (data/torch_multi.pt). Same metric as winrate.py.
import subprocess, json, os, torch, torch.nn as nn
from torch.distributions import Bernoulli, Normal, Categorical
N_OBS,N_HID,FCAND,KCAND,KP,FP=16,32,7,6,8,4
ATTACK_MAP=[None,1,2,10,9]; BUILD_MAP=[None,5,6,7,8,11,12]
SEEDS=int(os.environ.get("SEEDS","6")); MAXST=int(os.environ.get("MAXST","1500"))
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.fc1=nn.Linear(N_OBS,N_HID)
        s.expand=nn.Linear(N_HID,1); s.attack=nn.Linear(N_HID,5); s.build=nn.Linear(N_HID,7); s.diplo=nn.Linear(N_HID,1)
        s.res=nn.Linear(N_HID,1); s.vf=nn.Linear(N_HID,1)
        s.cand=nn.Sequential(nn.Linear(FCAND,16),nn.Tanh(),nn.Linear(16,1)); s.place=nn.Sequential(nn.Linear(FP,16),nn.Tanh(),nn.Linear(16,1))
        s.res_log_std=nn.Parameter(torch.tensor(-0.5))
    def forward(s,x):
        h=torch.tanh(s.fc1(x)); return s.expand(h).squeeze(-1),s.attack(h),s.build(h),s.diplo(h).squeeze(-1),s.res(h).squeeze(-1)
net=Net(); net.load_state_dict(torch.load("data/torch_multi.pt"),strict=False); net.eval()
def pad(rows,K,Fd):
    c=torch.zeros(K,Fd); m=torch.zeros(K)
    for i,r in enumerate(rows[:K]): c[i]=torch.tensor(r,dtype=torch.float32); m[i]=1.0
    return c,m
def tsx():
    local=os.path.join("node_modules",".bin","tsx.cmd" if os.name=="nt" else "tsx")
    return [local] if os.path.exists(local) else ["npx","tsx"]
p=subprocess.Popen(tsx()+["src/env/env_server.ts"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
def rpc(m): p.stdin.write(json.dumps(m)+"\n"); p.stdin.flush(); return json.loads(p.stdout.readline())
print(f"MULTI  MAP={os.environ.get('MAP','box')} DIFFICULTY={os.environ.get('DIFFICULTY','medium')} over {SEEDS} seeds")
wins=0
for sd in range(SEEDS):
    r=rpc({"cmd":"reset","seed":90001+sd}); o,cds,pts=r["obs"],r["cands"],r["ptiles"]; peak=0.0; alive=False; st=0
    while st<MAXST:
        cp,cm=pad(cds,KCAND,FCAND); pp,pm=pad(pts,KP,FP)
        with torch.no_grad():
            el,al,bl,dl,rm=net(torch.tensor(o,dtype=torch.float32))
            ex=int(Bernoulli(logits=el).sample()); at=int(Categorical(logits=al).sample()); bd=int(Categorical(logits=bl).sample()); dp=int(Bernoulli(logits=dl).sample())
            res=float(torch.sigmoid(Normal(rm,net.res_log_std.exp()).sample()))
            tg=int(Categorical(logits=net.cand(cp).squeeze(-1).masked_fill(cm==0,-1e9)).sample())
            ptg=int(Categorical(logits=net.place(pp).squeeze(-1).masked_fill(pm==0,-1e9)).sample())
        a=[0]*13
        if ex: a[0]=1
        if ATTACK_MAP[at] is not None: a[ATTACK_MAP[at]]=1
        if BUILD_MAP[bd] is not None: a[BUILD_MAP[bd]]=1
        if dp: a[3]=1
        r=rpc({"cmd":"step","actions":a,"reserve":res,"target":tg,"ptarget":ptg}); o=r["obs"]; cds=r["cands"]; pts=r["ptiles"]
        peak=max(peak,o[0]); alive=o[0]>0; st+=1
        if r["done"]: break
    won=alive and peak>=0.8; wins+=won
    print(f"  seed {90001+sd}: peak {peak*100:5.2f}%  {'WON' if won else ('survived' if alive else 'died')} ({st} steps)")
print(f"WIN-RATE: {wins}/{SEEDS} = {100*wins/SEEDS:.0f}%")
p.stdin.close(); p.terminate()
