from __future__ import annotations
import argparse, os, random, math, collections
import numpy as np
import tensorflow as tf
from tensorflow import keras
from fanorona_engine import *
from model import build_model, ACTIONS

class Node:
    __slots__=('state','parent','move','prior','n','w','children','expanded')
    def __init__(self,state,parent=None,move=None,prior=0.0):
        self.state=state; self.parent=parent; self.move=move; self.prior=prior; self.n=0; self.w=0.0; self.children=[]; self.expanded=False
    @property
    def q(self): return self.w/self.n if self.n else 0.0

def masked_priors(logits, turns):
    ids=[m.action_id for m,_ in turns]
    vals=np.asarray([logits[i] for i in ids],np.float64)
    vals-=vals.max(); p=np.exp(np.clip(vals,-40,40)); p/=max(p.sum(),1e-12)
    # Encourage distinct full-turn choices when they share the same first step.
    return p

def predict(model,state):
    pol,val=model(np.expand_dims(canonical_planes(state),0),training=False)
    return pol.numpy()[0],float(val.numpy()[0,0])

def expand(node,model,root_player):
    turns=complete_turns(node.state)
    if not turns: return heuristic(node.state,root_player)
    logits,v=predict(model,node.state)
    pri=masked_priors(logits,turns)
    node.children=[Node(ns,node,m,p) for (m,ns),p in zip(turns,pri)]
    node.expanded=True
    return v

def terminal_value(state,root_player):
    w=winner(state)
    if not w: return None
    return 1.0 if w==root_player else -1.0

def search(root,model,sims,c_puct,root_player):
    expand(root,model,root_player)
    for _ in range(sims):
        node=root; path=[node]
        while node.expanded and node.children:
            best=max(node.children,key=lambda ch: ch.q + c_puct*ch.prior*math.sqrt(max(1,node.n))/(1+ch.n))
            node=best; path.append(node)
            tv=terminal_value(node.state,root_player)
            if tv is not None: break
        tv=terminal_value(node.state,root_player)
        if tv is None:
            value=expand(node,model,root_player)
        else: value=tv
        for x in reversed(path):
            x.n+=1; x.w+=value
            value=-value
    return root

def target_from_root(root,temperature):
    visits=np.zeros(ACTIONS,np.float32)
    for ch in root.children:
        if ch.move:
            visits[ch.move.action_id]+=ch.n
    s=visits.sum()
    if s<=0: visits+=1.0/len(root.children)
    else:
        visits/=s
        if temperature>0:
            z=np.power(np.maximum(visits,1e-12),1.0/temperature); z/=z.sum(); visits=z.astype(np.float32)
    return visits

def choose(root,temp):
    cs=[c for c in root.children if c.n>0]
    if not cs: return random.choice(root.children)
    if temp<=0: return max(cs,key=lambda c:c.n)
    w=np.array([c.n for c in cs],dtype=np.float64); w=np.power(w,1.0/temp); w/=w.sum()
    return cs[np.random.choice(len(cs),p=w)]

def self_play(model,games,sims,vela=False,c_puct=1.35):
    data=[]; wins=collections.Counter()
    for g in range(games):
        s=State(vela=vela); history=[]; max_turns=260
        for turn in range(max_turns):
            root=Node(s)
            search(root,model,sims,c_puct,s.player)
            target=target_from_root(root,0.0 if turn>16 else 1.0)
            # Store canonical state; final result is filled after game.
            history.append((canonical_planes(s),target,s.player))
            chosen=choose(root,0.8 if turn<12 else 0.15)
            s=chosen.state
            if winner(s): break
        w=winner(s)
        if not w:
            # Draw by length: reward material/mobility rather than random winner.
            score=heuristic(s,BLACK); w=BLACK if score>0.03 else WHITE if score<-0.03 else 0
        wins[w]+=1
        for x,p,pl in history:
            z=0.0 if w==0 else (1.0 if pl==w else -1.0)
            data.append((x,p,z))
        print(f'game {g+1}/{games}: winner={w} plies={len(history)}',flush=True)
    return data,wins

def train_epoch(model,data,epochs,batch,lr):
    x=np.stack([a for a,_,_ in data]).astype(np.float32)
    p=np.stack([b for _,b,_ in data]).astype(np.float32)
    v=np.asarray([c for *_,c in data],np.float32).reshape(-1,1)
    ds=tf.data.Dataset.from_tensor_slices((x,{'policy':p,'value':v})).shuffle(min(len(x),20000)).batch(batch).prefetch(tf.data.AUTOTUNE)
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr,weight_decay=1e-5),
                  loss={'policy':keras.losses.CategoricalCrossentropy(from_logits=True), 'value':keras.losses.MeanSquaredError()},
                  loss_weights={'policy':1.0,'value':1.0})
    model.fit(ds,epochs=epochs,verbose=1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--iterations',type=int,default=12); ap.add_argument('--games-per-iteration',type=int,default=80)
    ap.add_argument('--simulations',type=int,default=600); ap.add_argument('--epochs',type=int,default=4)
    ap.add_argument('--batch',type=int,default=128); ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--resume',default=''); ap.add_argument('--out',default='checkpoints')
    ap.add_argument('--vela',action='store_true'); ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args(); random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
    os.makedirs(args.out,exist_ok=True)
    model=keras.models.load_model(args.resume) if args.resume else build_model(128,12)
    replay=collections.deque(maxlen=120000)
    for it in range(1,args.iterations+1):
        print(f'\n=== SELF PLAY ITERATION {it}/{args.iterations} ===')
        data,wins=self_play(model,args.games_per_iteration,args.simulations,args.vela)
        replay.extend(data)
        sample=list(replay)
        print('replay=',len(sample),'wins=',dict(wins))
        train_epoch(model,sample,args.epochs,args.batch,args.lr*(0.7**((it-1)//4)))
        path=os.path.join(args.out,'latest.keras'); model.save(path)
        model.save(os.path.join(args.out,f'iteration_{it:03d}.keras'))
        print('saved',path)

if __name__=='__main__': main()
