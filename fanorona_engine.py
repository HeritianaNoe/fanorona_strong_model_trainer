from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import random

ROWS, COLS = 5, 9
N_ACTIONS = ROWS * COLS * 8 * 3
EMPTY, BLACK, WHITE = 0, 1, 2
DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
DIR_TO_I = {d:i for i,d in enumerate(DIRS)}

@dataclass(frozen=True)
class Move:
    fr: int; fc: int; tr: int; tc: int; di: int; typ: int; captured: tuple[tuple[int,int], ...] = ()
    @property
    def action_id(self):
        return ((self.fr * COLS + self.fc) * 8 + self.di) * 3 + self.typ
    @property
    def is_capture(self): return bool(self.captured)

class State:
    __slots__ = ('b','player','first_turn','vela','chain_pos','last_dir','visited','ply')
    def __init__(self, b=None, player=BLACK, first_turn=True, vela=False):
        self.b = [row[:] for row in b] if b is not None else initial_board()
        self.player = player; self.first_turn = first_turn; self.vela = vela
        self.chain_pos = None; self.last_dir = None; self.visited = set(); self.ply = 0

def initial_board():
    b = [[EMPTY]*COLS for _ in range(ROWS)]
    for r in range(ROWS):
        for c in range(COLS):
            if r < 2: b[r][c] = WHITE
            elif r > 2: b[r][c] = BLACK
            elif c == 4: b[r][c] = EMPTY
            elif c < 4: b[r][c] = BLACK if c % 2 == 0 else WHITE
            else: b[r][c] = BLACK if c % 2 == 1 else WHITE
    return b

def copy_state(s):
    n = State(s.b, s.player, s.first_turn, s.vela)
    n.chain_pos=s.chain_pos; n.last_dir=s.last_dir; n.visited=set(s.visited); n.ply=s.ply
    return n

def opponent(p): return WHITE if p == BLACK else BLACK

def inside(r,c): return 0 <= r < ROWS and 0 <= c < COLS

def dirs_for(r,c):
    return range(8) if (r+c)%2==0 else range(4)

def captures_for(b, fr,fc,tr,tc,di,player):
    dr,dc=DIRS[di]; opp=opponent(player)
    app=[]; rr,cc=tr+dr,tc+dc
    while inside(rr,cc) and b[rr][cc]==opp:
        app.append((rr,cc)); rr+=dr; cc+=dc
    wd=[]; rr,cc=fr-dr,fc-dc
    while inside(rr,cc) and b[rr][cc]==opp:
        wd.append((rr,cc)); rr-=dr; cc-=dc
    return app,wd

def legal_step_moves(s: State, capture_only=False, piece=None, exclude_dir=None, visited=None):
    b=s.b; p=s.player; out=[]
    
    # Fiarovana sy fanitsiana raha ohatra ka int ny piece fa tsy tuple
    if piece is not None:
        if isinstance(piece, (tuple, list)):
            positions = [piece]
        else:
            # Raha ohatra ka int fotsiny no tafiditra dia tadiavo ny toerana misy azy na raiso araka ny tokony ho izy
            positions = [piece] if isinstance(piece, tuple) else []
    else:
        positions = [(r,c) for r in range(ROWS) for c in range(COLS) if b[r][c]==p]

    # Raha toa ka mbola banga ny positions satria nisy int nitsofoka dia avereno amin'ny fomba azo antoka
    if not positions and piece is not None:
        # Raha toa ka ohatra anarana r fotsiny izy dia avereno amin'ny fikarohana mahazatra na raiso ny s.chain_pos
        positions = [s.chain_pos] if s.chain_pos else [(r,c) for r in range(ROWS) for c in range(COLS) if b[r][c]==p]

    visited = visited or set()
    for fr,fc in positions:
        if not inside(fr,fc) or b[fr][fc]!=p: continue
        for di in dirs_for(fr,fc):
            if exclude_dir is not None and (di==exclude_dir or di==opposite_dir(exclude_dir)): continue
            dr,dc=DIRS[di]; tr,tc=fr+dr,fc+dc
            if not inside(tr,tc) or b[tr][tc]!=EMPTY or (tr,tc) in visited: continue
            app,wd=captures_for(b,fr,fc,tr,tc,di,p)
            if app:
                out.append(Move(fr,fc,tr,tc,di,1,tuple(app)))
            if wd:
                out.append(Move(fr,fc,tr,tc,di,2,tuple(wd)))
            if not app and not wd and not capture_only:
                out.append(Move(fr,fc,tr,tc,di,0,()))
    return out

def opposite_dir(di):
    dr,dc=DIRS[di]; return DIR_TO_I[(-dr,-dc)]

def legal_moves(s: State):
    caps=legal_step_moves(s, capture_only=True, piece=s.chain_pos, exclude_dir=s.last_dir, visited=s.visited) if s.chain_pos is not None else legal_step_moves(s, capture_only=True)
    if caps: return caps
    return legal_step_moves(s, capture_only=False, piece=s.chain_pos, exclude_dir=s.last_dir, visited=s.visited) if s.chain_pos is not None else legal_step_moves(s, capture_only=False)

def apply_step(s: State, m: Move):
    n=copy_state(s); p=n.player
    n.b[m.fr][m.fc]=EMPTY; n.b[m.tr][m.tc]=p
    for r,c in m.captured: n.b[r][c]=EMPTY
    n.ply += 1
    if m.is_capture and not n.vela:
        n.chain_pos=(m.tr,m.tc); n.last_dir=m.di; n.visited=set(s.visited) | {(m.fr,m.fc),(m.tr,m.tc)}
    else:
        n.chain_pos=None; n.last_dir=None; n.visited=set()
        n.player=opponent(p); n.first_turn=False
    return n

def _end_turn(cur: State):
    n=copy_state(cur)
    n.player=opponent(cur.player)
    n.first_turn=False
    n.chain_pos=None; n.last_dir=None; n.visited=set()
    return n

def complete_turns(s: State):
    roots=legal_moves(s)
    if not roots: return []
    if s.vela or not any(m.is_capture for m in roots):
        return [(m, apply_step(s,m)) for m in roots]
    result=[]
    def dfs(cur, first, last):
        result.append((first, _end_turn(cur)))
        # Eto no nisy ny diso teo aloha: natao (last.tr, last.tc) fa tsy last.tr irery fotsiny
        cont=legal_step_moves(cur, capture_only=True, piece=(last.tr, last.tc), exclude_dir=last.di, visited=cur.visited)
        for x in cont:
            dfs(apply_step(cur,x), first, x)
    for m in roots: dfs(apply_step(s,m),m,m)
    return result

def winner(s):
    bc=sum(x==BLACK for row in s.b for x in row); wc=sum(x==WHITE for row in s.b for x in row)
    if bc==0: return WHITE
    if wc==0: return BLACK
    return 0

def canonical_planes(s: State):
    p=s.player; o=opponent(p)
    import numpy as np
    x=np.zeros((ROWS,COLS,7),dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            x[r,c,0]=s.b[r][c]==p; x[r,c,1]=s.b[r][c]==o; x[r,c,2]=s.b[r][c]==EMPTY
            x[r,c,3]=((r+c)%2==0); x[r,c,4]=s.chain_pos is not None; x[r,c,6]=s.vela
    if s.chain_pos: x[s.chain_pos[0],s.chain_pos[1],5]=1.0
    return x

def heuristic(s: State, root_player: int):
    w=sum(x==root_player for row in s.b for x in row); o=sum(x==opponent(root_player) for row in s.b for x in row)
    if o==0: return 1.0
    if w==0: return -1.0
    old=s.player
    lm=len(legal_moves(s)); s.player=opponent(s.player); om=len(legal_moves(s)); s.player=old
    center=sum(1 for r in range(ROWS) for c in range(COLS) if (r,c)==(2,4) and s.b[r][c]==root_player)
    mycaps=sum(m.is_capture for m in legal_step_moves(s,capture_only=True)) if s.player==root_player else 0
    s.player=opponent(s.player); oppcaps=sum(m.is_capture for m in legal_step_moves(s,capture_only=True)); s.player=opponent(s.player)
    score=0.11*(w-o)+0.025*(lm-om)+0.07*(mycaps-oppcaps)+0.03*center
    return max(-1.0,min(1.0,score/2.5))