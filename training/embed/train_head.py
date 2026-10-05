"""Stage 1: frozen base image encoder + small MLP head -> 128-d (Matryoshka 64) space distilled from a teacher.

  python train_head.py --base mc2_s0 --teacher pe_l14 --name s0_pe
  python train_head.py --base b32_openai --teacher pe_l14 --name b32_pe

Shared 128-d space: P (teacher_dim -> 128, linear) is learned so that projected teacher image->label logits match the
teacher's own logits over the whole label table (KL, at 128 and 64 dims). Labels in 128-d are l2(P t_label), so new
labels need only the teacher text tower + P, offline. The head h (LN -> residual MLP adapter -> linear 128) learns
base(view) -> P t_img (cosine) + the same label KL + batch relational loss. P gets no gradient from the head losses.
"""
import os, sys, json, argparse, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK

DIMS = (128, 64)


class Head(nn.Module):
    def __init__(self, din, dout=128, hid=1024, depth=1, drop=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(din)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, din))
                                     for _ in range(depth)])
        self.out = nn.Linear(din, dout)

    def forward(self, x):
        x = self.ln(x)
        for b in self.blocks: x = x + b(x)
        return self.out(x)


def mkl(s_logits, t_logits):
    return F.kl_div(F.log_softmax(s_logits, -1), F.softmax(t_logits, -1), reduction='batchmean')


def norm(x): return F.normalize(x.float(), dim=-1)


def load_teacher(D, keys):
    """target = l2(mean of l2(box), l2(stretch)) per teacher, concat (/sqrt(k)) for an ensemble."""
    parts = []
    for k in keys:
        a = norm(torch.from_numpy(np.load(f'{D}/{k}_box.npy').astype(np.float32)))
        b = norm(torch.from_numpy(np.load(f'{D}/{k}_stretch.npy').astype(np.float32)))
        parts.append(norm(a + b))
    return torch.cat(parts, 1) / len(parts) ** 0.5


def load_text(keys):
    return torch.cat([norm(torch.from_numpy(np.load(f'{WORK}/labels/text_{k}.npy').astype(np.float32))) for k in keys], 1) / len(keys) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb', default=f'{WORK}/emb/lvis', help='comma list; the first may be partial (--done_only)')
    ap.add_argument('--base', default='mc2_s0')
    ap.add_argument('--teacher', default='pe_l14')
    ap.add_argument('--views', default='box,stretch,masked,aug')
    ap.add_argument('--name', default='')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--bs', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hid', type=int, default=1024)
    ap.add_argument('--depth', type=int, default=1)
    ap.add_argument('--tau_t', type=float, default=0.01)
    ap.add_argument('--w_cos', type=float, default=1.0)
    ap.add_argument('--w_kl', type=float, default=1.0)
    ap.add_argument('--w_rel', type=float, default=1.0)
    ap.add_argument('--crops', default=f'{WORK}/data/lvis_crops')
    ap.add_argument('--done_only', action='store_true', help='train only on rows of shards listed in done_shards.txt')
    ap.add_argument('--n_lab', type=int, default=8192, help='labels sampled per step for the KL (all if <=0)')
    a = ap.parse_args()
    tk = a.teacher.split(',')
    name = a.name or f'{a.base}_{"+".join(tk)}'
    run = f'{WORK}/runs/{name}'; os.makedirs(run, exist_ok=True); json.dump(vars(a), open(f'{run}/args.json', 'w'), indent=1)
    dev = 'cuda'; torch.manual_seed(0)
    dirs = a.emb.split(',')                                          # extra dirs (e.g. sim crops) are appended in full
    T = torch.cat([load_teacher(D, tk) for D in dirs])               # (N, Dt) cpu
    Lt = load_text(tk).to(dev)                                       # (K, Dt)
    views = a.views.split(',')
    X = {v: torch.cat([torch.from_numpy(np.load(f'{D}/{a.base}_{v}.npy')) for D in dirs]) for v in views}   # fp16 cpu
    N0 = sum(1 for _ in open(f'{dirs[0]}/ids.txt'))
    N, Dt, Db = T.shape[0], T.shape[1], X[views[0]].shape[1]
    keep = np.arange(N)
    if a.done_only:                                                  # extraction still running: use finished shards
        import glob
        done = set(open(f'{dirs[0]}/done_shards.txt').read().split())
        sh = sorted(glob.glob(f'{a.crops}/shard_*.tar')); cnt = [sum(1 for _ in open(x[:-4] + '.jsonl')) for x in sh]
        off = np.cumsum([0] + cnt[:-1])
        keep = np.concatenate([np.arange(o, o + c) for x, o, c in zip(sh, off, cnt) if os.path.basename(x) in done]
                              + [np.arange(N0, N)])
    g = np.random.default_rng(0); perm = keep[g.permutation(len(keep))]; nval = max(2000, len(keep) // 50)
    val, tr = perm[:nval], perm[nval:]
    P = nn.Linear(Dt, DIMS[0], bias=False).to(dev)
    h = Head(Db, DIMS[0], a.hid, a.depth).to(dev)
    ls_p = nn.Parameter(torch.tensor(np.log(1 / 0.03), device=dev)); ls_h = nn.Parameter(torch.tensor(np.log(1 / 0.03), device=dev))
    opt = torch.optim.AdamW([{'params': P.parameters(), 'weight_decay': 0}, {'params': h.parameters(), 'weight_decay': 0.05},
                             {'params': [ls_p, ls_h], 'weight_decay': 0}], lr=a.lr)
    steps = a.epochs * (len(tr) // a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    print(f'{name}: N {len(keep)} (val {nval}) base {Db} teacher {Dt} labels {Lt.shape[0]} views {views} params head '
          f'{sum(p.numel() for p in h.parameters()) / 1e6:.2f}M P {Dt * 128 / 1e6:.2f}M', flush=True)
    t0 = time.time(); step = 0
    for ep in range(a.epochs):
        g.shuffle(tr); h.train()
        for i in range(0, len(tr) - a.bs + 1, a.bs):
            idx = torch.from_numpy(np.sort(tr[i:i + a.bs]))
            vb = torch.from_numpy(g.integers(0, len(views), a.bs))   # random view per sample
            xb = torch.empty(len(idx), Db, dtype=torch.float16)
            for j, vv in enumerate(views):
                sel = vb == j
                if sel.any(): xb[sel] = X[vv][idx[sel]]
            xb = xb.to(dev).float()
            t = T[idx].to(dev)
            lab = Lt if a.n_lab <= 0 else Lt[torch.randint(0, Lt.shape[0], (a.n_lab,), device=dev)]
            tl = t @ lab.T / a.tau_t                                    # teacher logits
            pt_full, pl_full = P(t), P(lab)
            z = h(xb)
            loss = 0.; logs = {}
            for d in DIMS:
                pt, pl = norm(pt_full[:, :d]), norm(pl_full[:, :d])
                lp = mkl(pt @ pl.T * ls_p.exp(), tl)                       # projection keeps teacher's label ranking
                lp_rel = F.mse_loss(pt @ pt.T, t @ t.T)
                zz = norm(z[:, :d]); ptd, pld = pt.detach(), pl.detach()
                lc = (1 - (zz * ptd).sum(-1)).mean()
                lk = mkl(zz @ pld.T * ls_h.exp(), tl)
                lr = F.mse_loss(zz @ zz.T, t @ t.T)
                loss = loss + lp + lp_rel + a.w_cos * lc + a.w_kl * lk + a.w_rel * lr
                logs[d] = (lp.item(), lc.item(), lk.item())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
            with torch.no_grad(): ls_p.clamp_(0, 4.6); ls_h.clamp_(0, 4.6)
        # validation: cosine to projected teacher target, first view
        h.eval()
        with torch.no_grad():
            vi = torch.from_numpy(np.sort(val)); xb = X[views[0]][vi].to(dev).float(); t = T[vi].to(dev)
            cos = {d: (norm(h(xb)[:, :d]) * norm(P(t)[:, :d])).sum(-1).mean().item() for d in DIMS}
        print(f'ep {ep} loss {loss.item():.3f} P-kl/cos/h-kl128 {logs[128]} val cos128 {cos[128]:.3f} cos64 {cos[64]:.3f} '
              f'{time.time() - t0:.0f}s', flush=True)
    torch.save(dict(P=P.state_dict(), h=h.state_dict(), ls_p=ls_p.item(), ls_h=ls_h.item(), args=vars(a),
                    Db=Db, Dt=Dt), f'{run}/head.pt')
    # label table in 128-d (FP16) for this run
    with torch.no_grad():
        LB = norm(P(Lt)).cpu().numpy().astype(np.float16)
    np.save(f'{run}/labels128.npy', LB)
    print('saved', run, flush=True)


if __name__ == '__main__':
    main()
