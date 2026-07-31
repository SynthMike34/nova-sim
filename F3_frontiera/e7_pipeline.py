#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e7_pipeline.py v0.1 - CANTIERE FINALE: la PILA COMPLETA dell'atterraggio.
salto -> volo -> timone caviglie (E4) -> volano braccia (E5) -> arbitro con
passetto (E6) -> TOE-BRAKE (scoperta 30c) come ultimo anello. Punta UNITA
(scoperta 32), modello base. Richiede e6_supervisore.py ed e0_dita.py in
cartella (pila e iniettore calcolati in simulazione, qui solo orchestrati).

    python e7_pipeline.py            demo 3D: la pipeline completa (loop)
    python e7_pipeline.py --test     confronto nuda / toe-tutti / toe-stance

Target TERA: fermo eretta > 0,35 s dopo il salto completo.
"""
import os as _os
def _MP(_n):
    if _os.path.isabs(_n) or _os.sep in _n or '/' in _n:
        if _os.path.exists(_n): return _n
    _qui = _os.path.dirname(_os.path.abspath(__file__))
    _base = _os.path.dirname(_qui)
    for _c in (_n,
               _os.path.join(_base, 'models', _os.path.basename(_n)),
               _os.path.join(_base, 'config', _os.path.basename(_n)),
               _os.path.join(_qui, _os.path.basename(_n))):
        if _os.path.exists(_c): return _c
    return _os.path.join(_base, 'models', _os.path.basename(_n))
def _DER(_n):
    _qui = _os.path.dirname(_os.path.abspath(__file__))
    _d = _os.path.join(_os.path.dirname(_qui), 'models')
    if not _os.path.isdir(_d): _d = _qui
    return _os.path.join(_d, _os.path.basename(_n))
def _OUT(_n):
    _qui = _os.path.dirname(_os.path.abspath(__file__))
    _d = _os.path.join(_os.path.dirname(_qui), 'outputs')
    if not _os.path.isdir(_d):
        try: _os.makedirs(_d)
        except Exception: _d = _qui
    return _os.path.join(_d, _os.path.basename(_n))
import sys
import json
import importlib
import numpy as np
import mujoco

VERSIONE = '1.0'
TOE_PUSH, TOE_BRAKE, TOE_BASE = 0.45, 0.55, 0.03

import e6_supervisore as E6
from e0_dita import Iniettore

def legge_pipeline(g, solo_stance):
    def legge(lato, gruppo='toe'):
        if g.fase == 'spinta':
            u = 0.5 - 0.5*np.cos(np.pi*np.clip(
                (g.t_f - 0.6*E6.T_PUSH)/(0.4*E6.T_PUSH), 0, 1))
            return TOE_BASE + (TOE_PUSH - TOE_BASE)*u
        if g.fase == 'atterrata':
            if g.modo == 'FERMA':
                d = g.d
                com = d.subtree_com[0]
                feet = 0.5*(d.xpos[E6.BID['r_foot']][0]
                            + d.xpos[E6.BID['l_foot']][0])
                resto = float(com[0] + d.qvel[0]/g.om) - feet
                return float(np.clip(3.0*resto, 0.0, 0.60))
            if solo_stance and lato != g.stance:
                return TOE_BASE            # piede di volo pulito
            return TOE_BRAKE
        return TOE_BASE
    return legge

def prova(config):
    """config: 'nuda' | 'tutti' | 'stance'"""
    importlib.reload(E6)
    g = E6.Atterratrice()
    dt = E6.m.opt.timestep
    ctx = (Iniettore(legge_pipeline(g, config == 'stance'))
           if config != 'nuda' else _Nulla())
    t_volo, clear = 0.0, 0.0
    z0 = float(g.d.xpos[E6.BID['r_foot']][2])
    with ctx:
        for i in range(int(10.0/dt)):
            vivo = g.frame()
            if g.fase == 'volo':
                t_volo += dt
                zp = min(float(g.d.xpos[E6.BID['r_foot']][2]),
                         float(g.d.xpos[E6.BID['l_foot']][2])) - z0
                clear = max(clear, zp)
            if not vivo:
                break
    return dict(volo_ms=round(t_volo*1000, 0), quota_cm=round(clear*100, 1),
                passi=int(g.passi), assestamenti=int(g.assestamenti),
                fermo_s=round(float(getattr(g, 'fermo_max', g.t_fermo)), 2),
                in_piedi=bool(g.d.qpos[2] > 0.85 and not g.caduta))

class _Nulla:
    def __enter__(self): return self
    def __exit__(self, *a): return False

def campagna():
    print('e7_pipeline.py - versione ' + VERSIONE)
    print('PILA COMPLETA: salto -> timone -> volano -> arbitro -> toe-brake.')
    ris = {}
    for cfg, eti in (('nuda', 'pila NUDA (E6)'),
                     ('tutti', '+ toe-brake su entrambi'),
                     ('stance', '+ toe-brake SOLO-STANCE')):
        ris[cfg] = prova(cfg)
        r = ris[cfg]
        print('%-26s: volo %3.0f ms | passi %d+%d | fermo %.2f s | eretta: %s'
              % (eti, r['volo_ms'], r['passi'], r['assestamenti'],
                 r['fermo_s'], r['in_piedi']))
    best = max(('tutti', 'stance'), key=lambda k: ris[k]['fermo_s'])
    guadagno = ris[best]['fermo_s'] - ris['nuda']['fermo_s']
    ok = ris[best]['fermo_s'] > 0.35
    print('MIGLIORE: %s -> fermo %.2f s (nuda %.2f, guadagno %+.2f) | '
          'target >0,35 s: %s'
          % (best, ris[best]['fermo_s'], ris['nuda']['fermo_s'],
             guadagno, 'SUPERATO' if ok else 'NON superato'))
    esito = ('PASS (pipeline > 0,35 s)' if ok else
             'MISURATO (guadagno %+.2f s; target 0,35 non superato)' % guadagno)
    print('ESITO E7: ' + esito)
    json.dump(dict(versione=VERSIONE, esito=esito, migliore=best, **ris),
              open(_OUT('e7_pipeline.json'), 'w'), indent=1)
    print('Salvato e7_pipeline.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.5))
    eti = ['pila\nNUDA', 'toe\nENTRAMBI', 'toe\nSOLO-STANCE']
    vals = [ris['nuda']['fermo_s'], ris['tutti']['fermo_s'],
            ris['stance']['fermo_s']]
    ax.bar(eti, vals, color=['tab:gray', 'tab:green', 'tab:purple'], width=0.55)
    for x, v in enumerate(vals):
        ax.text(x, v, '%.2f s' % v, ha='center', va='bottom',
                fontsize=14, fontweight='bold')
    ax.axhline(0.35, color='r', ls='--', lw=1.5, label='target TERA 0,35 s')
    ax.set_ylabel('time to upright rest after jump [s]', fontsize=12)
    ax.set_title('E7 landing stack - claim withdrawn at 66.23 kg',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(_OUT('e7_pipeline.png'), dpi=150)
    print('Salvato e7_pipeline.png')

def demo():
    import time
    import mujoco.viewer
    print(__doc__)
    importlib.reload(E6)
    g = E6.Atterratrice()
    dt = E6.m.opt.timestep
    with Iniettore(legge_pipeline(g, True)):
        t_tot = 0.0
        with mujoco.viewer.launch_passive(E6.m, g.d) as v:
            while v.is_running():
                t0 = time.time()
                vivo = g.frame()
                t_tot += dt
                if not vivo or t_tot > 10.0:
                    if vivo and g.t_fermo > 0.3:
                        print('fermata: %.2f s (%d passi + %d assestamenti)'
                              % (g.t_fermo, g.passi, g.assestamenti))
                    g.reset()
                    t_tot = 0.0
                v.sync()
                resto = dt - (time.time() - t0)
                if resto > 0:
                    time.sleep(resto)

if __name__ == '__main__':
    if '--test' in sys.argv:
        campagna()
    else:
        demo()
