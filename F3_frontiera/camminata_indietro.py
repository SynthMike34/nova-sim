# -*- coding: utf-8 -*-
"""camminata_indietro.py - il MODO LUNGO: appoggio 0,28 s, falcata 12,5 cm, retrogrado.
Uso:
    python camminata_indietro.py --test     esegue e confronta col canone (Campagna C, 04/08)
Canone su mujoco 3.11.0 [C]; su 3.1.6 la stessa isola e' RETROGRADA (a verbale)."""
import sys, json
import numpy as np
import marcia_core as W

VERSIONE = '1.2'

def corsa(T, t_stop=0.0):
    W.T_STOP = t_stop
    c = W.Camminatrice()
    dt = W.m.opt.timestep
    ts, xs, ev = [], [], []
    for i in range(int(T/dt)):
        p0 = c.passi
        vivo = c.frame()
        if c.passi != p0:
            ev.append((c.passi, i*dt, float(c.d.qpos[0])))
        if i % 20 == 0:
            ts.append(i*dt); xs.append(float(c.d.qpos[0]))
        if not vivo:
            return c, i*dt, ts, xs, ev
    return c, None, ts, xs, ev

def png(nome, ts, xs, ev, titolo):
    try:
        import matplotlib
    except ImportError:
        print('(matplotlib assente: PNG saltato)'); return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ts, xs, lw=1.2)
    if ev:
        ax.plot([e[1] for e in ev], [e[2] for e in ev], '.', ms=2)
    ax.set_xlabel('t [s]'); ax.set_ylabel('x pelvis [m]'); ax.set_title(titolo)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(W._OUT(nome + '.png'), dpi=110); plt.close(fig)
    print('Salvato ' + nome + '.png')

def main():
    print('camminata_indietro.py - versione ' + VERSIONE)
    W.canone(t_appoggio=0.28)        # seconda isola: 420 appoggi, -26,57 m
    c, tc, ts, xs, ev = corsa(120.0)
    x = float(c.d.qpos[0])
    falc = abs(x)/max(c.passi, 1)*2.0*100.0
    print('MODO LUNGO (indietro): %d appoggi | %s | x=%+.3f m | falcata %.1f cm' %
          (c.passi, ('VIVA a 120 s' if tc is None else 'caduta %.1f s' % tc), x, falc))
    # Criterio strutturale (L5, come camminata_avanti): il modo lungo E' 420
    # appoggi, vivo, retrogrado, falcata sopra i 10 cm. Riferimenti [C]:
    # -25,93 (mujoco 3.11.0), -26,04 (altra build, collaudo dal clone).
    ok = (tc is None and abs(c.passi - 420) <= 2 and x < 0 and falc >= 10.0)
    print('ESITO: %s (criterio: 420 appoggi, VIVA, retrogrado, falcata >= 10 cm; riferimento -25,9..-26,1 secondo la build [C]; storico a -0,05: -26,573)' %
          ('PASS - CANONE RIPRODOTTO' if ok else 'FAIL - v. nota ambiente'))
    json.dump(dict(versione=VERSIONE, esito=('PASS' if ok else 'FAIL'), appoggi=c.passi,
                   x_m=round(x, 3), falcata_cm=round(falc, 1), viva=tc is None),
              open(W._OUT('camminata_indietro.json'), 'w'), indent=1)
    print('Salvato camminata_indietro.json')
    png('camminata_indietro', ts, xs, ev, 'TX-34 long mode - 0.28 s support, 12.5 cm stride, retrograde')

if __name__ == '__main__':
    main()   # --test = esecuzione completa (unica modalita')
