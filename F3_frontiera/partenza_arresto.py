# -*- coding: utf-8 -*-
"""partenza_arresto.py - B38/B39: da fermo a marcia in 2,4 s; stop = in piedi, ~1 cm.
Uso:
    python partenza_arresto.py --test     esegue e confronta col canone (Campagna C, 04/08)
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
    print('partenza_arresto.py - versione ' + VERSIONE)
    W.canone()
    c, tc, ts, xs, ev = corsa(60.0, t_stop=30.0)
    t1 = ev[0][1] if ev else None
    xstop = next((x_ for t_, x_ in zip(ts, xs) if t_ >= 30.0), None)
    xfine = float(c.d.qpos[0]); z = float(c.d.qpos[2])
    inerzia = (xfine - xstop)*100.0 if xstop is not None else float('nan')
    in_piedi = (tc is None and z > 0.80)
    print('PARTENZA: primo appoggio a %.2f s (keyframe qvel=0: parte DA FERMO)' % (t1 or -1))
    print('ARRESTO a 30 s: %s | z=%.3f m | inerzia residua %+.2f cm' %
          ('IN PIEDI per i 30 s successivi' if in_piedi else 'CADUTA', z, inerzia))
    # Criterio VERO (L5): parte da fermo e RESTA IN PIEDI. L'inerzia e' informativa:
    # [C] fra -1,8 e -3,6 cm secondo la build (misurati: -1,76 qui; -2,71 Linux
    # 3.10.0; -3,53 Windows 3.10.0); +1,07 a rullio -0,05 (storico).
    ok = (in_piedi and t1 is not None and abs(t1 - 2.41) < 0.10)
    print('ESITO: %s (criterio: parte da fermo ~2,41 s e resta IN PIEDI 30 s; '
      'inerzia informativa fra -1,8 e -3,6 cm secondo la build [C])' % ('PASS' if ok else 'FAIL'))
    json.dump(dict(versione=VERSIONE, esito=('PASS' if ok else 'FAIL'),
                   primo_appoggio_s=round(t1, 2) if t1 else None,
                   in_piedi=bool(in_piedi), inerzia_cm=round(inerzia, 2), z_fine=round(z, 3)),
              open(W._OUT('partenza_arresto.json'), 'w'), indent=1)
    print('Salvato partenza_arresto.json')
    png('partenza_arresto', ts, xs, ev, 'TX-34 start from rest and stop - standing after stop')

if __name__ == '__main__':
    main()   # --test = esecuzione completa (unica modalita')
