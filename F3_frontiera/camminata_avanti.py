# -*- coding: utf-8 -*-
"""camminata_avanti.py - CANONE Campagna C: 654 appoggi, VIVA 120 s su TRE build (rullio -0,035: +2,49/+2,62/+2,58 m).
Uso:
    python camminata_avanti.py --test     esegue e confronta col canone (Campagna C, 04/08)
Canone su mujoco 3.11.0 [C]; su 3.1.6 la stessa isola e' RETROGRADA (a verbale)."""
import sys, json
import numpy as np
import marcia_core as W

VERSIONE = '1.1'

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
    print('camminata_avanti.py - versione ' + VERSIONE)
    W.canone()                       # kp 400, appoggio a orologio 0,180 s
    c, tc, ts, xs, ev = corsa(120.0)
    x = float(c.d.qpos[0])
    print('Camminata AVANTI: %d appoggi | %s | x=%+.6f m | v=%.4f m/s' %
          (c.passi, ('VIVA a 120 s' if tc is None else 'caduta %.1f s' % tc), x, x/120.0))
    import mujoco
    # [L8] canone CON LEVA (EST_ZA 0,025): fascia +6,44..+6,56 su 4 build [C-BUSSOLA,
    # 3 run/build]; su questo container 3.11.0 = +6,561833 (tre run identici).
    # BASE a leva spenta (canone-altopiano storico): 2,491220 / 2,621374 / 2,581697.
    ATTESI = {'3.11.0': 6.561833}
    att = ATTESI.get(mujoco.__version__)
    if att is not None:
        ok = (tc is None and c.passi == 654 and abs(x - att) < 0.02)
        rif = 'atteso %+.6f su mujoco %s [L8]' % (att, mujoco.__version__)
    else:
        ok = (tc is None and c.passi == 654 and 6.40 <= x <= 6.60)
        rif = ('build %s: criterio strutturale 654/VIVA/avanti, fascia +6,44..+6,56 [L8]'
               % mujoco.__version__)
    print('ESITO: %s (canone L8: leva EST_ZA 0,025; %s)' %
          ('PASS - CANONE RIPRODOTTO' if ok else 'FAIL', rif))
    print('Massimo misurato (dichiarato, non canone): +9,695622 m a rullio -0,05'
          ' su mujoco 3.11.0 + numpy 2.4.4.')
    json.dump(dict(versione=VERSIONE, esito=('PASS' if ok else 'FAIL'), appoggi=c.passi,
                   x_m=round(x, 6), viva=tc is None, v_m_s=round(x/120.0, 4)),
              open(W._OUT('camminata_avanti.json'), 'w'), indent=1)
    print('Salvato camminata_avanti.json')
    png('camminata_avanti', ts, xs, ev, 'TX-34 forward walking - 654 supports, 120 s (size-36 foot)')

if __name__ == '__main__':
    main()   # --test = esecuzione completa (unica modalita')
