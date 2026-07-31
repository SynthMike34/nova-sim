#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e3_termico.py - figura del bilancio termico dell'anca roll su E3 a cadenza reale.
Rigenera outputs/e3_termico.png dai dati registrati da e3_rms: nessun valore
scritto a mano, tutto letto dall'esecuzione.

    python e3_termico.py           genera la figura
    python e3_termico.py --test    genera e verifica i valori (RMS>0, frazioni in (0,1))
"""
import os as _os


def _OUT(_n):
    _qui = _os.path.dirname(_os.path.abspath(__file__))
    _d = _os.path.join(_os.path.dirname(_qui), 'outputs')
    if not _os.path.isdir(_d):
        try:
            _os.makedirs(_d)
        except Exception:
            _d = _qui
    return _os.path.join(_d, _os.path.basename(_n))


import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VERSIONE = '1.0'
NOMINALE = 43.0      # X8-P20, coppia continuativa dichiarata dal costruttore
CAP = 80.0           # limitatore di progetto sul giunto
SOGLIA_TRANS = 70.0  # confine posturale / transitorio di contatto


def main():
    test = '--test' in sys.argv
    print('e3_termico.py - versione ' + VERSIONE)

    _qui = _os.path.dirname(_os.path.abspath(__file__))
    sys.path.insert(0, _qui)
    sys.path.insert(0, _os.path.join(_os.path.dirname(_qui), 'core'))

    import e3_rms                      # riusa la registrazione, non la duplica
    e3_rms.main.__globals__['__name__'] = 'e3_rms'
    try:
        e3_rms.main()
    except SystemExit:
        pass

    t = np.array(e3_rms.T)
    F = np.abs(np.array(e3_rms.FR))     # lato R, come in e3_rms
    if t.size < 100:
        print('DATI INSUFFICIENTI'); sys.exit(1)

    # finestra di regime: dal primo evento di presa oltre 40 N*m
    i0 = int(np.argmax(F > 40.0))
    t, F = t[i0:], F[i0:]
    dt = np.gradient(t)

    rms = float(np.sqrt(np.sum(F * F * dt) / np.sum(dt)))
    sopra = F >= SOGLIA_TRANS
    fr_tempo = float(np.sum(dt[sopra]) / np.sum(dt))
    en_tot = float(np.sum(F * F * dt))
    fr_calore = float(np.sum(F[sopra] ** 2 * dt[sopra]) / en_tot)
    post = float(np.sqrt(np.sum(F[~sopra] ** 2 * dt[~sopra]) / np.sum(dt[~sopra])))

    print(' steady-state window [%.2f, %.2f] s' % (t[0], t[-1]))
    print(' total RMS      %.1f N*m = %.0f%% of rated' % (rms, 100 * rms / NOMINALE))
    print(' postural RMS   %.1f N*m = %.0f%% of rated (%.1f%% of cycle)'
          % (post, 100 * post / NOMINALE, 100 * (1 - fr_tempo)))
    print(' transients     %.1f%% of cycle -> %.0f%% of dissipation'
          % (100 * fr_tempo, 100 * fr_calore))

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(t, F, color='#1f77b4', lw=0.9)
    ax.fill_between(t, SOGLIA_TRANS, np.minimum(F, CAP), where=sopra,
                    color='#d62728', alpha=0.22, linewidth=0)
    ax.axhline(CAP, color='#d62728', ls='--', lw=1.2)
    ax.axhline(NOMINALE, color='#ff7f0e', ls='--', lw=1.4)
    ax.text(t[-1], CAP + 1.2, '%.0f cap' % CAP, ha='right', va='bottom',
            color='#d62728', fontsize=9.5)
    ax.text(t[-1], NOMINALE + 1.2, '%.0f rated (X8)' % NOMINALE, ha='right',
            va='bottom', color='#ff7f0e', fontsize=9.5)
    ax.annotate('contact transients - %.1f%% of cycle' % (100 * fr_tempo),
                xy=(0.012, 0.94), xycoords='axes fraction', fontsize=11,
                fontweight='bold', color='#b02020', va='top')
    ax.annotate('-> %.0f%% of dissipation' % (100 * fr_calore),
                xy=(0.012, 0.855), xycoords='axes fraction', fontsize=11,
                color='#b02020', va='top')
    ax.set_xlabel('t [s] - steady-state window (start = first grab > 40 N·m)')
    ax.set_ylabel('hip roll torque [N·m]')
    ax.set_title('E3 walking @66.23 kg - RMS %.1f N·m (%.0f%% of rated) [C]'
                 % (rms, 100 * rms / NOMINALE), fontsize=12, fontweight='bold')
    ax.set_ylim(0, CAP * 1.12)
    ax.grid(alpha=0.25)
    fig.text(0.012, 0.012,
             'postural %.1f N·m = %.0f%% of rated (RMS outside contact transients)'
             % (post, 100 * post / NOMINALE), fontsize=9, style='italic', color='#555')
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(_OUT('e3_termico.png'), dpi=150)
    print('Salvato e3_termico.png')

    json.dump(dict(versione=VERSIONE, rms_Nm=round(rms, 1),
                   posturale_Nm=round(post, 1),
                   transitori_tempo_pct=round(100 * fr_tempo, 1),
                   transitori_calore_pct=round(100 * fr_calore, 0),
                   nominale_Nm=NOMINALE, cap_Nm=CAP,
                   finestra_s=[round(float(t[0]), 2), round(float(t[-1]), 2)]),
              open(_OUT('e3_termico.json'), 'w'), indent=1)

    if test:
        ok = rms > 0 and 0 < fr_tempo < 1 and 0 < fr_calore < 1 and post > 0
        print('TEST ' + ('PASS' if ok else 'FAIL'))
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
