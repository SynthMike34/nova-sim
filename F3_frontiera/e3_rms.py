#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e3_rms.py — RMS e duty anca roll sulla camminata E3 a cadenza reale
(masse definitive 66,23 kg). Il dato per il modello termico di SESTANTE.
Metodo: monkeypatch di mujoco.mj_step (registra t + coppie hip roll)
attorno alla prova E3 in avanti (v_des target).

    python e3_rms.py           analisi completa
    python e3_rms.py --test    verifica (RMS>0, duty in (0,1))
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
import sys
import os
import numpy as np

VERSIONE = '1.4'
SOGLIA = 43.0
_QUI = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_QUI)
sys.path.insert(0, _QUI)
sys.path.insert(0, os.path.join(_BASE, 'core'))
import mujoco

T, FR, FL = [], [], []
_idx = {}
_vero_step = mujoco.mj_step


def _step_registrato(m, d, *a, **k):
    _vero_step(m, d, *a, **k)
    if not _idx:
        for i in range(m.nu):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i][0])
            if n in ('r_hip_roll', 'l_hip_roll'):
                _idx[n] = i
    T.append(float(d.time))
    FR.append(float(d.actuator_force[_idx['r_hip_roll']]))
    FL.append(float(d.actuator_force[_idx['l_hip_roll']]))


def main():
    test = '--test' in sys.argv
    print('e3_rms VERSIONE %s — soglia %.0f Nm — E3 a cadenza reale' % (VERSIONE, SOGLIA))
    mujoco.mj_step = _step_registrato
    import e3_accoppiato as E3
    r = E3.prova(T=30.0)   # produzione: P0 come manopola (config campagna)
    mujoco.mj_step = _vero_step
    print('E3 prova:', r if not isinstance(r, dict) else
          {k: v for k, v in r.items() if not isinstance(v, (list, dict))})
    t = np.array(T)
    dt = np.gradient(t)
    # v1.4: FINESTRA DI REGIME (Sestante): il run include ~2,3 s di quiete/
    # assestamento che diluiscono il denominatore. Inizio marcia = primo
    # campione con |coppia R| > 40 Nm (primo evento di presa), dichiarato.
    _Fr0 = np.abs(np.array(FR))
    _i0 = int(np.argmax(_Fr0 > 40.0))
    T0_REGIME = float(t[_i0])
    print('finestra di regime dichiarata: [%.2f, %.2f] s (inizio = primo F>40)' % (T0_REGIME, t[-1]))
    print('registrati %d campioni, %.1f s' % (len(t), t[-1] - t[0]))
    parete = None
    _mreg = t >= T0_REGIME
    t, dt = t[_mreg], dt[_mreg]
    for nome, F in (('R', np.abs(np.array(FR))[_mreg]), ('L', np.abs(np.array(FL))[_mreg])):
        rms = float(np.sqrt(np.sum(F*F*dt)/np.sum(dt)))
        sopra = F > SOGLIA
        duty = float(np.sum(dt[sopra])/np.sum(dt))
        media_s = float(F[sopra].mean()) if sopra.any() else 0.0
        print(' anca roll %s: RMS %.1f Nm | picco %.1f | duty>%.0f: %.1f%% | '
              'media sopra %.1f Nm' % (nome, rms, F.max(), SOGLIA, 100*duty, media_s))
        if nome == 'R':
            out = (rms, duty, float(F.max()))
            parete = float(np.sum(dt[F >= 79.5]))
            print(' tempo a parete >=79,5 Nm: %.0f ms totali' % (1000*parete))
            sotto = F < 70.0  # (finestra regime)
            rms70 = float(np.sqrt(np.sum(F[sotto]**2*dt[sotto])/np.sum(dt[sotto])))
            fr70 = 100*float(np.sum(dt[~sotto])/np.sum(dt))
            print('Run: %.1f s · RMS <70 Nm: %.1f Nm · Frazione >=70 Nm: %.1f%%'
                  % (t[-1]-t[0], rms70, fr70))
    # T_sw effettivo: dalla lista passi di E3 (fonte vera; lo zero-crossing
    # v1.0 misurava le oscillazioni della coppia, non i passi: scartato)
    if isinstance(r, tuple) and len(r) >= 4 and isinstance(r[3], list) and r[3]:
        print(' T_sw effettivo mediano (da E3): %.3f s (n=%d passi)'
              % (float(np.median(r[3])), len(r[3])))
    if test:
        ok = out[0] > 0 and 0.0 <= out[1] < 1.0 and len(t) > 400
        print('TEST ' + ('PASS' if ok else 'FAIL'))
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
