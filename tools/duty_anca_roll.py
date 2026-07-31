#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
duty_anca_roll.py - Per SESTANTE (modello termico anca roll).
Duty cycle dell'anca roll sopra la coppia nominale continuativa
dell'RMD-X8 (43 Nm) durante la camminata campione (2 passi).

    python duty_anca_roll.py           analisi + grafico duty_anca_roll.png
    python duty_anca_roll.py --test    verifica pass/fail senza grafico

Config campione IDENTICA a metriche_coppie.py (riga 39):
step=0.14, t_sw=0.6, lift=0.04, y_in=0.0, lean=0.15, n_passi=2.
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
import os
import numpy as np

VERSIONE = '1.3'
SOGLIA = 43.0
SOGLIA_PRESA = 40.0   # inizio finestra di regime (come e3_rms)          # Nm - coppia nominale continuativa RMD-X8 (datasheet)
BASE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))


def esegui():
    sys.path.insert(0, os.path.join(BASE, 'core'))
    pass  # percorsi risolti da _MP
    import mujoco
    from gait_core import gait

    T, F = [], []
    nomi_box = {}

    def rec(t, d):
        T.append(t)
        F.append(d.actuator_force.copy())

    # config CAMPIONE (identica a metriche_coppie.py)
    passi, x, caduta, m_o_d = gait(step=0.14, t_sw=0.6, lift=0.04, y_in=0.02,   # CAMPIONE v2 @66,23 kg
                                   lean=0.15, n_passi=2, recorder=rec)
    m = m_o_d if hasattr(m_o_d, 'actuator_trnid') else None
    if m is None:
        # ricarico il modello solo per i nomi attuatori
        m = mujoco.MjModel.from_xml_path(_MP('tx34_v1.xml'))
        total_mass = m.body_mass.sum()
        print(f"[MASSA MODELLO] {total_mass:.3f} kg")
        assert 65.5 < total_mass < 67.0, f"MASSA FUORI RANGE: {total_mass:.3f} kg - verifica XML"
    nomi = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT,
                              m.actuator_trnid[i][0]) for i in range(m.nu)]
    T = np.array(T)
    F = np.array(F)

    # FINESTRA DI REGIME - il run contiene una fase di quiete/assestamento
    # iniziale che diluisce il denominatore e rende il duty ottimistico.
    # Inizio = primo evento di presa oltre SOGLIA_PRESA su UNO QUALSIASI dei due
    # lati: nel campione la presa parte a sinistra, quindi il solo lato R
    # (convenzione di e3_rms su E3) qui farebbe partire la finestra alla fine.
    # Duty e picco sono riportati come CASO PEGGIORE, con il lato dichiarato.
    jr = nomi.index('r_hip_roll')
    jl = nomi.index('l_hip_roll')
    _pres = np.maximum(np.abs(F[:, jr]), np.abs(F[:, jl])) > SOGLIA_PRESA
    i0 = int(np.argmax(_pres)) if _pres.any() else 0
    T_full = T
    T = T[i0:]
    F = F[i0:]

    out = {'passi': passi, 'caduta': bool(caduta), 't_tot': float(T[-1] - T[0]),
           'finestra_regime_s': [round(float(T[0]), 2), round(float(T[-1]), 2)],
           't_run_intero_s': round(float(T_full[-1] - T_full[0]), 2)}
    out['t_ciclo_passo'] = out['t_tot'] / 2.0  # 2 passi campione

    for lato in ('r', 'l'):
        j = nomi.index(lato + '_hip_roll')
        tau = np.abs(F[:, j])
        dt = np.gradient(T)
        sopra = tau > SOGLIA
        t_sopra = float(np.sum(dt[sopra]))
        out[lato] = {
            'picco_Nm': float(tau.max()),
            'frazione_sopra': t_sopra / out['t_tot'],
            'media_sopra_Nm': float(tau[sopra].mean()) if sopra.any() else 0.0,
            's_sopra_per_passo': t_sopra / 2.0,
            'tau': tau,
        }
    return out, T


def main():
    test = '--test' in sys.argv
    print('duty_anca_roll VERSIONE ' + VERSIONE + ' - soglia %.0f Nm' % SOGLIA)
    out, T = esegui()
    print('camminata campione: %d passi, caduta=%s, t_tot=%.2f s, '
          'ciclo di passo=%.2f s' % (out['passi'], out['caduta'],
                                     out['t_tot'], out['t_ciclo_passo']))
    for lato in ('r', 'l'):
        o = out[lato]
        print(' anca roll %s: picco %.1f Nm | sopra %.0f Nm per %.1f%% del '
              'ciclo | media (quando sopra) %.1f Nm | %.2f s per passo'
              % (lato.upper(), o['picco_Nm'], SOGLIA,
                 100 * o['frazione_sopra'], o['media_sopra_Nm'],
                 o['s_sopra_per_passo']))
    peggio = max(('r', 'l'), key=lambda k: out[k]['frazione_sopra'])
    pk_lato = max(('r', 'l'), key=lambda k: out[k]['picco_Nm'])
    o = out[peggio]
    print('finestra di regime: [%.2f, %.2f] s su un run di %.2f s'
          % (out['finestra_regime_s'][0], out['finestra_regime_s'][1],
             out['t_run_intero_s']))
    print('DUTY PEGGIORE (%s): %.1f%% del ciclo sopra soglia | media %.1f Nm '
          '· %.2f s/passo' % (peggio.upper(), 100 * o['frazione_sopra'],
                              o['media_sopra_Nm'], o['s_sopra_per_passo']))
    print('PICCO PEGGIORE (%s): %.1f Nm' % (pk_lato.upper(), out[pk_lato]['picco_Nm']))

    if test:
        ok = (not out['caduta'] and out['passi'] == 2
              and 66.0 < max(out['r']['picco_Nm'], out['l']['picco_Nm']) < 74.0
              and 0.0 < o['frazione_sopra'] < 1.0)
        print('TEST ' + ('PASS' if ok else 'FAIL'))
        sys.exit(0 if ok else 1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for lato, col in (('r', '#1f77b4'), ('l', '#2ca02c')):
        ax.plot(T, out[lato]['tau'], color=col, lw=1.1,
                label='hip roll |torque| ' + lato.upper())
    ax.axhline(SOGLIA, color='crimson', ls='--', lw=1.4,
               label='X8 continuous rating = %.0f N·m' % SOGLIA)
    tp = out[peggio]['tau']
    ax.fill_between(T, SOGLIA, tp, where=tp > SOGLIA,
                    color='crimson', alpha=0.25)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('|torque| [N·m]')
    ax.set_title('Hip roll duty - sample gait, steady-state window: '
                 '%.1f%% of cycle above %d N·m · worst peak %.1f N·m (side %s)'
                 % (100 * o['frazione_sopra'], int(SOGLIA),
                    out[pk_lato]['picco_Nm'], pk_lato.upper()),
                 fontsize=11, fontweight='bold')
    ax.set_xlim(T[0], T[-1])
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    dest = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'outputs', 'duty_anca_roll.png')
    fig.savefig(dest, dpi=110)
    print('grafico: ' + dest)


if __name__ == '__main__':
    main()
