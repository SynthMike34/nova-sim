#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metriche_coppie.py — Categoria J, modulo 1: coppie richieste durante la
camminata (2 passi) contro le coppie disponibili degli attuatori del TDD.

    python metriche_coppie.py            grafico + tabella + metriche.json
    python metriche_coppie.py --test     verifica pass/fail senza grafico

Coppie disponibili (TDD par.2.3 / joint_limits_v2): anca pitch e ginocchio
120 Nm (RMD-X12); anca yaw/roll e caviglia 80 Nm (RMD-X8); vita 30 Nm.
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
import json
import numpy as np

VERSIONE = '1.2'
LIMITI = {}
for lato in ('r', 'l'):
    LIMITI[lato + '_hip_pitch'] = 120.0
    LIMITI[lato + '_knee'] = 120.0
    LIMITI[lato + '_hip_yaw'] = 80.0
    LIMITI[lato + '_hip_roll'] = 80.0
    LIMITI[lato + '_ankle_pitch'] = 80.0
    LIMITI[lato + '_ankle_roll'] = 80.0
LIMITI['waist_yaw'] = 30.0

def main():
    import mujoco
    from gait_core import gait
    m = mujoco.MjModel.from_xml_path(_MP('tx34_v1.xml'))
    total_mass = m.body_mass.sum()
    print(f"[MASSA MODELLO] {total_mass:.3f} kg")
    assert 65.5 < total_mass < 67.0, f"MASSA FUORI RANGE: {total_mass:.3f} kg — verifica XML"
    nomi = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i][0])
            for i in range(m.nu)]
    T, F, V = [], [], []
    def rec(t, d):
        T.append(t)
        F.append(d.actuator_force.copy())
        V.append(d.actuator_velocity.copy())
    passi, x, caduta, _ = gait(step=0.14, t_sw=0.6, lift=0.04, y_in=0.02,   # CAMPIONE v2 @66,23 kg
                               lean=0.15, n_passi=2, recorder=rec)
    T = np.array(T); F = np.array(F); V = np.array(V)
    dt = T[1] - T[0]
    print('metriche_coppie.py - versione ' + VERSIONE)
    print('Camminata registrata: %d passi, avanzamento %+.2f m, %s campioni, caduta: %s'
          % (passi, x, len(T), 'no' if caduta is None else ('%.1f s' % caduta)))

    tab = []
    for j, nome in enumerate(nomi):
        if nome not in LIMITI:
            continue
        picco = float(np.max(np.abs(F[:, j])))
        rms = float(np.sqrt(np.mean(F[:, j]**2)))
        lim = LIMITI[nome]
        tab.append(dict(giunto=nome, picco_Nm=round(picco, 1),
                        rms_Nm=round(rms, 1), limite_Nm=lim,
                        margine_pct=round((1 - picco/lim)*100, 1)))
    tab.sort(key=lambda r: r['margine_pct'])
    print('%-15s %9s %8s %9s %9s' % ('giunto', 'picco Nm', 'RMS Nm', 'limite', 'margine'))
    ok = True
    for r in tab:
        print('%-15s %9.1f %8.1f %9.0f %8.1f%%' %
              (r['giunto'], r['picco_Nm'], r['rms_Nm'], r['limite_Nm'], r['margine_pct']))
        if r['picco_Nm'] >= r['limite_Nm']:
            ok = False

    # energia meccanica: integrale di somma |tau*omega| sui giunti gamba+vita
    idx = [j for j, n in enumerate(nomi) if n in LIMITI]
    P = np.sum(np.abs(F[:, idx] * V[:, idx]), axis=1)
    trap = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
    E = float(trap(P, dx=dt))
    print('Energia meccanica totale (2 passi, %.1f s): %.1f J -> %.1f J/passo; potenza media %.1f W'
          % (T[-1]-T[0], E, E/max(passi, 1), E/(T[-1]-T[0])))
    print('[A] Il consumo ELETTRICO (perdite I2R nel mantenimento di coppia) richiede'
          ' Kt e R dei motori: PROPOSTA TDD -> aggiungerli al par.2.3.')

    json.dump(dict(versione=VERSIONE, passi=passi, avanzamento_m=round(x, 3),
                   energia_mecc_J=round(E, 1), tabella=tab),
              open('metriche_coppie.json', 'w'), indent=1)
    print('Salvato metriche_coppie.json')

    if '--test' in sys.argv:
        print('ESITO: %s' % ('DIMENSIONAMENTO BOM VERIFICATO - tutti i picchi sotto i limiti'
                             if ok else 'ATTENZIONE: picco oltre il limite!'))
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    coppie_plot = [('hip_pitch', 120, 'Anca (pitch) - RMD-X12'),
                   ('knee', 120, 'Ginocchio - RMD-X12'),
                   ('ankle_pitch', 80, 'Caviglia (pitch) - RMD-X8')]
    for ax, (base, lim, titolo) in zip(axs, coppie_plot):
        for lato, colore in (('r', 'tab:blue'), ('l', 'tab:green')):
            j = nomi.index(lato + '_' + base)
            ax.plot(T, F[:, j], color=colore, lw=1,
                    label=('destra' if lato == 'r' else 'sinistra'))
        ax.axhline(lim, color='r', ls='--', lw=1)
        ax.axhline(-lim, color='r', ls='--', lw=1, label='limite attuatore')
        ax.set_ylabel('coppia [Nm]')
        ax.set_title(titolo, fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
    axs[-1].set_xlabel('tempo [s]')
    fig.suptitle('TX-34 - coppie richieste nella camminata (2 passi) vs coppie RMD disponibili')
    fig.tight_layout()
    fig.savefig('coppie_camminata_v1.png', dpi=150)
    print('Salvato coppie_camminata_v1.png')

if __name__ == '__main__':
    main()
