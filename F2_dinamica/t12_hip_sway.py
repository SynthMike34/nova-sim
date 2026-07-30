#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t12_hip_sway.py v1.1 - Test T12 (TDD par.12.2): hip sway, oscillazione del
bacino +/-8..12 gradi a 0,8-1,0 Hz mantenendo l'equilibrio. (Acustica <42 dB:
fuori dominio simulazione.)

    python t12_hip_sway.py           demo 3D: NOVA ancheggia alla config certificata
    python t12_hip_sway.py --test    campagna di misura headless: tabella+PNG+JSON

Interpretazione [A]: "oscillazione bacino" = rollio del bacino attorno all'asse
di marcia, generato dalle anche (roll) con piedi a terra, equilibrio di caviglia.
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
import mujoco

VERSIONE = '1.2'
CONFIG_CERT = (14.0, 0.9)    # (comando anche [gradi], frequenza [Hz]) - vedi tabella --test
XML = _MP('tx34_v1.xml')
K1, K2, XB = 4.0, 0.8, 0.05
LIMITE_ROLL = 80.0           # Nm, RMD-X8 (TDD par.2.3)

m = mujoco.MjModel.from_xml_path(XML)
total_mass = m.body_mass.sum()
print(f"[MASSA MODELLO] {total_mass:.3f} kg")
assert 65.5 < total_mass < 67.0, f"MASSA FUORI RANGE: {total_mass:.3f} kg — verifica XML"
NOMI = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i][0])
        for i in range(m.nu)]
AID = dict((n, i) for i, n in enumerate(NOMI))
FID = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ('r_foot', 'l_foot')]

def controlla(d, cp, t, amp, f_hz):
    """Applica equilibrio + sway; ritorna il nuovo CoM per il ciclo successivo."""
    com = d.subtree_com[0]
    v = (com - cp) / m.opt.timestep
    feet = 0.5 * (d.xpos[FID[0]] + d.xpos[FID[1]])
    ap = float(np.clip(K1*(com[0]-feet[0]-XB) + K2*v[0], -0.45, 0.45))
    ar = float(np.clip(-(K1*(com[1]-feet[1]) + K2*v[1]), -0.20, 0.20))
    sway = amp * np.sin(2.0*np.pi*f_hz*t) * min(t/2.0, 1.0)   # rampa 2 s
    for j in ('r_ankle_pitch', 'l_ankle_pitch'):
        d.ctrl[AID[j]] = ap
    for j in ('r_ankle_roll', 'l_ankle_roll'):
        d.ctrl[AID[j]] = ar
    for j in ('r_hip_roll', 'l_hip_roll'):
        d.ctrl[AID[j]] = sway
    return com.copy()

def esegui(amp_deg, f_hz, T=20.0, registra=False):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    cp = d.subtree_com[0].copy()
    amp = np.radians(amp_deg)
    tr_t, tr_phi, tr_fr, tr_fl, tr_cy = [], [], [], [], []
    for i in range(int(T / dt)):
        t = i * dt
        cp = controlla(d, cp, t, amp, f_hz)
        mujoco.mj_step(m, d)
        qw, qx, qy, qz = d.qpos[3:7]
        phi = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))
        tr_t.append(t)
        tr_phi.append(np.degrees(phi))
        tr_fr.append(d.actuator_force[AID['r_hip_roll']])
        tr_fl.append(d.actuator_force[AID['l_hip_roll']])
        tr_cy.append(d.subtree_com[0][1]*1000.0)
        if d.qpos[2] < 0.45:
            return dict(caduta=True, t_caduta=round(t, 1),
                        amp_mis=0.0, picco_Nm=0.0, cy_mm=0.0)
    tr_phi = np.array(tr_phi)
    meta = len(tr_phi)//2
    amp_mis = 0.5*(tr_phi[meta:].max() - tr_phi[meta:].min())
    picco = float(max(np.abs(np.array(tr_fr)).max(), np.abs(np.array(tr_fl)).max()))
    cy = float(np.array(tr_cy)[meta:].max() - np.array(tr_cy)[meta:].min())
    out = dict(caduta=False, amp_mis=round(float(amp_mis), 1),
               picco_Nm=round(picco, 1), cy_mm=round(cy, 1))
    if registra:
        out['tracce'] = (np.array(tr_t), tr_phi, np.array(tr_fr), np.array(tr_fl))
    return out

def campagna():
    print('t12_hip_sway.py - versione ' + VERSIONE)
    print('T12-sim: oscillazione bacino target 8-12 gradi a 0,8-1,0 Hz (TDD par.12.2)')
    print('%6s %9s | %12s %11s %11s  %s' %
          ('f [Hz]', 'cmd [deg]', 'bacino [deg]', 'picco [Nm]', 'CoM_y [mm]', 'esito'))
    risultati = []
    pass_per_f = {}
    for f in (0.8, 0.9, 1.0):
        pass_per_f[f] = False
        for cmd in (8.0, 10.0, 12.0, 14.0, 16.0):
            r = esegui(cmd, f)
            if r['caduta']:
                esito = 'CADUTA a %.1f s' % r['t_caduta']
            else:
                dentro = 8.0 <= r['amp_mis'] <= 12.0
                sotto = r['picco_Nm'] < LIMITE_ROLL
                esito = ('PASS' if (dentro and sotto) else
                         ('fuori banda' if not dentro else 'COPPIA OLTRE LIMITE'))
                if dentro and sotto:
                    pass_per_f[f] = True
            print('%6.1f %9.1f | %12s %11s %11s  %s' %
                  (f, cmd, r['amp_mis'], r['picco_Nm'], r['cy_mm'], esito))
            risultati.append(dict(f_hz=f, cmd_deg=cmd,
                                  **dict((k, v) for k, v in r.items() if k != 'tracce')))
    ok = all(pass_per_f.values())
    print('ESITO T12-sim: %s' % ('PASS - banda 8-12 gradi raggiungibile a 0,8/0,9/1,0 Hz'
                                 if ok else 'FAIL - vedi tabella'))
    json.dump(dict(versione=VERSIONE, esito=('PASS' if ok else 'FAIL'),
                   config_certificata=dict(cmd_deg=CONFIG_CERT[0], f_hz=CONFIG_CERT[1]),
                   risultati=risultati),
              open('t12_hip_sway.json', 'w'), indent=1)
    print('Salvato t12_hip_sway.json')
    r = esegui(CONFIG_CERT[0], CONFIG_CERT[1], registra=True)
    if not r['caduta']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t, phi, fr, fl = r['tracce']
        fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        axs[0].plot(t, phi, color='tab:purple', lw=1)
        for yv in (8, 12, -8, -12):
            axs[0].axhline(yv, color=('g' if abs(yv) == 8 else 'r'), ls='--', lw=0.8)
        axs[0].set_ylabel('rollio bacino [deg]')
        axs[0].set_title('T12 hip sway - cmd %.0f deg @ %.1f Hz -> bacino %.1f deg, picco %.1f Nm'
                         % (CONFIG_CERT[0], CONFIG_CERT[1], r['amp_mis'], r['picco_Nm']))
        axs[1].plot(t, fr, color='tab:blue', lw=1, label='anca roll dx')
        axs[1].plot(t, fl, color='tab:green', lw=1, label='anca roll sx')
        axs[1].axhline(LIMITE_ROLL, color='r', ls='--', lw=1)
        axs[1].axhline(-LIMITE_ROLL, color='r', ls='--', lw=1, label='limite RMD-X8')
        axs[1].set_ylabel('coppia [Nm]')
        axs[1].set_xlabel('tempo [s]')
        axs[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig('t12_hip_sway.png', dpi=150)
        print('Salvato t12_hip_sway.png')

def demo():
    import time
    import mujoco.viewer
    print(__doc__)
    print('Config certificata: comando %.0f deg @ %.1f Hz' % CONFIG_CERT)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    cp = d.subtree_com[0].copy()
    amp = np.radians(CONFIG_CERT[0])
    t = 0.0
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            t0 = time.time()
            cp = controlla(d, cp, t, amp, CONFIG_CERT[1])
            mujoco.mj_step(m, d)
            t += dt
            v.sync()
            resto = dt - (time.time() - t0)
            if resto > 0:
                time.sleep(resto)

if __name__ == '__main__':
    if '--test' in sys.argv:
        campagna()
    else:
        demo()
