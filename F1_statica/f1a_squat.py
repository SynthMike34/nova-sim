#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f1a_squat.py v1.0 - Modulo F1-A (campagna par.12 / range par.3.2): squat.
Discesa da eretta alla massima profondita' raggiungibile, tenuta 2 s, risalita.
PASS: nessuna caduta e ritorno eretta (bacino >= 0,88 m, assetto < 9 gradi).

    python f1a_squat.py           demo 3D: squat alla profondita' di riferimento (loop)
    python f1a_squat.py --test    campagna: sweep profondita' + tabella + PNG + JSON

Misure: flessione max ginocchio [deg] vs range TDD par.3.2 (2-117) · dorsiflessione
caviglia comandata vs limite +/-27 · picchi coppia ginocchio/anca vs RMD (120/80 Nm)
· quota minima bacino. Nota di perimetro: l'inginocchiamento con contatto a terra
e' il modulo separato F1-A2 (richiede collisioni ginocchia nel sorgente).
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
import numpy as np
import mujoco

VERSIONE = '1.2'
XML = _MP('tx34_v1.xml')
K1, K2, XB = 4.0, 0.8, 0.03
KXA = 0.75                    # anche indietro con la profondita': xa = KXA*(0.78-za)
LEAN0 = 2.0                   # busto avanti in proporzione alla profondita' (rif. assetto)
KC, KCD = 3.0, 0.5            # correzione del riferimento col baricentro
KP, KD = 2.5, 0.4             # servo d'assetto sul bacino (via anche)
ZA_CERT = 0.60                # profondita' di riferimento v1.1 [C] (sweep fine, senza dita)
TOE_RAMP = 0.30               # v1.1: rampino plantare (rad, ~17 gr) ultimo quarto
TID = []                      # pigro: il modello m nasce piu' avanti
def _tid():
    if not TID:
        TID.extend(i for i in range(m.nu)
                   if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT,
                                         m.actuator_trnid[i][0]) or '')
                   .endswith('toe_pitch'))
    return TID
L1, L2 = 0.38, 0.40
LIM_ANKLE_CMD = 0.47

m = mujoco.MjModel.from_xml_path(XML)
total_mass = m.body_mass.sum()
print(f"[MASSA MODELLO] {total_mass:.3f} kg")
assert 65.5 < total_mass < 67.0, f"MASSA FUORI RANGE: {total_mass:.3f} kg — verifica XML"
NOMI = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i][0])
        for i in range(m.nu)]
AID = dict((n, i) for i, n in enumerate(NOMI))
FID = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ('r_foot', 'l_foot')]
JQ = dict((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j), m.jnt_qposadr[j])
          for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)

def ik(xa, za):
    D = np.hypot(xa, za)
    D = min(D, L1 + L2 - 1e-4)
    gi = np.arccos(np.clip((L1*L1 + L2*L2 - D*D) / (2*L1*L2), -1, 1))
    qk = np.pi - gi
    beta = np.arccos(np.clip((L1*L1 + D*D - L2*L2) / (2*L1*D), -1, 1))
    return -(np.arctan2(xa, za) + beta), qk

def smooth(u):
    return 0.5 - 0.5*np.cos(np.pi*np.clip(u, 0, 1))

def profilo_za(t, za_tgt, t_giu=3.0, t_hold=2.0, t_su=3.0, t_pre=1.0):
    if t < t_pre:
        return 0.78
    t -= t_pre
    if t < t_giu:
        return 0.78 + (za_tgt - 0.78)*smooth(t/t_giu)
    t -= t_giu
    if t < t_hold:
        return za_tgt
    t -= t_hold
    if t < t_su:
        return za_tgt + (0.78 - za_tgt)*smooth(t/t_su)
    return 0.78

def passo_controllo(d, cp, t, za_tgt, dt, dita=False):
    com = d.subtree_com[0]
    v = (com - cp)/dt
    qw, qx, qy, qz = d.qpos[3:7]
    pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))
    feet = 0.5*(d.xpos[FID[0]] + d.xpos[FID[1]])
    za = profilo_za(t, za_tgt)
    xa = KXA*(0.78 - za)
    qh, qk = ik(xa, za)
    dx = com[0] - feet[0] - XB
    wy = float(d.qvel[4])
    ref = float(np.clip(LEAN0*(0.78 - za) - KC*dx - KCD*v[0], -0.20, 0.90))
    tau = -ref + KP*(pitch - ref) + KD*wy
    qh_cmd = float(np.clip(qh + tau, -2.0, 0.5))
    liv = -(qh + qk)                      # piede piatto dall'IK pura: il delta d'anca
    a_cmd = float(np.clip(liv, -LIM_ANKLE_CMD, LIM_ANKLE_CMD))   # diventa beccheggio busto
    trim = float(np.clip(K1*dx + K2*v[0], -0.12, 0.12))
    ar = float(np.clip(-(K1*(com[1]-feet[1]) + K2*v[1]), -0.20, 0.20))
    for lato in ('r', 'l'):
        d.ctrl[AID[lato+'_hip_pitch']] = qh_cmd
        d.ctrl[AID[lato+'_knee']] = qk
        d.ctrl[AID[lato+'_ankle_pitch']] = float(np.clip(a_cmd + trim,
                                                         -LIM_ANKLE_CMD, LIM_ANKLE_CMD))
        d.ctrl[AID[lato+'_ankle_roll']] = ar
        d.ctrl[AID[lato+'_hip_roll']] = 0.0
    if dita and _tid():
        u = float(np.clip((0.78 - za)/max(0.78 - za_tgt, 1e-6), 0.0, 1.0))
        toe = TOE_RAMP*smooth((u - 0.75)/0.25)
        for i in _tid():
            d.ctrl[i] = toe
    return com.copy(), abs(liv)

def esegui(za_tgt, registra=False, dita=False):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    cp = d.subtree_com[0].copy()
    T = 1.0 + 3.0 + 2.0 + 3.0 + 2.0
    max_knee = 0.0
    max_ankle_cmd = 0.0
    picchi = dict(knee=0.0, hip_pitch=0.0, ankle_pitch=0.0)
    z_min = 1.0
    tr = dict(t=[], z=[], knee=[], tk=[], th=[])
    for i in range(int(T/dt)):
        t = i*dt
        cp, liv_abs = passo_controllo(d, cp, t, za_tgt, dt, dita)
        max_ankle_cmd = max(max_ankle_cmd, liv_abs)
        mujoco.mj_step(m, d)
        max_knee = max(max_knee, float(d.qpos[JQ['r_knee']]))
        for nome in ('knee', 'hip_pitch', 'ankle_pitch'):
            f = max(abs(float(d.actuator_force[AID['r_'+nome]])),
                    abs(float(d.actuator_force[AID['l_'+nome]])))
            picchi[nome] = max(picchi[nome], f)
        z_min = min(z_min, float(d.qpos[2]))
        if registra and i % 4 == 0:
            tr['t'].append(t)
            tr['z'].append(float(d.qpos[2]))
            tr['knee'].append(np.degrees(float(d.qpos[JQ['r_knee']])))
            tr['tk'].append(float(d.actuator_force[AID['r_knee']]))
            tr['th'].append(float(d.actuator_force[AID['r_hip_pitch']]))
        if d.qpos[2] < 0.30:
            return dict(caduta=True, t_caduta=round(t, 1))
    qw, qx, qy, qz = d.qpos[3:7]
    pitch_fin = abs(np.degrees(np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))))
    tornata = d.qpos[2] >= 0.88 and pitch_fin < 9.0
    out = dict(caduta=False, tornata_eretta=bool(tornata),
               z_min_bacino_m=round(z_min, 3),
               flessione_max_ginocchio_deg=round(np.degrees(max_knee), 1),
               caviglia_cmd_max_deg=round(np.degrees(max_ankle_cmd), 1),
               caviglia_satura=bool(max_ankle_cmd > LIM_ANKLE_CMD + 1e-6),
               picco_ginocchio_Nm=round(picchi['knee'], 1),
               picco_anca_Nm=round(picchi['hip_pitch'], 1),
               picco_caviglia_Nm=round(picchi['ankle_pitch'], 1))
    if registra:
        out['tracce'] = tr
    return out

def campagna():
    print('f1a_squat.py - versione ' + VERSIONE)
    print('F1-A squat v1.1: sweep SENZA e CON rampino plantare (dita %.0f gr '
          'nell\'ultimo 40%% della discesa).' % np.degrees(TOE_RAMP))
    tabella = {}
    for dita, eti in ((False, 'SENZA DITA'), (True, 'CON RAMPINO')):
        print('--- %s ---' % eti)
        print('%7s | %7s %8s %9s %7s %9s  %s' %
              ('za [m]', 'z_min', 'ginoc.', 'cav.cmd', 'satura', 'Nm cavig.', 'esito'))
        risultati, migliore = [], None
        for za in (0.70, 0.65, 0.60, 0.55, 0.50, 0.48, 0.46, 0.44, 0.42):
            r = esegui(za, dita=dita)
            if r['caduta']:
                print('%7.2f | CADUTA a %.1f s' % (za, r['t_caduta']))
            else:
                esito = 'PASS' if r['tornata_eretta'] else 'non torna eretta'
                if r['tornata_eretta']:
                    migliore = za
                print('%7.2f | %7.3f %8.1f %9.1f %7s %9.1f  %s' %
                      (za, r['z_min_bacino_m'], r['flessione_max_ginocchio_deg'],
                       r['caviglia_cmd_max_deg'],
                       'SI' if r['caviglia_satura'] else 'no',
                       r['picco_caviglia_Nm'], esito))
            risultati.append(dict(za_target=za, dita=dita, **dict(
                (k, v) for k, v in r.items() if k != 'tracce')))
        tabella[eti] = dict(migliore=migliore, risultati=risultati)
    zs = tabella['SENZA DITA']['migliore']
    zc = tabella['CON RAMPINO']['migliore']
    print('CONFRONTO: za certificabile %s m (senza) vs %s m (con rampino) -> %s'
          % (zs, zc, ('MIGLIORA di %d mm' % round((zs - zc)*1000)) if zc and zs
             and zc < zs else 'RAMPINO BOCCIATO (scoperta 33) - il dato e za=%.2f dal baseline' % (zs or 0)))
    esito_f1a = 'PASS' if zs is not None else 'FAIL'
    json.dump(dict(versione=VERSIONE, esito=esito_f1a, za_riferimento_C=zs,
                   za_con_rampino=zc,
                   risultati=tabella['SENZA DITA']['risultati']
                   + tabella['CON RAMPINO']['risultati']),
              open(_OUT('f1a_squat.json'), 'w'), indent=1)
    print('Salvato f1a_squat.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    for dita, col, eti, zpick in ((False, 'tab:gray', 'senza dita', zs),
                                  (True, 'tab:green', 'con rampino', zc)):
        if zpick is None:
            continue
        r = esegui(zpick, registra=True, dita=dita)
        tr = r['tracce']
        axs[0].plot(tr['t'], tr['z'], color=col, lw=2,
                    label='%s (za %.2f)' % (eti, zpick))
    axs[0].set_xlabel('tempo [s]'); axs[0].set_ylabel('quota bacino [m]')
    axs[0].set_title('discesa piu profonda certificabile'); axs[0].legend()
    axs[0].grid(alpha=0.3)
    vs = [zs or 0, zc or 0]
    axs[1].bar(['senza\ndita', 'con\nRAMPINO'], vs,
               color=['tab:gray', 'tab:green'], width=0.5)
    for x, v in enumerate(vs):
        axs[1].text(x, v, '%.2f m' % v, ha='center', va='bottom',
                    fontsize=13, fontweight='bold')
    axs[1].set_ylabel('za di riferimento [m] (piu basso = piu giu)')
    axs[1].set_title('F1-A v1.1: il rampino plantare')
    axs[1].grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(_OUT('f1a_squat.png'), dpi=150)
    print('Salvato f1a_squat.png')

def demo():
    import time
    import mujoco.viewer
    print(__doc__)
    print('Squat continuo alla profondita di riferimento za = %.2f m [C] (loop automatico).' % ZA_CERT)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    cp = d.subtree_com[0].copy()
    t = 0.0
    ciclo = 1.0 + 3.0 + 2.0 + 3.0 + 1.0
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            t0 = time.time()
            cp, _ = passo_controllo(d, cp, t % ciclo, ZA_CERT, dt, dita=False)
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
