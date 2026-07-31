#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t1_caduta.py v1.0 - Test T1-sim (TDD par.12-T1, PARZIALE: forze d'impatto;
il cedimento strutturale resta al laboratorio/FEM). Caduta laterale da 60 cm.

    python t1_caduta.py           demo 3D: drop laterale da CoM 60 cm (loop)
    python t1_caduta.py --test    campagna S1+S2: tabella + PNG + JSON

S1 DROP: corpo in coast, sdraiato in aria sul fianco destro, CoM esattamente
a 0,60 m, rilascio in caduta libera. S2 SERVIZIO: in piedi, spinta laterale
oltre l'inviluppo (0,8 m/s), SafeGuard E1 rileva |rollio|>15 gradi e taglia
le coppie (coast, da T18); caduta e impatto reali.
Misure [C]: velocita' CoM all'impatto - forza di picco totale e per segmento -
accelerazione di picco del torso [g] - tempo dal primo contatto ai 150 N di E2 -
impulso. [A] dichiarato: il PICCO dipende dalla rigidezza di contatto del
modello (solref MuJoCo standard, non calibrata sul rivestimento cedevole di progetto): usare impulso e
velocita' come dati robusti; PROPOSTA TDD: calibrare il contatto sul rivestimento cedevole reale.
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

VERSIONE = '1.1'
XML = _MP('tx34_v1.xml')
COM_DROP = 0.60
SOGLIA_E2 = 150.0
K1, K2, XB = 4.0, 0.8, 0.05

def modello():
    xml = open(XML).read().replace('<geom contype="0" conaffinity="0"',
                                   '<geom contype="1" conaffinity="1"', 1)
    m = mujoco.MjModel.from_xml_string(xml)
    total_mass = m.body_mass.sum()
    print(f"[MASSA MODELLO] {total_mass:.3f} kg")
    assert 65.5 < total_mass < 67.0, f"MASSA FUORI RANGE: {total_mass:.3f} kg - verifica XML"
    return m

def spegni(m, d):
    m.actuator_gainprm[:, 0] = 0.0
    m.actuator_biasprm[:, 1] = 0.0
    m.actuator_biasprm[:, 2] = 0.0
    d.ctrl[:] = 0.0

def misura_impatto(m, d, T, registra=False, gia_spento=True):
    """Dal momento della chiamata: integra e misura l'impatto col suolo."""
    dt = m.opt.timestep
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    nomi_body = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
    v_prev = d.cvel[tid][3:6].copy()
    com_v_prev = None
    primo_contatto = None
    t_150 = None
    F_picco = 0.0
    acc_picco = 0.0
    impulso = 0.0
    v_impatto = 0.0
    per_body = {}
    PIEDI = ('r_foot', 'l_foot')
    tr = dict(t=[], F=[])
    f6 = np.zeros(6)
    for i in range(int(T/dt)):
        t = i*dt
        com = d.subtree_com[0].copy()
        mujoco.mj_step(m, d)
        vcom = (d.subtree_com[0] - com)/dt
        F_tot = 0.0
        F_corpo = 0.0                              # tutto tranne i piedi
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if 0 in (g1, g2):                      # 0 = pavimento
                mujoco.mj_contactForce(m, d, c, f6)
                fN = abs(float(f6[0]))
                F_tot += fN
                altro = g2 if g1 == 0 else g1
                nb = nomi_body[m.geom_bodyid[altro]]
                per_body[nb] = max(per_body.get(nb, 0.0), fN)
                if nb not in PIEDI:
                    F_corpo += fN
        if F_corpo > 20.0 and primo_contatto is None:  # i piedi non sono un impatto
            primo_contatto = t
        if primo_contatto is not None and t_150 is None:
            v_impatto = max(v_impatto, float(np.linalg.norm(
                com_v_prev if com_v_prev is not None else vcom)))
            if F_corpo >= SOGLIA_E2:
                t_150 = (t - primo_contatto)*1000.0
        if primo_contatto is not None:
            F_picco = max(F_picco, F_tot)
            if t - primo_contatto <= 0.15:         # impulso della finestra d'impatto
                impulso += F_tot*dt
        v_now = d.cvel[tid][3:6].copy()
        acc = float(np.linalg.norm(v_now - v_prev))/dt
        v_prev = v_now
        if primo_contatto is not None:
            acc_picco = max(acc_picco, acc)
        com_v_prev = vcom
        if registra and i % 2 == 0:
            tr['t'].append(t)
            tr['F'].append(F_tot)
    seg = max(per_body.items(), key=lambda kv: kv[1]) if per_body else ('nessuno', 0.0)
    out = dict(v_impatto_m_s=round(v_impatto, 2),
               F_picco_tot_N=round(F_picco, 0),
               segmento_critico=seg[0],
               F_segmento_N=round(seg[1], 0),
               acc_picco_torso_g=round(acc_picco/9.81, 1),
               t_oltre_150N_ms=(round(t_150, 1) if t_150 is not None else None),
               impulso_150ms_Ns=round(impulso, 1),
               forze_per_segmento=dict((k, round(v, 0)) for k, v in
                                       sorted(per_body.items(), key=lambda kv: -kv[1])[:6]))
    if registra:
        out['tracce'] = tr
    return out

def s1_drop(registra=False):
    m = modello()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    # sdraiata in aria sul fianco destro: rollio -90 gradi
    d.qpos[3:7] = [np.cos(-np.pi/4), np.sin(-np.pi/4), 0.0, 0.0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    d.qpos[2] += COM_DROP - float(d.subtree_com[0][2])   # CoM esatto a 0,60
    mujoco.mj_forward(m, d)
    spegni(m, d)
    return misura_impatto(m, d, 2.5, registra)

def s2_servizio(registra=False):
    m = modello()
    d = mujoco.MjData(m)
    aid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i][0]): i
           for i in range(m.nu)}
    fid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ('r_foot', 'l_foot')]
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    cp = d.subtree_com[0].copy()
    t_e1 = None
    for i in range(int(6.0/dt)):
        t = i*dt
        com = d.subtree_com[0]
        v = (com - cp)/dt
        cp = com.copy()
        if t_e1 is None:
            feet = 0.5*(d.xpos[fid[0]] + d.xpos[fid[1]])
            ap = float(np.clip(K1*(com[0]-feet[0]-XB) + K2*v[0], -0.45, 0.45))
            ar = float(np.clip(-(K1*(com[1]-feet[1]) + K2*v[1]), -0.20, 0.20))
            for j in ('r_ankle_pitch', 'l_ankle_pitch'):
                d.ctrl[aid[j]] = ap
            for j in ('r_ankle_roll', 'l_ankle_roll'):
                d.ctrl[aid[j]] = ar
            if abs(t - 2.0) < dt/2:
                d.qvel[1] -= 0.8          # spinta laterale oltre l'inviluppo (0,4)
            qw, qx, qy, qz = d.qpos[3:7]
            rollio = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))
            if t > 2.0 and abs(rollio) > np.radians(15.0):
                spegni(m, d)              # SafeGuard E1: caduta rilevata -> coast
                t_e1 = t
                break
        mujoco.mj_step(m, d)
    r = misura_impatto(m, d, 3.0, registra)
    r['t_trigger_E1_s'] = round(t_e1 - 2.0, 2) if t_e1 else None
    return r

def campagna():
    print('t1_caduta.py - versione ' + VERSIONE)
    print('T1-sim: caduta laterale, forze di impatto. [A] picchi dipendenti dalla')
    print('rigidezza di contatto del modello (non calibrata sul rivestimento): impulso e')
    print('velocita sono i dati robusti. Soglia E2 = 150 N (finestra 100 ms).')
    ris = {}
    for nome, fn in (('S1 drop 60 cm', s1_drop), ('S2 servizio (spinta+E1)', s2_servizio)):
        r = fn(registra=(nome.startswith('S1')))
        ris[nome] = dict((k, v) for k, v in r.items() if k != 'tracce')
        if nome.startswith('S1'):
            tr = r['tracce']
        print('%s:' % nome)
        print('  v impatto %.2f m/s | F picco %.0f N (%s: %.0f N) | acc torso %.1f g'
              % (r['v_impatto_m_s'], r['F_picco_tot_N'], r['segmento_critico'],
                 r['F_segmento_N'], r['acc_picco_torso_g']))
        print('  150 N (corpo) superati in %s ms | impulso 150 ms %.1f N*s | zone: %s'
              % (r['t_oltre_150N_ms'], r['impulso_150ms_Ns'],
                 ', '.join('%s %.0fN' % kv for kv in list(r['forze_per_segmento'].items())[:4])))
        if 't_trigger_E1_s' in r:
            print('  E1 (|rollio|>15 gradi) scattato %.2f s dopo la spinta' % r['t_trigger_E1_s'])
    print('ESITO T1-sim: MISURATO (parziale come da handoff: strutturale al laboratorio).')
    print('SafeGuard: nel drop i 150 N scattano al primo istante (0 ms). Nella caduta')
    print('in servizio il primo contatto (mano, >20 N) PRECEDE l\'impatto del tronco')
    print('di %.0f ms: la mano e\' il preavviso naturale per E1/E2.' %
          ris['S2 servizio (spinta+E1)']['t_oltre_150N_ms'])
    json.dump(dict(versione=VERSIONE, esito='MISURATO', com_drop_m=COM_DROP,
                   soglia_E2_N=SOGLIA_E2, preavviso_mano_ms=ris['S2 servizio (spinta+E1)']['t_oltre_150N_ms'],
                   scenari=ris), open(_OUT('t1_caduta.json'), 'w'), indent=1)
    print('Salvato t1_caduta.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(9, 7))
    axs[0].plot(tr['t'], tr['F'], color='tab:red', lw=1.0)
    axs[0].axhline(SOGLIA_E2, color='g', ls='--', lw=1, label='E2 threshold 150 N')
    axs[0].set_xlabel('time [s]')
    axs[0].set_ylabel('total ground force [N]')
    axs[0].set_title('T1-sim S1: lateral CoM drop 60 cm - v %.2f m/s, peak %.0f N [A stiffness]'
                     % (ris['S1 drop 60 cm']['v_impatto_m_s'],
                        ris['S1 drop 60 cm']['F_picco_tot_N']))
    axs[0].legend(fontsize=8)
    zone = ris['S1 drop 60 cm']['forze_per_segmento']
    axs[1].barh(list(zone.keys())[::-1], list(zone.values())[::-1], color='tab:orange')
    axs[1].set_xlabel('peak per segment [N]')
    axs[1].set_title('impact map: segments in contact (S1)')
    fig.tight_layout()
    fig.savefig(_OUT('t1_caduta.png'), dpi=150)
    print('Salvato t1_caduta.png')

def demo():
    import time
    import mujoco.viewer
    print(__doc__)
    m = modello()
    d = mujoco.MjData(m)
    def reset():
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[3:7] = [np.cos(-np.pi/4), np.sin(-np.pi/4), 0.0, 0.0]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        d.qpos[2] += COM_DROP - float(d.subtree_com[0][2])
        mujoco.mj_forward(m, d)
        spegni(m, d)
    reset()
    dt = m.opt.timestep
    t = 0.0
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            t0 = time.time()
            mujoco.mj_step(m, d)
            t += dt
            if t > 3.5:
                reset()
                t = 0.0
            v.sync()
            resto = dt - (time.time() - t0)
            if resto > 0:
                time.sleep(resto)

if __name__ == '__main__':
    if '--test' in sys.argv:
        campagna()
    else:
        demo()
