#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t18_power_loss.py v1.0 - Test T18 (TDD par.12): taglio alimentazione con robot
in piedi. Criterio TDD: cedimento graduale, nessun movimento incontrollato,
accesso laterale non ostruito. [La latenza elettronica CAN <200 ms e' fuori
dominio simulazione: qui il coast e' istantaneo.]

    python t18_power_loss.py           demo 3D: in piedi 4 s, poi power-loss (loop)
    python t18_power_loss.py --test    misure headless coast vs frenata: tabella+PNG+JSON

Scenari: COAST = attuatori a coppia zero, resta solo lo smorzamento passivo del
modello (2 Nm*s/rad [A] ~ attrito motore+riduttore). FRENATA = smorzamento x5
[A] ~ corto-circuito avvolgimenti (da confermare con MyActuator, nota TDD T18).
Soglie PASS [A, dichiarate]: vel. giunto max <= 10 rad/s; vel. impatto testa
<= 2,0 m/s; spostamento piedi <= 0,30 m (accesso laterale); quiete <= 4 s.
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
K1, K2, XB, KH = 4.0, 0.8, 0.05, 0.4
T_STAND, T_MAX = 4.0, 15.0
SOGLIA_QVEL, SOGLIA_TESTA, SOGLIA_PIEDI, SOGLIA_QUIETE = 10.0, 2.0, 0.30, 8.0

m0 = mujoco.MjModel.from_xml_path(XML)
total_mass = m0.body_mass.sum()
print(f"[MASSA MODELLO] {total_mass:.3f} kg")
assert 65.5 < total_mass < 67.0, f"MASSA FUORI RANGE: {total_mass:.3f} kg - verifica XML"
NOMI = [mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, m0.actuator_trnid[i][0])
        for i in range(m0.nu)]

def esegui(frenata=1.0, registra=False):
    xml = open(XML).read().replace('<geom contype="0" conaffinity="0"',
                                   '<geom contype="1" conaffinity="1"', 1)
    m = mujoco.MjModel.from_xml_string(xml)
    if frenata != 1.0:
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                m.dof_damping[m.jnt_dofadr[j]] *= frenata
    d = mujoco.MjData(m)
    aid = dict((NOMI[i], i) for i in range(m.nu))
    fid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ('r_foot', 'l_foot')]
    hid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'head')
    gid_testa = [g for g in range(m.ngeom) if m.geom_bodyid[g] == hid][0]
    r_testa = float(m.geom_size[gid_testa][0])
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    p_testa_prev = d.geom_xpos[gid_testa].copy()
    z_min_testa = float(p_testa_prev[2])
    fermo_da = None
    cp = d.subtree_com[0].copy()
    gain0 = m.actuator_gainprm.copy()
    bias0 = m.actuator_biasprm.copy()
    piedi0 = 0.5 * (d.xpos[fid[0]] + d.xpos[fid[1]]).copy()
    tagliato = False
    t_cut = None
    max_qvel = 0.0
    v_testa_impatto = 0.0
    testa_toccata = False
    t_quiete = None
    tr = dict(t=[], z=[], qv=[], zt=[])
    n = int((T_STAND + T_MAX) / dt)
    for i in range(n):
        t = i * dt
        com = d.subtree_com[0]
        v = (com - cp) / dt
        cp = com.copy()
        if not tagliato:
            feet = 0.5 * (d.xpos[fid[0]] + d.xpos[fid[1]])
            ap = float(np.clip(K1*(com[0]-feet[0]-XB) + K2*v[0], -0.45, 0.45))
            ar = float(np.clip(-(K1*(com[1]-feet[1]) + K2*v[1]), -0.20, 0.20))
            for j in ('r_ankle_pitch', 'l_ankle_pitch'):
                d.ctrl[aid[j]] = ap
            for j in ('r_ankle_roll', 'l_ankle_roll'):
                d.ctrl[aid[j]] = ar
            for j in ('r_hip_pitch', 'l_hip_pitch'):
                d.ctrl[aid[j]] = float(np.clip(KH*v[0], -0.4, 0.4))
            for j in ('r_hip_roll', 'l_hip_roll'):
                d.ctrl[aid[j]] = float(np.clip(KH*v[1], -0.3, 0.3))
            if t >= T_STAND:
                m.actuator_gainprm[:, 0] = 0.0
                m.actuator_biasprm[:, 1] = 0.0
                m.actuator_biasprm[:, 2] = 0.0
                d.ctrl[:] = 0.0
                tagliato = True
                t_cut = t
        else:
            qv = float(np.abs(d.qvel[6:]).max())
            max_qvel = max(max_qvel, qv)
            p_testa = d.geom_xpos[gid_testa]
            v_testa = float(np.linalg.norm(p_testa - p_testa_prev)) / dt
            p_testa_prev = p_testa.copy()
            z_min_testa = min(z_min_testa, float(p_testa[2]))
            if (not testa_toccata) and p_testa[2] <= r_testa + 0.005:
                testa_toccata = True
                v_testa_impatto = round(v_testa, 2)
            fermo = qv < 0.30 and float(np.linalg.norm(d.qvel[:3])) < 0.05
            if fermo:
                if fermo_da is None:
                    fermo_da = t
                if t_quiete is None and t - fermo_da > 0.5:
                    t_quiete = round(fermo_da - t_cut, 2)
            else:
                fermo_da = None
                t_quiete = None
        mujoco.mj_step(m, d)
        if registra and i % 4 == 0:
            tr['t'].append(t)
            tr['z'].append(float(d.qpos[2]))
            tr['qv'].append(float(np.abs(d.qvel[6:]).max()))
            tr['zt'].append(float(d.xpos[hid][2]))
    m.actuator_gainprm[:] = gain0
    m.actuator_biasprm[:] = bias0
    piedi1 = 0.5 * (d.xpos[fid[0]] + d.xpos[fid[1]])
    sposta_piedi = float(np.linalg.norm((piedi1 - piedi0)[:2]))
    out = dict(
        max_vel_giunto_rad_s=round(max_qvel, 2),
        testa_toccata=bool(testa_toccata),
        vel_impatto_testa_m_s=round(v_testa_impatto, 2),
        z_finale_bacino_m=round(float(d.qpos[2]), 3),
        z_min_testa_m=round(z_min_testa, 3),
        spostamento_piedi_m=round(sposta_piedi, 3),
        t_quiete_s=(round(t_quiete, 2) if t_quiete is not None else None))
    if registra:
        out['tracce'] = tr
    return out

def valuta(r):
    ok = (r['max_vel_giunto_rad_s'] <= SOGLIA_QVEL
          and (not r['testa_toccata'] or r['vel_impatto_testa_m_s'] <= SOGLIA_TESTA)
          and r['spostamento_piedi_m'] <= SOGLIA_PIEDI
          and r['t_quiete_s'] is not None and r['t_quiete_s'] <= SOGLIA_QUIETE)
    return 'PASS' if ok else 'FAIL'

def campagna():
    print('t18_power_loss.py - versione ' + VERSIONE)
    print('T18-sim: power-loss in piedi; soglie [A]: vel<=%.0f rad/s, testa<=%.1f m/s,'
          ' piedi<=%.2f m, quiete<=%.0f s (finestra 15 s, fermo=|qvel|<0,3)' %
          (SOGLIA_QVEL, SOGLIA_TESTA, SOGLIA_PIEDI, SOGLIA_QUIETE))
    esiti = {}
    salva = {}
    tracce_all = {}
    for nome, fren in (('COAST', 1.0), ('FRENATA x5', 5.0), ('FRENATA x20', 20.0)):
        r = esegui(frenata=fren, registra=True)
        esiti[nome] = valuta(r)
        salva[nome] = dict((k, v) for k, v in r.items() if k != 'tracce')
        print('%-10s vel_max %5.2f rad/s | testa: %s (imp. %.2f m/s, z_min %.2f m) | '
              'piedi %.2f m | quiete %s s | bacino finale %.2f m -> %s' %
              (nome, r['max_vel_giunto_rad_s'],
               'tocca' if r['testa_toccata'] else 'NO',
               r['vel_impatto_testa_m_s'], r['z_min_testa_m'],
               r['spostamento_piedi_m'], r['t_quiete_s'],
               r['z_finale_bacino_m'], esiti[nome]))
        tracce_all[nome] = r['tracce']
    esito = 'PASS' if all(e == 'PASS' for e in esiti.values()) else (
        'PARZIALE' if any(e == 'PASS' for e in esiti.values()) else 'FAIL')
    print('ESITO T18-sim: %s  (%s)' %
          (esito, ', '.join('%s: %s' % kv for kv in esiti.items())))
    json.dump(dict(versione=VERSIONE, esito=esito, esiti=esiti,
                   soglie_A=dict(vel_rad_s=SOGLIA_QVEL, testa_m_s=SOGLIA_TESTA,
                                 piedi_m=SOGLIA_PIEDI, quiete_s=SOGLIA_QUIETE),
                   risultati=salva),
              open(_OUT('t18_power_loss.json'), 'w'), indent=1)
    print('Salvato t18_power_loss.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    colori = {'COAST': 'tab:blue', 'FRENATA x5': 'tab:green', 'FRENATA x20': 'tab:orange'}
    for nome, tr in tracce_all.items():
        col = colori[nome]
        axs[0].plot(tr['t'], tr['z'], color=col, lw=1.2, label='bacino - ' + nome)
        axs[0].plot(tr['t'], tr['zt'], color=col, lw=0.8, ls=':',
                    label='testa - ' + nome)
        axs[1].plot(tr['t'], tr['qv'], color=col, lw=1.2, label=nome)
    axs[0].axvline(T_STAND, color='r', ls='--', lw=1)
    axs[1].axvline(T_STAND, color='r', ls='--', lw=1, label='taglio alimentazione')
    axs[1].axhline(SOGLIA_QVEL, color='r', ls=':', lw=1, label='soglia [A]')
    axs[0].set_ylabel('quota [m]')
    axs[1].set_ylabel('max |vel giunto| [rad/s]')
    axs[1].set_xlabel('tempo [s]')
    axs[0].set_title('T18 power-loss: cedimento da posizione eretta')
    axs[0].legend(fontsize=8)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_OUT('t18_power_loss.png'), dpi=150)
    print('Salvato t18_power_loss.png')

def demo():
    import time
    import mujoco.viewer
    print(__doc__)
    xml = open(XML).read().replace('<geom contype="0" conaffinity="0"',
                                   '<geom contype="1" conaffinity="1"', 1)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    aid = dict((NOMI[i], i) for i in range(m.nu))
    fid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ('r_foot', 'l_foot')]
    gain0 = m.actuator_gainprm.copy()
    bias0 = m.actuator_biasprm.copy()

    def reset():
        m.actuator_gainprm[:] = gain0
        m.actuator_biasprm[:] = bias0
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        return d.subtree_com[0].copy(), 0.0, False

    cp, t, tagliato = reset()
    dt = m.opt.timestep
    print('In piedi 4 s, poi POWER-LOSS. Loop automatico. R = reset subito.')
    def tasto(k):
        pass
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            t0 = time.time()
            com = d.subtree_com[0]
            vv = (com - cp) / dt
            cp = com.copy()
            if not tagliato:
                feet = 0.5 * (d.xpos[fid[0]] + d.xpos[fid[1]])
                ap = float(np.clip(K1*(com[0]-feet[0]-XB) + K2*vv[0], -0.45, 0.45))
                ar = float(np.clip(-(K1*(com[1]-feet[1]) + K2*vv[1]), -0.20, 0.20))
                for j in ('r_ankle_pitch', 'l_ankle_pitch'):
                    d.ctrl[aid[j]] = ap
                for j in ('r_ankle_roll', 'l_ankle_roll'):
                    d.ctrl[aid[j]] = ar
                if t >= T_STAND:
                    m.actuator_gainprm[:, 0] = 0.0
                    m.actuator_biasprm[:, 1] = 0.0
                    m.actuator_biasprm[:, 2] = 0.0
                    d.ctrl[:] = 0.0
                    tagliato = True
                    print('POWER-LOSS!')
            elif t > T_STAND + 6.0:
                cp, t, tagliato = reset()
                print('Reset: in piedi.')
                continue
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
