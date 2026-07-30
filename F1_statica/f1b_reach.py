#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f1b_reach.py v1.0 - Modulo F1-B (campagna par.12-T2 parziale / range par.3.2):
reach del braccio destro in 4 direzioni + reach esteso col busto (legge F1-A).
PASS: target raggiunto entro 5 cm, equilibrio mantenuto, coppie sotto forcerange.

    python f1b_reach.py           demo 3D: sequenza di reach in loop
    python f1b_reach.py --test    campagna: envelope + tabella + PNG + JSON

Misure per direzione: distanza massima dalla spalla [m] · errore finale [mm] ·
spostamento CoM [mm] · picchi coppia spalla/gomito vs forcerange XML (40/40/30 Nm)
· angoli articolari usati vs range. Braccio: L1=0,24 + L2=0,25 (spalla->mano).
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
L1, L2 = 0.38, 0.40            # gamba
L1A, L2A = 0.24, 0.25          # braccio
K1, K2, XB = 4.0, 0.8, 0.03
KXA, KC, KCD, KP, KD = 0.75, 3.0, 0.5, 2.5, 0.4   # legge posturale F1-A
ZA0 = 0.78
LIM_S = (-2.5, 1.5)
LIM_E = (-2.3, 0.0)
LIM_R = (-1.6, 0.3)

m = mujoco.MjModel.from_xml_path(XML)
total_mass = m.body_mass.sum()
print(f"[MASSA MODELLO] {total_mass:.3f} kg")
assert 65.5 < total_mass < 67.0, f"MASSA FUORI RANGE: {total_mass:.3f} kg — verifica XML"
NOMI = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i][0])
        for i in range(m.nu)]
AID = dict((n, i) for i, n in enumerate(NOMI))
FID = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ('r_foot', 'l_foot')]
BH = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'r_hand')
JID = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(m.njnt)}

def ik_gamba(xa, za):
    D = np.hypot(xa, za); D = min(D, L1 + L2 - 1e-4)
    gi = np.arccos(np.clip((L1*L1+L2*L2-D*D)/(2*L1*L2), -1, 1))
    qk = np.pi - gi
    beta = np.arccos(np.clip((L1*L1+D*D-L2*L2)/(2*L1*D), -1, 1))
    return -(np.arctan2(xa, za) + beta), qk

def ik_braccio(dx, dz_giu):
    D = np.hypot(dx, dz_giu); D = min(D, L1A + L2A - 1e-4)
    gi = np.arccos(np.clip((L1A*L1A+L2A*L2A-D*D)/(2*L1A*L2A), -1, 1))
    q_e = -(np.pi - gi)
    beta = np.arccos(np.clip((L1A*L1A+D*D-L2A*L2A)/(2*L1A*D), -1, 1))
    q_s = -(np.arctan2(dx, dz_giu) - beta)
    return q_s, q_e

def smooth(u):
    return 0.5 - 0.5*np.cos(np.pi*np.clip(u, 0, 1))

def passo(d, cp, lean_ref, arm):
    """Equilibrio+postura (legge F1-A, za fissa) + comando braccio destro.
    arm = (q_s, q_e, q_r). Ritorna nuovo CoM."""
    dt = m.opt.timestep
    com = d.subtree_com[0]
    v = (com - cp)/dt
    qw, qx, qy, qz = d.qpos[3:7]
    pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))
    wy = float(d.qvel[4])
    feet = 0.5*(d.xpos[FID[0]] + d.xpos[FID[1]])
    dx = com[0] - feet[0] - (XB + 0.12*max(0.0, lean_ref))
    qh, qk = ik_gamba(KXA*max(0.0, lean_ref)*0.10, ZA0)
    ref = float(np.clip(lean_ref - KC*dx - KCD*v[0], -0.20, 0.90))
    tau = -ref + KP*(pitch - ref) + KD*wy
    qh_cmd = float(np.clip(qh + tau, -2.0, 0.5))
    liv = -(qh + qk)
    trim = float(np.clip(K1*dx + K2*v[0], -0.20, 0.20))
    a_cmd = float(np.clip(liv + trim, -0.47, 0.47))
    ar = float(np.clip(-(K1*(com[1]-feet[1]) + K2*v[1]), -0.20, 0.20))
    for lato in ('r', 'l'):
        d.ctrl[AID[lato+'_hip_pitch']] = qh_cmd
        d.ctrl[AID[lato+'_knee']] = qk
        d.ctrl[AID[lato+'_ankle_pitch']] = a_cmd
        d.ctrl[AID[lato+'_ankle_roll']] = ar
        d.ctrl[AID[lato+'_hip_roll']] = 0.0
    q_s, q_e, q_r = arm
    d.ctrl[AID['r_shoulder_pitch']] = float(np.clip(q_s, *LIM_S))
    d.ctrl[AID['r_elbow']] = float(np.clip(q_e, *LIM_E))
    d.ctrl[AID['r_shoulder_roll']] = float(np.clip(q_r, *LIM_R))
    d.ctrl[AID['r_wrist_pitch']] = 0.0
    return com.copy()

def braccio_verso(d, target, frazione):
    """IK del braccio verso il punto interpolato mano_riposo->target (frame busto)."""
    sh = d.xanchor[JID['r_shoulder_pitch']]
    mano0 = np.array([sh[0], sh[1], sh[2]-0.49])
    tgt = mano0 + (np.array(target) - mano0)*frazione
    vec = tgt - sh
    qw, qx, qy, qz = d.qpos[3:7]
    th = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))
    # mondo -> busto (rotazione -th attorno a y)
    vx = np.cos(th)*vec[0] + np.sin(th)*vec[2]
    vz = -np.sin(th)*vec[0] + np.cos(th)*vec[2]
    q_r = float(np.clip(np.arctan2(vec[1]-0.0, 0.49)*1.0, *LIM_R)) if abs(vec[1]) > 0.06 else 0.0
    q_s, q_e = ik_braccio(vx, -vz)
    return (q_s, q_e, q_r), tgt

def esegui(nome, target, lean_ref=0.0, T=4.0, registra=False):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    cp = d.subtree_com[0].copy()
    com0x = cp[0]
    sh0 = d.xanchor[JID['r_shoulder_pitch']].copy()
    picchi = dict(sp=0.0, sr=0.0, go=0.0)
    ang = dict(qs=0.0, qe=0.0)
    dcom = 0.0
    err_fin = []
    tr = dict(t=[], err=[], pitch=[], ts=[])
    for i in range(int(T/dt)):
        t = i*dt
        fr = smooth(t/2.0)
        lr = lean_ref*smooth(t/2.0)
        arm, tgt = braccio_verso(d, target, fr)
        mano = d.xpos[BH]
        ez = float(tgt[2] - mano[2])
        sh_now = d.xanchor[JID['r_shoulder_pitch']]
        er = float(np.linalg.norm(np.array(tgt) - sh_now) -
                   np.linalg.norm(mano - sh_now))
        arm = (arm[0] - 1.5*ez, arm[1] + 2.0*er, arm[2])
        cp = passo(d, cp, lr, arm)
        mujoco.mj_step(m, d)
        picchi['sp'] = max(picchi['sp'], abs(float(d.actuator_force[AID['r_shoulder_pitch']])))
        picchi['sr'] = max(picchi['sr'], abs(float(d.actuator_force[AID['r_shoulder_roll']])))
        picchi['go'] = max(picchi['go'], abs(float(d.actuator_force[AID['r_elbow']])))
        ang['qs'] = max(ang['qs'], abs(float(d.qpos[m.jnt_qposadr[JID['r_shoulder_pitch']]])))
        ang['qe'] = max(ang['qe'], abs(float(d.qpos[m.jnt_qposadr[JID['r_elbow']]])))
        dcom = max(dcom, abs(float(d.subtree_com[0][0]) - com0x))
        err = float(np.linalg.norm(d.xpos[BH] - np.array(target)))
        if t > T - 0.5:
            err_fin.append(err)
        if registra and i % 4 == 0:
            qw, qx, qy, qz = d.qpos[3:7]
            tr['t'].append(t)
            tr['err'].append(err*1000)
            tr['pitch'].append(np.degrees(np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1))))
            tr['ts'].append(float(d.actuator_force[AID['r_shoulder_pitch']]))
        if d.qpos[2] < 0.70:
            return dict(caduta=True, t_caduta=round(t, 1))
    err_m = float(np.mean(err_fin))
    out = dict(caduta=False, raggiunto=bool(err_m < 0.05),
               err_finale_mm=round(err_m*1000, 1),
               dist_spalla_m=round(float(np.linalg.norm(np.array(target)-sh0)), 3),
               dCoM_mm=round(dcom*1000, 1),
               picco_spalla_Nm=round(picchi['sp'], 1),
               picco_roll_Nm=round(picchi['sr'], 1),
               picco_gomito_Nm=round(picchi['go'], 1),
               spalla_max_deg=round(np.degrees(ang['qs']), 1),
               gomito_max_deg=round(np.degrees(ang['qe']), 1))
    if registra:
        out['tracce'] = tr
    return out

def _spalla0():
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return d.xanchor[JID['r_shoulder_pitch']].copy()

def campagna():
    print('f1b_reach.py - versione ' + VERSIONE)
    print('F1-B reach braccio dx: envelope 4 direzioni + avanti esteso col busto '
          '(forcerange spalla/gomito: 43/30 Nm)')
    sh = _spalla0()
    print('spalla dx a riposo: x=%.2f y=%.2f z=%.2f' % (sh[0], sh[1], sh[2]))
    piani = dict(
        avanti=[(sh[0]+dd, sh[1], sh[2]) for dd in (0.30, 0.40, 0.45, 0.48)],
        alto=[(sh[0]+0.10, sh[1], sh[2]+dd) for dd in (0.25, 0.35, 0.42, 0.46)],
        basso=[(sh[0]+0.15, sh[1], sh[2]-dd) for dd in (0.30, 0.40, 0.45)],
    )
    ris = {}
    tab = []
    # laterale: abduzione diretta (qr), envelope = escursione y della mano
    print('-- laterale: abduzione diretta --')
    best_lat = None
    for qr in (-0.6, -1.1, -1.55):
        d2 = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d2, 0); mujoco.mj_forward(m, d2)
        cp2 = d2.subtree_com[0].copy()
        y0 = float(d2.xpos[BH][1])
        pic = 0.0
        caduta = False
        for i in range(int(3.5/m.opt.timestep)):
            fr = smooth(i*m.opt.timestep/1.8)
            cp2 = passo(d2, cp2, 0.0, (0.0, 0.0, qr*fr))
            mujoco.mj_step(m, d2)
            pic = max(pic, abs(float(d2.actuator_force[AID['r_shoulder_roll']])))
            if d2.qpos[2] < 0.70:
                caduta = True
                break
        dy = abs(float(d2.xpos[BH][1]) - y0)
        r = dict(direzione='laterale', qr_cmd_deg=round(np.degrees(qr), 1),
                 caduta=caduta, escursione_y_m=round(dy, 3),
                 picco_roll_Nm=round(pic, 1))
        tab.append(r)
        print('  qr=%5.1f deg: escursione y=%.2f m, roll %.1f Nm, %s' %
              (np.degrees(qr), dy, pic, 'CADUTA' if caduta else 'in equilibrio'))
        if not caduta:
            best_lat = dict(dist_spalla_m=dy)
    ris['laterale'] = best_lat
    print('%9s %7s | %5s %9s %7s %8s %8s %8s  %s' %
          ('direz.', 'd [m]', 'ok', 'err [mm]', 'dCoM', 'Nm spal', 'Nm gom', 'ang sp',
           'esito'))
    for direz, targets in piani.items():
        best = None
        for tg in targets:
            r = esegui(direz, tg)
            d_sp = r.get('dist_spalla_m', 0)
            if r['caduta']:
                print('%9s %7.2f | CADUTA a %.1f s' % (direz, d_sp, r['t_caduta']))
                tab.append(dict(direzione=direz, **r))
                break
            ok = r['raggiunto']
            print('%9s %7.2f | %5s %9.1f %7.1f %8.1f %8.1f %8.1f  %s' %
                  (direz, d_sp, 'SI' if ok else 'no', r['err_finale_mm'], r['dCoM_mm'],
                   r['picco_spalla_Nm'], r['picco_gomito_Nm'], r['spalla_max_deg'],
                   'PASS' if ok else 'fuori portata'))
            tab.append(dict(direzione=direz, target=list(np.round(tg, 3)), **r))
            if ok:
                best = r
        ris[direz] = best
    # avanti esteso col busto (legge F1-A)
    print('-- avanti ESTESO col busto (lean F1-A) --')
    best_est = None
    for dd, lean in ((0.55, 0.35), (0.60, 0.45), (0.66, 0.55)):
        tg = (sh[0]+dd, sh[1], sh[2]-0.05)
        r = esegui('avanti_busto', tg, lean_ref=lean, T=5.5)
        if r['caduta']:
            print('%9s %7.2f | CADUTA a %.1f s' % ('av+busto', dd, r['t_caduta']))
            tab.append(dict(direzione='avanti_busto', **r))
            break
        ok = r['raggiunto']
        print('%9s %7.2f | %5s %9.1f %7.1f %8.1f %8.1f %8.1f  %s' %
              ('av+busto', r['dist_spalla_m'], 'SI' if ok else 'no', r['err_finale_mm'],
               r['dCoM_mm'], r['picco_spalla_Nm'], r['picco_gomito_Nm'],
               r['spalla_max_deg'], 'PASS' if ok else 'fuori portata'))
        tab.append(dict(direzione='avanti_busto', target=list(np.round(tg, 3)),
                        lean_ref=lean, **r))
        if ok:
            best_est = r
    ris['avanti_busto'] = best_est
    ok_tot = all(v is not None for k, v in ris.items() if k != 'avanti_busto')
    print('ESITO F1-B: %s - envelope: ' % ('PASS' if ok_tot else 'FAIL') +
          ', '.join('%s %.2f m' % (k, v['dist_spalla_m'])
                    for k, v in ris.items() if v))
    json.dump(dict(versione=VERSIONE, esito=('PASS' if ok_tot else 'FAIL'),
                   envelope=dict((k, (v['dist_spalla_m'] if v else None))
                                 for k, v in ris.items()),
                   risultati=tab),
              open(_OUT('f1b_reach.json'), 'w'), indent=1)
    print('Salvato f1b_reach.json')
    r = esegui('alto', (sh[0]+0.10, sh[1], sh[2]+0.43), T=4.5, registra=True)
    if not r['caduta']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        tr = r['tracce']
        fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        axs[0].plot(tr['t'], tr['err'], color='tab:purple', lw=1.2)
        axs[0].axhline(50, color='g', ls='--', lw=1, label='tolleranza 50 mm')
        axs[0].set_ylabel('errore mano-target [mm]')
        axs[0].set_title('F1-B reach ALTO 0,43 m: err fin %.0f mm - spalla SATURA a %.0f Nm' %
                         (r['err_finale_mm'], r['picco_spalla_Nm']))
        axs[0].legend(fontsize=8)
        axs[1].plot(tr['t'], tr['pitch'], color='tab:orange', lw=1.2)
        axs[1].set_ylabel('inclinazione busto [deg]')
        axs[2].plot(tr['t'], tr['ts'], color='tab:blue', lw=1.2)
        axs[2].axhline(40, color='r', ls='--', lw=1, label='forcerange spalla')
        axs[2].axhline(-40, color='r', ls='--', lw=1)
        axs[2].set_ylabel('coppia spalla [Nm]')
        axs[2].set_xlabel('tempo [s]')
        axs[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(_OUT('f1b_reach.png'), dpi=150)
        print('Salvato f1b_reach.png')

def demo():
    import time
    import mujoco.viewer
    print(__doc__)
    sh = _spalla0()
    seq = [('avanti', (sh[0]+0.45, sh[1], sh[2]), None),
           ('alto', (sh[0]+0.10, sh[1], sh[2]+0.42), None),
           ('basso', (sh[0]+0.15, sh[1], sh[2]-0.42), None),
           ('laterale (abduzione)', None, -1.4)]
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    cp = d.subtree_com[0].copy()
    t, idx = 0.0, 0
    print('Sequenza reach in loop: ' + ' -> '.join(s[0] for s in seq))
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            t0 = time.time()
            nome, tgt, qr = seq[idx]
            fr = smooth(t/2.0)
            if tgt is None:
                arm = (0.0, 0.0, qr*fr)
            else:
                arm, _ = braccio_verso(d, tgt, fr)
                mano = d.xpos[BH]
                ez = float(tgt[2] - mano[2])
                sh_now = d.xanchor[JID['r_shoulder_pitch']]
                er = float(np.linalg.norm(np.array(tgt) - sh_now) -
                           np.linalg.norm(mano - sh_now))
                arm = (arm[0] - 1.5*ez, arm[1] + 2.0*er, arm[2])
            cp = passo(d, cp, 0.0, arm)
            mujoco.mj_step(m, d)
            t += dt
            if t > 3.5:
                t = 0.0
                idx = (idx + 1) % len(seq)
                print('-> ' + seq[idx][0])
            v.sync()
            resto = dt - (time.time() - t0)
            if resto > 0:
                time.sleep(resto)

if __name__ == '__main__':
    if '--test' in sys.argv:
        campagna()
    else:
        demo()
