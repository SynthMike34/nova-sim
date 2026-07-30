#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f1d_seduta.py v1.0 - Modulo F1-D (TDD par.12-T2 mobilita' / par.3.2): sedersi
su sedia h=45 cm e rialzarsi. PASS: seduta verificata (peso sulla sedia),
ritorno eretta (bacino >= 0,88 m, assetto < 9 gradi), coppie entro par.2.3.

    python f1d_seduta.py           demo 3D: siediti/alzati in loop
    python f1d_seduta.py --test    ciclo misurato: metriche + PNG + JSON

Sedia nel SORGENTE (regola piattaforma): modello derivato tx34_v1_sedia.xml
rigenerato da tx34_v1.xml a ogni esecuzione (sedile 45x44 cm, piano a 0,45 m,
collisioni bacino+cosce abilitate nel derivato).
Fasi: discesa 3 s -> seduta 2 s -> pendenza avanti 1,5 s -> risalita 3 s -> eretta.
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

VERSIONE = '1.5'
XML = _MP('tx34_v1.xml')
XML_SEDIA = 'tx34_v1_sedia.xml'
L1, L2 = 0.38, 0.40
K1, K2, XB = 4.0, 0.8, 0.03
KXA_SED = 0.15                # PIEDI TUCKED sotto il bacino (task 0.1-quinquies):
                              # agli SW (-108) la china non basta a portare il CoM sui
                              # piedi se l'anca e' 17 cm dietro le caviglie. Finestra
                              # PASS misurata: 0,05-0,25 [C]. Era 0,55 (gambe distese).
KC, KCD, KP, KD = 3.0, 0.5, 3.0, 0.4
ZA_SED = 0.46                 # profondita' di seduta (bacino ~0,51 m)
T_GIU, T_SED, T_PREP, T_SU, T_FIN = 3.0, 2.0, 1.5, 3.0, 2.5
LEAN_GIU, LEAN_SU = 1.4, 2.44

def genera_sedia():
    s = open(XML).read()
    sedia = ('    <geom name="sedia" type="box" pos="-0.30 0 0.42" '
             'size="0.225 0.22 0.03" contype="1" conaffinity="1" '
             'rgba="0.55 0.38 0.22 1"/>\n  </worldbody>')
    assert '</worldbody>' in s
    s = s.replace('  </worldbody>', sedia, 1)
    v = '<body name="pelvis" pos="0 0 0.912">\n      <freejoint/>\n      <geom type="box" size="0.10 0.13 0.06" mass="5.4"/>'
    n = '<body name="pelvis" pos="0 0 0.912">\n      <freejoint/>\n      <geom name="pelvi_geom" type="box" size="0.10 0.13 0.06" mass="5.4" contype="1" conaffinity="1"/>'
    assert v in s
    s = s.replace(v, n, 1)
    for L in ('r', 'l'):
        v = ('<joint name="' + L + '_hip_pitch" axis="0 1 0" range="-1.885 0.4712"/>\n'
             '        <geom type="capsule" fromto="0 0 0 0 0 -0.38" size="0.055" mass="6.8"/>')
        n = ('<joint name="' + L + '_hip_pitch" axis="0 1 0" range="-1.885 0.4712"/>\n'
             '        <geom name="' + L + '_coscia" type="capsule" fromto="0 0 0 0 0 -0.38" '
             'size="0.055" mass="6.8" contype="1" conaffinity="1"/>')
        assert v in s
        s = s.replace(v, n, 1)
    s = s.replace('TX34_NOVA_v1', 'TX34_NOVA_v1_sedia')
    open(_DER(XML_SEDIA), 'w').write(s)

def ik(xa, za):
    D = np.hypot(xa, za)
    D = min(D, L1 + L2 - 1e-4)
    gi = np.arccos(np.clip((L1*L1 + L2*L2 - D*D)/(2*L1*L2), -1, 1))
    qk = np.pi - gi
    beta = np.arccos(np.clip((L1*L1 + D*D - L2*L2)/(2*L1*D), -1, 1))
    return -(np.arctan2(xa, za) + beta), qk

def smooth(u):
    return 0.5 - 0.5*np.cos(np.pi*np.clip(u, 0, 1))

def profilo(t):
    """Ritorna (za, lean_ff, com_attivo, fase)."""
    if t < T_GIU:
        u = smooth(t/T_GIU)
        za = 0.78 + (ZA_SED - 0.78)*u
        return za, LEAN_GIU*(0.78 - za), False, 'discesa'
    t -= T_GIU
    if t < T_SED:
        return ZA_SED, 0.15, False, 'seduta'
    t -= T_SED
    if t < T_PREP:
        u = smooth(t/T_PREP)
        return ZA_SED, 0.15 + (LEAN_SU*(0.78 - ZA_SED) - 0.15)*u, False, 'prep'
    t -= T_PREP
    if t < T_SU:
        u = smooth(t/T_SU)
        za = ZA_SED + (0.78 - ZA_SED)*u
        return za, LEAN_SU*(0.78 - za), True, 'risalita'
    return 0.78, 0.0, True, 'eretta'

_BIL = {'t': [], 's': [], 'p': []}
def _bilancio_step(m, d):
    import numpy as _np
    fs = fp = 0.0
    for _i in range(d.ncon):
        _c = d.contact[_i]
        _f6 = _np.zeros(6); mujoco.mj_contactForce(m, d, _i, _f6)
        _Fw = _np.array(_c.frame).reshape(3, 3).T @ _f6[:3]
        _n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, _c.geom1) or ''
        _n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, _c.geom2) or ''
        if 'sedia' in _n1 or 'sedia' in _n2: fs += abs(_Fw[2])
        else: fp += abs(_Fw[2])
    _BIL['t'].append(float(d.time)); _BIL['s'].append(fs); _BIL['p'].append(fp)

def bilancio_chiusura():
    import numpy as _np
    t = _np.array(_BIL['t']); S = _np.array(_BIL['s']); P = _np.array(_BIL['p'])
    m = (t >= 4.0) & (t < 5.0)     # ultimo secondo della fase seduta - plateau (T_GIU=3, T_SED=2)
    sed, pie = float(_np.median(S[m])), float(_np.median(P[m]))
    tot, mg = sed + pie, 649.7
    chi = 100.0 * tot / mg
    print('[BILANCIO Fz, finestra 4,0-5,0 s] seduta %.0f N + piedi %.0f N = %.0f N vs m*g %.1f N -> chiusura %.1f%%'
          % (sed, pie, tot, mg, chi))
    assert 97.0 <= chi <= 103.0, 'BILANCIO NON CHIUDE (%.1f%%): numero non pubblicabile' % chi
    return sed, pie

def esegui(registra=False):
    genera_sedia()
    m = mujoco.MjModel.from_xml_path(_DER(XML_SEDIA))
    total_mass = m.body_mass.sum()
    print(f"[MASSA MODELLO] {total_mass:.3f} kg")
    assert 65.5 < total_mass < 67.0, f"MASSA FUORI RANGE: {total_mass:.3f} kg - verifica XML"
    d = mujoco.MjData(m)
    aid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i][0]): i
           for i in range(m.nu)}
    fid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ('r_foot', 'l_foot')]
    gid_sedia = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'sedia')
    jq = dict((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j), m.jnt_qposadr[j])
              for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    cp = d.subtree_com[0].copy()
    T = T_GIU + T_SED + T_PREP + T_SU + T_FIN
    z_min = 1.0
    forza_sedia = 0.0
    _fs_acc = []
    picchi = dict(knee=0.0, hip=0.0, ankle=0.0)
    t_alzata = None
    seduta_ok = False
    tr = dict(t=[], z=[], fs=[], tk=[], th=[], fase=[])
    for i in range(int(T/dt)):
        t = i*dt
        za, ref_ff, com_att, fase = profilo(t)
        com = d.subtree_com[0]
        v = (com - cp)/dt
        cp = com.copy()
        qw, qx, qy, qz = d.qpos[3:7]
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))
        wy = float(d.qvel[4])
        feet = 0.5*(d.xpos[fid[0]] + d.xpos[fid[1]])
        xa = KXA_SED*(0.78 - za)
        qh, qk = ik(xa, za)
        dx = com[0] - feet[0] - XB
        ref = ref_ff - (KC*dx + KCD*v[0] if com_att else 0.0)
        ref = float(np.clip(ref, -0.20, 0.95))
        tau = KP*(pitch - ref) + KD*wy
        qh_cmd = float(np.clip(qh + tau, -1.885, 0.4712))   # SW anca (-108/+27)
        liv = -(qh + qk)
        trim = float(np.clip(K1*dx + K2*v[0], -0.20, 0.20)) if com_att else 0.0
        a_cmd = float(np.clip(liv + trim, -0.47, 0.47))
        ar = float(np.clip(-(K1*(com[1]-feet[1]) + K2*v[1]), -0.20, 0.20))
        for lato in ('r', 'l'):
            d.ctrl[aid[lato+'_hip_pitch']] = qh_cmd
            d.ctrl[aid[lato+'_knee']] = qk
            d.ctrl[aid[lato+'_ankle_pitch']] = a_cmd
            d.ctrl[aid[lato+'_ankle_roll']] = ar
            d.ctrl[aid[lato+'_hip_roll']] = 0.0
        mujoco.mj_step(m, d)
        _bilancio_step(m, d)
        fs = 0.0
        for c in range(d.ncon):
            if gid_sedia in (d.contact[c].geom1, d.contact[c].geom2):
                f6 = np.zeros(6)
                mujoco.mj_contactForce(m, d, c, f6)
                fs += abs(float(f6[0]))
        forza_sedia = max(forza_sedia, fs)
        if fase == 'seduta' and t > T_GIU + T_SED - 1.0:
            _fs_acc.append(fs)
            if fs > 150.0:
                seduta_ok = True
        if fase in ('risalita', 'eretta'):
            for nome, ch in (('knee', 'r_knee'), ('hip', 'r_hip_pitch'),
                             ('ankle', 'r_ankle_pitch')):
                f = max(abs(float(d.actuator_force[aid[ch]])),
                        abs(float(d.actuator_force[aid['l'+ch[1:]]])))
                picchi[nome] = max(picchi[nome], f)
        z_min = min(z_min, float(d.qpos[2]))
        if (t_alzata is None and fase == 'eretta' and d.qpos[2] >= 0.88
                and abs(np.degrees(pitch)) < 9.0):
            t_alzata = t - (T_GIU + T_SED + T_PREP)
        if registra and i % 4 == 0:
            tr['t'].append(t)
            tr['z'].append(float(d.qpos[2]))
            tr['fs'].append(fs)
            tr['tk'].append(float(d.actuator_force[aid['r_knee']]))
            tr['th'].append(float(d.actuator_force[aid['r_hip_pitch']]))
            tr['fase'].append(fase)
        if d.qpos[2] < 0.30:
            return dict(caduta=True, t_caduta=round(t, 1))
    qw, qx, qy, qz = d.qpos[3:7]
    pitch_fin = abs(np.degrees(np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))))
    eretta = d.qpos[2] >= 0.88 and pitch_fin < 9.0
    out = dict(caduta=False, seduta_verificata=bool(seduta_ok),
               tornata_eretta=bool(eretta),
               z_min_bacino_m=round(z_min, 3),
               forza_sedia_N=round(float(np.mean(_fs_acc)) if _fs_acc else 0.0, 0),
               peso_scaricato_pct=round((100.0*float(np.mean(_fs_acc))/(66.23*9.81)) if _fs_acc else 0.0, 0),
               t_alzata_s=(round(t_alzata, 2) if t_alzata is not None else None),
               picco_ginocchio_Nm=round(picchi['knee'], 1),
               picco_anca_Nm=round(picchi['hip'], 1),
               picco_caviglia_Nm=round(picchi['ankle'], 1))
    if registra:
        out['tracce'] = tr
    return out

def campagna():
    print('f1d_seduta.py - versione ' + VERSIONE)
    print('F1-D: sedersi su sedia 45 cm e rialzarsi (par.12-T2). Limiti par.2.3: 120/120/80 Nm')
    r = esegui(registra=True)
    if r['caduta']:
        print('CADUTA a %.1f s -> FAIL' % r['t_caduta'])
        esito = 'FAIL'
        json.dump(dict(versione=VERSIONE, esito=esito, caduta_s=r['t_caduta']),
                  open(_OUT('f1d_seduta.json'), 'w'), indent=1)
        return
    ok = (r['seduta_verificata'] and r['tornata_eretta']
          and r['picco_ginocchio_Nm'] < 120 and r['picco_anca_Nm'] < 120
          and r['picco_caviglia_Nm'] < 80)
    esito = 'PASS' if ok else 'FAIL'
    print('Seduta verificata: %s (forza sedia %.0f N = %.0f%% del peso) | '
          'bacino min %.3f m' % ('SI' if r['seduta_verificata'] else 'NO',
                                 r['forza_sedia_N'], r['peso_scaricato_pct'],
                                 r['z_min_bacino_m']))
    print('Rialzata: %s in %.2f s | picchi Nm ginocchio %.1f / anca %.1f / caviglia %.1f'
          % ('SI' if r['tornata_eretta'] else 'NO',
             r['t_alzata_s'] if r['t_alzata_s'] else -1,
             r['picco_ginocchio_Nm'], r['picco_anca_Nm'], r['picco_caviglia_Nm']))
    bilancio_chiusura()
    print('ESITO F1-D: ' + esito)
    json.dump(dict(versione=VERSIONE, esito=esito,
                   **dict((k, v) for k, v in r.items() if k != 'tracce')),
              open(_OUT('f1d_seduta.json'), 'w'), indent=1)
    print('Salvato f1d_seduta.json')
    tr = r['tracce']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axs[0].plot(tr['t'], tr['z'], color='tab:purple', lw=1.2)
    axs[0].axhline(0.88, color='g', ls='--', lw=0.8, label='soglia eretta')
    axs[0].set_ylabel('quota bacino [m]')
    axs[0].set_title('F1-D sedia 45 cm: seduta %.0f N (%.0f%% peso), rialzo %.2f s -> %s'
                     % (r['forza_sedia_N'], r['peso_scaricato_pct'],
                        r['t_alzata_s'] or -1, esito))
    axs[0].legend(fontsize=8)
    axs[1].plot(tr['t'], tr['fs'], color='tab:brown', lw=1.2)
    axs[1].set_ylabel('forza sulla sedia [N]')
    axs[2].plot(tr['t'], tr['tk'], color='tab:blue', lw=1.2, label='ginocchio')
    axs[2].plot(tr['t'], tr['th'], color='tab:green', lw=1.2, label='anca pitch')
    axs[2].axhline(120, color='r', ls='--', lw=1)
    axs[2].axhline(-120, color='r', ls='--', lw=1, label='limite RMD-X12')
    axs[2].set_ylabel('coppia [Nm]')
    axs[2].set_xlabel('tempo [s]')
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_OUT('f1d_seduta.png'), dpi=150)
    print('Salvato f1d_seduta.png')

def demo():
    import time
    import mujoco.viewer
    print(__doc__)
    genera_sedia()
    m = mujoco.MjModel.from_xml_path(_DER(XML_SEDIA))
    total_mass = m.body_mass.sum()
    print(f"[MASSA MODELLO] {total_mass:.3f} kg")
    assert 65.5 < total_mass < 67.0, f"MASSA FUORI RANGE: {total_mass:.3f} kg - verifica XML"
    d = mujoco.MjData(m)
    aid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i][0]): i
           for i in range(m.nu)}
    fid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ('r_foot', 'l_foot')]
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    cp = d.subtree_com[0].copy()
    T = T_GIU + T_SED + T_PREP + T_SU + T_FIN
    t = 0.0
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            t0 = time.time()
            za, ref_ff, com_att, fase = profilo(t % T)
            com = d.subtree_com[0]
            vv = (com - cp)/dt
            cp = com.copy()
            qw, qx, qy, qz = d.qpos[3:7]
            pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))
            wy = float(d.qvel[4])
            feet = 0.5*(d.xpos[fid[0]] + d.xpos[fid[1]])
            xa = KXA_SED*(0.78 - za)
            qh, qk = ik(xa, za)
            dx = com[0] - feet[0] - XB
            ref = ref_ff - (KC*dx + KCD*vv[0] if com_att else 0.0)
            ref = float(np.clip(ref, -0.20, 0.95))
            tau = KP*(pitch - ref) + KD*wy
            qh_cmd = float(np.clip(qh + tau, -1.885, 0.4712))   # SW anca (-108/+27)
            liv = -(qh + qk)
            trim = float(np.clip(K1*dx + K2*vv[0], -0.20, 0.20)) if com_att else 0.0
            a_cmd = float(np.clip(liv + trim, -0.47, 0.47))
            ar = float(np.clip(-(K1*(com[1]-feet[1]) + K2*vv[1]), -0.20, 0.20))
            for lato in ('r', 'l'):
                d.ctrl[aid[lato+'_hip_pitch']] = qh_cmd
                d.ctrl[aid[lato+'_knee']] = qk
                d.ctrl[aid[lato+'_ankle_pitch']] = a_cmd
                d.ctrl[aid[lato+'_ankle_roll']] = ar
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
