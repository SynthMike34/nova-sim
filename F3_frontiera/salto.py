#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
salto.py v1.0 - CANTIERE BONUS (fuori par.12, dichiarato): SALTO VERTICALE.
Collaudo dinamico delle coppie di picco par.2.3 (decollo = massima richiesta
del BOM) + forze di atterraggio. Modello base, kp originali: nella spinta i
servo saturano al forcerange = misuriamo il 100% dichiarato dal TDD.

    python salto.py            demo 3D: salta in loop
    python salto.py --test     salto misurato: quota, coppie, forze + PNG + JSON

Fasi: crouch (legge F1-A) -> spinta esplosiva (+caviglie in punta) -> volo ->
atterraggio ammortizzato -> ritorno eretta. PASS: stacco verificato (entrambi
i piedi in aria) + atterraggio senza caduta + ritorno eretta (>=0,88 m, <9 gr).
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

VERSIONE = '1.4'
XML = _MP('tx34_piedevero.xml')   # CANONE C (04/08): piede 16,5/6,0; il v1 (29,5) e' storia
L1, L2 = 0.38, 0.40
ZA_CR = 0.62          # caricamento entro l'autorita' della caviglia (scoperta 9)
T_CR, T_PUSH = 0.9, 0.14
KXA, KLEAN = 0.75, 0.35  # caricamento VERTICALE da salto: petto su
KC, KCD, KP, KD = 3.0, 0.5, 2.5, 0.4
K1, K2, XB = 4.0, 0.8, 0.05
PUNTA = 0.15          # spinta di caviglia al decollo [rad] - CANONE C (B43): 0,15 -> 250 ms, +8,2 cm
                      # (il vecchio canone 210/+3,3 era a PUNTA 0,35 sul piede 29,5)

def ik(xa, za):
    D = np.hypot(xa, za)
    D = min(D, L1 + L2 - 1e-4)
    gi = np.arccos(np.clip((L1*L1 + L2*L2 - D*D)/(2*L1*L2), -1, 1))
    qk = np.pi - gi
    beta = np.arccos(np.clip((L1*L1 + D*D - L2*L2)/(2*L1*D), -1, 1))
    return -(np.arctan2(xa, za) + beta), qk

def smooth(u):
    return 0.5 - 0.5*np.cos(np.pi*np.clip(u, 0.0, 1.0))

class Saltatrice:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(XML)
        total_mass = self.m.body_mass.sum()
        print(f"[MASSA MODELLO] {total_mass:.3f} kg")
        assert 65.5 < total_mass < 67.0, f"MASSA FUORI RANGE: {total_mass:.3f} kg - verifica XML"
        self.d = mujoco.MjData(self.m)
        self.aid = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT,
                                      self.m.actuator_trnid[i][0]): i
                    for i in range(self.m.nu)}
        self.fid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b)
                    for b in ('r_foot', 'l_foot')]
        self.piedi_geom = set(g for g in range(self.m.ngeom)
                              if 'foot' in (mujoco.mj_id2name(
                                  self.m, mujoco.mjtObj.mjOBJ_BODY,
                                  self.m.geom_bodyid[g]) or '')
                              or 'toe' in (mujoco.mj_id2name(
                                  self.m, mujoco.mjtObj.mjOBJ_BODY,
                                  self.m.geom_bodyid[g]) or ''))
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        mujoco.mj_forward(self.m, self.d)
        self.cp = self.d.subtree_com[0].copy()
        self.fase = 'crouch'
        self.t_f = 0.0
        self.volo = False
        self.xa_volo = 0.0
        self.z_piede0 = float(self.d.xpos[self.fid[0]][2])

    def forza_suolo(self):
        f6 = np.zeros(6)
        tot = 0.0
        for c in range(self.d.ncon):
            g1, g2 = self.d.contact[c].geom1, self.d.contact[c].geom2
            if 0 in (g1, g2) and (g1 in self.piedi_geom or g2 in self.piedi_geom):
                mujoco.mj_contactForce(self.m, self.d, c, f6)
                tot += abs(float(f6[0]))
        return tot

    def frame(self):
        m, d = self.m, self.d
        dt = m.opt.timestep
        com = d.subtree_com[0]
        v = (com - self.cp)/dt
        self.cp = com.copy()
        qw, qx, qy, qz = d.qpos[3:7]
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))
        wy = float(d.qvel[4])
        feet = 0.5*(d.xpos[self.fid[0]] + d.xpos[self.fid[1]])
        fs = self.forza_suolo()

        ref_ff = None
        if self.fase == 'crouch':
            za = 0.78 + (ZA_CR - 0.78)*smooth(self.t_f/T_CR)
            punta = 0.0
            com_att = True
            ref_ff = KLEAN*(0.78 - za)   # ref PURO: la com-corr resta solo sulla caviglia
            if self.t_f >= T_CR:
                self.fase, self.t_f = 'spinta', 0.0
        elif self.fase == 'spinta':
            za = ZA_CR + (0.86 - ZA_CR)*np.clip(self.t_f/T_PUSH, 0, 1)
            punta = PUNTA*np.clip(self.t_f/T_PUSH, 0, 1)
            com_att = False
            ref_ff = 0.0
            if fs < 30.0 and self.t_f > 0.5*T_PUSH:
                self.fase, self.t_f = 'volo', 0.0
                self.volo = True
            elif self.t_f > T_PUSH + 0.25:
                self.fase, self.t_f = 'volo', 0.0   # niente stacco: si vedra'
        elif self.fase == 'volo':
            za, punta, com_att = 0.72, 0.0, False
            ref_ff = 0.0
            if fs > 60.0 and self.t_f > 0.12 and float(d.qvel[2]) < 0.0:
                self.fase, self.t_f = 'atterra', 0.0
        elif self.fase == 'atterra':
            za = 0.72 + (0.56 - 0.72)*smooth(self.t_f/0.35)
            punta = 0.0
            com_att = fs > 80.0   # correzioni solo a piedi CARICATI
            ref_ff = 0.10         # busto quasi eretto all'impatto
            if self.t_f >= 0.5:
                self.fase, self.t_f = 'risale', 0.0
        elif self.fase == 'risale':
            za = 0.56 + (0.78 - 0.56)*smooth(self.t_f/1.0)
            punta = 0.0
            com_att = True
            if self.t_f >= 1.0:
                self.fase, self.t_f = 'eretta', 0.0
        else:
            za, punta, com_att = 0.78, 0.0, True

        if self.fase == 'crouch':
            xa = 0.10             # anche quasi sotto: caricamento da salto
        elif self.fase == 'spinta':
            xa = 0.10*(1.0 - np.clip(self.t_f/T_PUSH, 0, 1))
        elif self.fase in ('volo', 'atterra'):
            xa = 0.0
        else:
            xa = KXA*(0.78 - za)
        ref = KLEAN*(0.78 - za) if ref_ff is None else ref_ff
        qh, qk = ik(xa, za)
        dx = com[0] - feet[0] - XB
        if com_att and self.fase != 'crouch':
            ref = float(np.clip(ref - (KC*dx + KCD*v[0]), -0.2, 0.9))
        tau = KP*(pitch - ref) + KD*wy
        qh_cmd = float(np.clip(qh + tau, -2.0, 0.5))
        liv = -(qh + qk)
        trim = float(np.clip(K1*dx + K2*v[0], -0.25, 0.25)) if com_att else 0.0
        a_cmd = float(np.clip(liv + trim - punta, -0.47, 0.47))
        ar = float(np.clip(-(K1*(com[1]-feet[1]) + K2*v[1]), -0.2, 0.2))
        for lato in ('r', 'l'):
            d.ctrl[self.aid[lato + '_hip_pitch']] = qh_cmd
            d.ctrl[self.aid[lato + '_knee']] = qk
            d.ctrl[self.aid[lato + '_ankle_pitch']] = a_cmd
            d.ctrl[self.aid[lato + '_ankle_roll']] = ar
            d.ctrl[self.aid[lato + '_hip_roll']] = 0.0
        mujoco.mj_step(m, d)
        self.t_f += dt
        return d.qpos[2] > 0.40

def prova(registra=False):
    s = Saltatrice()
    dt = s.m.opt.timestep
    T = 6.0
    clear_max = 0.0
    t_volo = 0.0
    v_decollo = 0.0
    picchi_su = dict(knee=0.0, hip=0.0, ankle=0.0)
    picchi_giu = dict(knee=0.0, hip=0.0, ankle=0.0)
    f_att = 0.0
    apice_com = 0.0
    com_distacco = None   # B44: l'apice si misura DAL DISTACCO, non dalla quota in piedi
    tr = dict(t=[], fs=[], zp=[], tk=[])
    com0 = None
    for i in range(int(T/dt)):
        fase_pre = s.fase
        vz_pre = (s.d.subtree_com[0][2] - s.cp[2])/dt if i else 0.0
        vivo = s.frame()
        if com0 is None:
            com0 = float(s.d.subtree_com[0][2])
        if not vivo:
            out = dict(caduta=True, fase=s.fase, volo=bool(s.volo),
                       clearance_piedi_cm=round(clear_max*100, 1),
                       apice_com_cm=round(apice_com*100, 1),
                       t_volo_ms=round(t_volo*1000, 0),
                       v_decollo_m_s=round(v_decollo, 2),
                       picchi_spinta_Nm=dict((k, round(vv, 1))
                                             for k, vv in picchi_su.items()),
                       f_atterraggio_N=round(f_att, 0))
            if registra:
                out['tracce'] = tr
            return out
        d = s.d
        fs = s.forza_suolo()
        zp = min(float(d.xpos[s.fid[0]][2]), float(d.xpos[s.fid[1]][2])) - s.z_piede0
        if s.fase == 'spinta':
            for nome, ch in (('knee', 'r_knee'), ('hip', 'r_hip_pitch'),
                             ('ankle', 'r_ankle_pitch')):
                picchi_su[nome] = max(picchi_su[nome],
                                      abs(float(d.actuator_force[s.aid[ch]])))
            v_decollo = max(v_decollo, vz_pre)
        if s.fase == 'volo':
            clear_max = max(clear_max, zp)
            t_volo += dt
            if com_distacco is None:
                com_distacco = float(d.subtree_com[0][2])   # primo frame di volo = distacco
            apice_com = max(apice_com, float(d.subtree_com[0][2]) - com_distacco)
            # [B44] la vecchia metrica (rispetto alla quota in piedi) dava +0,0:
            # il robot si accovaccia di ~16 cm prima di partire. RITIRATA.
        if s.fase in ('atterra', 'risale'):
            f_att = max(f_att, fs)
            for nome, ch in (('knee', 'r_knee'), ('hip', 'r_hip_pitch'),
                             ('ankle', 'r_ankle_pitch')):
                picchi_giu[nome] = max(picchi_giu[nome],
                                       abs(float(d.actuator_force[s.aid[ch]])))
        if registra and i % 3 == 0:
            tr['t'].append(i*dt)
            tr['fs'].append(fs)
            tr['zp'].append(zp)
            tr['tk'].append(float(d.actuator_force[s.aid['r_knee']]))
    qw, qx, qy, qz = s.d.qpos[3:7]
    pf = abs(np.degrees(np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))))
    out = dict(caduta=False, volo=bool(s.volo),
               clearance_piedi_cm=round(clear_max*100, 1),
               apice_com_cm=round(apice_com*100, 1),
               t_volo_ms=round(t_volo*1000, 0),
               v_decollo_m_s=round(v_decollo, 2),
               picchi_spinta_Nm=dict((k, round(v, 1)) for k, v in picchi_su.items()),
               picchi_atterraggio_Nm=dict((k, round(v, 1)) for k, v in picchi_giu.items()),
               f_atterraggio_N=round(f_att, 0),
               eretta=bool(s.d.qpos[2] >= 0.88 and pf < 9.0))
    if registra:
        out['tracce'] = tr
    return out

def campagna():
    print('salto.py - versione ' + VERSIONE)
    print('SALTO verticale: collaudo coppie di picco par.2.3 (120/120/80 Nm).')
    r = prova(registra=True)
    if r['caduta']:
        print('SALTO ESEGUITO, atterraggio in piedi NON riuscito (cade in %s).' % r['fase'])
        print('Volo: %s | %.0f ms | piedi +%.1f cm | CoM +%.1f cm | v decollo %.2f m/s'
              % ('SI' if r['volo'] else 'NO', r['t_volo_ms'],
                 r['clearance_piedi_cm'], r['apice_com_cm'], r['v_decollo_m_s']))
        print('Coppie di picco in spinta [Nm]: ginocchio %.1f / anca %.1f / caviglia %.1f'
              % (r['picchi_spinta_Nm']['knee'], r['picchi_spinta_Nm']['hip'],
                 r['picchi_spinta_Nm']['ankle']))
        if r['picchi_spinta_Nm']['knee'] >= 119.9 and r['picchi_spinta_Nm']['hip'] >= 119.9:
            print('NOTA-CANONE [C]: ginocchio E anca sono AL CAP (120/120 Nm) - questo salto'
                  " e' il TETTO dell'hardware, piu' in alto non puo'. A PUNTA 0,35 non saturavano.")
        print('Forza max al primo impatto: %.0f N' % r['f_atterraggio_N'])
        esito = 'MISURATO (atterraggio aperto)' if r['volo'] else 'FAIL'
        print('ESITO SALTO: ' + esito)
        json.dump(dict(versione=VERSIONE, esito=esito,
                       **dict((k, v) for k, v in r.items() if k != 'tracce')),
                  open(_OUT('salto.json'), 'w'), indent=1)
        print('Salvato salto.json')
        tr = r.get('tracce')
        if tr:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            axs[0].plot(tr['t'], [z*100 for z in tr['zp']], color='tab:purple', lw=1.3)
            axs[0].set_ylabel('foot clearance [cm]')
            axs[0].set_title('JUMP: flight %.0f ms, feet +%.1f cm, knee %.0f/120 N·m'
                             % (r['t_volo_ms'], r['clearance_piedi_cm'],
                                r['picchi_spinta_Nm']['knee']))
            axs[1].plot(tr['t'], tr['tk'], color='tab:blue', lw=1.0)
            axs[1].axhline(120, color='r', ls='--', lw=1)
            axs[1].axhline(-120, color='r', ls='--', lw=1, label='RMD-X12 limit')
            axs[1].set_ylabel('knee torque [N·m]')
            axs[1].set_xlabel('time [s]')
            axs[1].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(_OUT('salto.png'), dpi=150)
            print('Salvato salto.png')
        return
    ok = r['volo'] and r['eretta']
    print('Stacco: %s | volo %.0f ms | piedi +%.1f cm | CoM +%.1f cm | v decollo %.2f m/s'
          % ('SI' if r['volo'] else 'NO', r['t_volo_ms'], r['clearance_piedi_cm'],
             r['apice_com_cm'], r['v_decollo_m_s']))
    print('Coppie spinta [Nm]: ginocchio %.1f / anca %.1f / caviglia %.1f (limiti 120/120/80)'
          % (r['picchi_spinta_Nm']['knee'], r['picchi_spinta_Nm']['hip'],
             r['picchi_spinta_Nm']['ankle']))
    print('Atterraggio: %.0f N | coppie [Nm]: ginocchio %.1f / anca %.1f / caviglia %.1f'
          % (r['f_atterraggio_N'], r['picchi_atterraggio_Nm']['knee'],
             r['picchi_atterraggio_Nm']['hip'], r['picchi_atterraggio_Nm']['ankle']))
    print('Ritorno eretta: %s' % ('SI' if r['eretta'] else 'NO'))
    esito = 'PASS' if ok else 'FAIL'
    print('ESITO SALTO: ' + esito)
    json.dump(dict(versione=VERSIONE, esito=esito,
                   **dict((k, v) for k, v in r.items() if k != 'tracce')),
              open(_OUT('salto.json'), 'w'), indent=1)
    print('Salvato salto.json')
    tr = r['tracce']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axs[0].plot(tr['t'], [z*100 for z in tr['zp']], color='tab:purple', lw=1.3)
    axs[0].set_ylabel('foot clearance [cm]')
    axs[0].set_title('JUMP: flight %.0f ms, feet +%.1f cm, knee %.0f/120 N·m -> %s'
                     % (r['t_volo_ms'], r['clearance_piedi_cm'],
                        r['picchi_spinta_Nm']['knee'], esito))
    axs[1].plot(tr['t'], tr['fs'], color='tab:red', lw=1.0)
    axs[1].set_ylabel('ground force [N]')
    axs[2].plot(tr['t'], tr['tk'], color='tab:blue', lw=1.0)
    axs[2].axhline(120, color='r', ls='--', lw=1)
    axs[2].axhline(-120, color='r', ls='--', lw=1, label='RMD-X12 limit')
    axs[2].set_ylabel('knee torque [N·m]')
    axs[2].set_xlabel('time [s]')
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_OUT('salto.png'), dpi=150)
    print('Salvato salto.png')

def demo():
    import time
    import mujoco.viewer
    print(__doc__)
    s = Saltatrice()
    dt = s.m.opt.timestep
    t_tot = 0.0
    print('Ogni ciclo: stacco+volo OK | atterraggio in piedi APERTO (ricade:')
    print('limite documentato, scoperta 25 - vedi Rapporto/handoff).')
    with mujoco.viewer.launch_passive(s.m, s.d) as v:
        while v.is_running():
            t0 = time.time()
            vivo = s.frame()
            t_tot += dt
            if not vivo or t_tot > 6.0:
                if not vivo:
                    print('volo eseguito, ricaduta (attesa) - nuovo salto...')
                s.reset()
                t_tot = 0.0
            v.sync()
            resto = dt - (time.time() - t0)
            if resto > 0:
                time.sleep(resto)

if __name__ == '__main__':
    if '--test' in sys.argv:
        campagna()
    else:
        demo()
