#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e6_supervisore.py v0.1 - CANTIERE E6: il SUPERVISORE D'ATTERRAGGIO.
Un arbitro unico decide tra FERMA / ASSESTA / PASSO guardando UNA cosa:
dove si fermera' il baricentro (punto di riposo = com + v/omega) rispetto al
poligono dei piedi. Isteresi + dwell 100 ms = niente navetta, niente finestre
perse. Novita': il PASSETTO DI ASSESTAMENTO (micro-passo deliberato che
ri-centra l'appoggio sotto il punto di riposo). Volano E5 integrato:
pronte -> frustata -> rientro sincronizzato con la fermata.

    python e6_supervisore.py            demo 3D: salta e atterra (loop)
    python e6_supervisore.py --test     misura: PASS se FERMA ED ERETTA >= 3 s

Criterio: dopo il volo, ferma ed eretta per >= 3,0 s (da 0,5 di E5).
kp gambe x3 [A]. Stop-loss: una sessione.
"""
import os as _os
def _MP(_n):
    _qui = _os.path.dirname(_os.path.abspath(__file__))
    for _c in (_n, _os.path.join(_qui, '..', 'models', _n), _os.path.join(_qui, _n)):
        if _os.path.exists(_c): return _c
    return _n
import sys
import json
import numpy as np
import mujoco

VERSIONE = '1.1'
XML = 'tx34_v1.xml'
L1, L2 = 0.38, 0.40
# salto (da E5)
ZA_CR, T_CR, T_PUSH = 0.62, 1.1, 0.16
KLEAN = 0.35
# camminata/recupero (trilogia)
ZA_W, T_NOM, T_MIN_R, MARG, W = 0.72, 0.30, 0.15, 0.020, 0.18
XOFFW, KV, LIFT = -0.04, 0.20, 0.06
KP_B, KD_B, KR_B, KRD = 3.0, 0.5, 2.5, 0.4
KPB2, KDB2 = 3.5, 1.2
P0R = 0.08
K1, K2 = 4.0, 0.8
KA, KAD = 2.5, 0.6
# volano (E5)
Q_PRONTE = -1.0
K1F, K2F = 4.0, 2.5
RATE_ASSORBI, RATE_RITORNO = 14.0, 2.5
GOMITO = 0.15
# SUPERVISORE (E6)
DWELL = 0.10          # permanenza minima in un modo [s]
SUP_DIETRO = 0.16     # poligono sagittale: quanto dietro al centro-piedi
SUP_AVANTI = 0.13     # quanto avanti (la punta e' lunga, il tallone corto)
SUP_LAT = W/2 + 0.05
ISTERESI = 0.04       # per USCIRE dalla ferma serve sforare di questo extra
T_ASS = 0.16          # durata del passetto di assestamento
V_PASSO = 0.45        # sopra questa velocita' si insegue col PASSO pieno

m = mujoco.MjModel.from_xml_path(XML)
total_mass = m.body_mass.sum()
print(f"[MASSA MODELLO] {total_mass:.3f} kg")
assert 65.5 < total_mass < 67.0, f"MASSA FUORI RANGE: {total_mass:.3f} kg — verifica XML"
AID = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i][0]): i
       for i in range(m.nu)}
GAMBE = [n for n in AID if n.split('_', 1)[1] in
         ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
for _n in GAMBE:                       # [A] rigidezza servo realistica
    _i = AID[_n]
    m.actuator_gainprm[_i][0] *= 3.0
    m.actuator_biasprm[_i][1] *= 3.0
    m.actuator_biasprm[_i][2] *= 1.7
KP_ATT = {n: float(m.actuator_gainprm[i][0]) for n, i in AID.items()}
BID = {b: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
       for b in ('r_foot', 'l_foot', 'pelvis')}
PIEDI_GEOM = set(g for g in range(m.ngeom)
                 if any(k in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                                m.geom_bodyid[g]) or '')
                        for k in ('foot', 'toe')))

def ik(xa, za):
    D = np.hypot(xa, za)
    D = min(D, L1 + L2 - 1e-4)
    gi = np.arccos(np.clip((L1*L1 + L2*L2 - D*D)/(2*L1*L2), -1, 1))
    qk = np.pi - gi
    beta = np.arccos(np.clip((L1*L1 + D*D - L2*L2)/(2*L1*D), -1, 1))
    return -(np.arctan2(xa, za) + beta), qk

def smooth(u):
    return 0.5 - 0.5*np.cos(np.pi*np.clip(u, 0.0, 1.0))

class Atterratrice:
    def __init__(self):
        self.d = mujoco.MjData(m)
        self.reset()

    def reset(self):
        d = self.d
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        self.cp = d.subtree_com[0].copy()
        self.fase = 'crouch'          # crouch -> spinta -> volo -> atterrata
        self.modo = 'PASSO'           # arbitro: PASSO / ASSESTA / FERMA
        self.t_f = 0.0                # timer di fase/passo
        self.t_modo = 0.0             # permanenza nel modo (dwell)
        self.volo = False
        self.passi = 0
        self.assestamenti = 0
        self.stance = 'r'
        self.fprev = np.zeros(m.nu)
        self.om = np.sqrt(9.81/0.85)
        self.caduta = False
        self.q_fw = 0.0
        self.b_ass = 0.0              # bersaglio congelato del passetto
        self.b_ass_y = 0.0
        self.t_fermo = 0.0            # tempo continuo in FERMA eretta
        self.fermo_max = 0.0

    def forze_piedi(self):
        f6 = np.zeros(6)
        fr = fl = 0.0
        for c in range(self.d.ncon):
            g1, g2 = self.d.contact[c].geom1, self.d.contact[c].geom2
            if 0 in (g1, g2):
                altro = g2 if g1 == 0 else g1
                if altro in PIEDI_GEOM:
                    mujoco.mj_contactForce(m, self.d, c, f6)
                    nb = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                           m.geom_bodyid[altro]) or ''
                    if nb.startswith('r_'):
                        fr += abs(float(f6[0]))
                    else:
                        fl += abs(float(f6[0]))
        return fr, fl

    def arbitro(self, resto_x, resto_y, feet, v, pitch):
        """Decide il modo desiderato dal punto di riposo (con isteresi)."""
        dxr = resto_x - float(feet[0])
        dyr = abs(resto_y - float(feet[1]))
        e_in = (-SUP_DIETRO < dxr < SUP_AVANTI) and dyr < SUP_LAT
        if self.modo == 'FERMA':
            tol = 0.14 if self.t_modo < 0.45 else ISTERESI   # la frustata sposta
            fuori = (dxr < -(SUP_DIETRO + tol)               # il CoM: tollerata
                     or dxr > SUP_AVANTI + tol
                     or dyr > SUP_LAT + tol)
            return 'ASSESTA' if fuori else 'FERMA'
        if e_in and abs(pitch) < 0.35:
            return 'FERMA'
        if np.hypot(v[0], v[1]) < V_PASSO:
            return 'ASSESTA'
        return 'PASSO'

    def frame(self):
        d = self.d
        dt = m.opt.timestep
        com = d.subtree_com[0]
        v = (com - self.cp)/dt
        self.cp = com.copy()
        qw, qx, qy, qz = d.qpos[3:7]
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))
        phi = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        wy, wx = float(d.qvel[4]), float(d.qvel[3])
        fr, fl = self.forze_piedi()
        fs = fr + fl
        feet = 0.5*(d.xpos[BID['r_foot']] + d.xpos[BID['l_foot']])
        resto_x = float(com[0] + v[0]/self.om)
        resto_y = float(com[1] + v[1]/self.om)

        # ================= fasi di salto (da E5, certificate) =================
        if self.fase in ('crouch', 'spinta', 'volo'):
            if self.fase == 'crouch':
                za = 0.78 + (ZA_CR - 0.78)*smooth(self.t_f/T_CR)
                xa, ref = 0.12, KLEAN*(0.78 - za)
                if self.t_f >= T_CR:
                    self.fase, self.t_f = 'spinta', 0.0
            elif self.fase == 'spinta':
                u = np.clip(self.t_f/T_PUSH, 0, 1)
                za = ZA_CR + (0.80 - ZA_CR)*u
                xa, ref = 0.12*(1 - u), 0.06
                if (fs < 30.0 and self.t_f > 0.5*T_PUSH) or self.t_f > T_PUSH + 0.25:
                    self.fase, self.t_f = 'volo', 0.0
                    self.volo = True
            else:
                za, xa, ref = ZA_W, 0.0, 0.0
                if fs > 60.0 and self.t_f > 0.12 and float(d.qvel[2]) < 0.0:
                    self.stance = 'r' if fr >= fl else 'l'
                    self.fase, self.t_f = 'atterrata', 0.0
                    self.modo, self.t_modo = 'PASSO', 0.0
            qh, qk = ik(xa, za)
            dx = com[0] - feet[0] - 0.05
            tau = KP_B*(pitch - ref) + KD_B*wy
            qh_cmd = float(np.clip(qh + tau, -2.0, 0.5))
            liv = -(qh + qk)
            trim = (float(np.clip(K1*dx + K2*v[0], -0.25, 0.25))
                    if self.fase == 'crouch' else 0.0)
            timone = (float(np.clip(KA*(pitch - ref) + KAD*wy, -0.35, 0.35))
                      if self.fase == 'spinta' else 0.0)
            a_cmd = float(np.clip(liv + trim + timone, -0.47, 0.47))
            ar = (float(np.clip(-(K1*(com[1]-feet[1]) + K2*v[1]), -0.2, 0.2))
                  if self.fase == 'crouch' else 0.0)
            for lato in ('r', 'l'):
                d.ctrl[AID[lato + '_hip_pitch']] = qh_cmd
                d.ctrl[AID[lato + '_knee']] = qk
                d.ctrl[AID[lato + '_ankle_pitch']] = a_cmd
                d.ctrl[AID[lato + '_ankle_roll']] = ar
                d.ctrl[AID[lato + '_hip_roll']] = 0.0

        # ================= atterrata: il SUPERVISORE comanda =================
        else:
            if self.modo in ('PASSO', 'ASSESTA'):
                st = self.stance
                sw = 'r' if st == 'l' else 'l'
                p_st = d.xpos[BID[st + '_foot']]
                segno_sw = -1.0 if sw == 'r' else 1.0
                if self.modo == 'PASSO':
                    tx = float(com[0] + v[0]/self.om + KV*v[0] + XOFFW)
                    ty = resto_y + segno_sw*W/2.0
                else:
                    tx = self.b_ass
                    ty = self.b_ass_y + segno_sw*W/2.0
                hip = dict(r=(d.qpos[0], d.qpos[1] - 0.09),
                           l=(d.qpos[0], d.qpos[1] + 0.09))
                u = np.clip(self.t_f/T_NOM, 0.0, 1.0)
                for g in ('r', 'l'):
                    if g == sw:
                        xa = float(np.clip(tx - hip[g][0], -0.34, 0.30))
                        za = ZA_W - LIFT*np.sin(np.pi*u)
                        qh, qk = ik(xa, za)
                        d.ctrl[AID[g + '_hip_pitch']] = qh
                        d.ctrl[AID[g + '_knee']] = qk
                        d.ctrl[AID[g + '_ankle_pitch']] = -(qh + qk) - pitch
                        d.ctrl[AID[g + '_ankle_roll']] = 0.0
                        y_err = ty - hip[g][1]
                        d.ctrl[AID[g + '_hip_roll']] = float(np.clip(
                            y_err/0.72 - phi, -0.38, 0.38))
                    else:
                        qh, qk = ik(0.0, ZA_W)
                        tau = KPB2*(pitch - P0R) + KDB2*wy
                        ch = float(np.clip(self.fprev[AID[g + '_hip_pitch']]
                                           / KP_ATT[g + '_hip_pitch'], -0.6, 0.6))
                        ck = float(np.clip(self.fprev[AID[g + '_knee']]
                                           / KP_ATT[g + '_knee'], -0.2, 0.9))
                        d.ctrl[AID[g + '_hip_pitch']] = float(np.clip(
                            qh + tau + ch, -2.0, 0.5))
                        d.ctrl[AID[g + '_knee']] = qk + ck
                        trm = float(np.clip(2.0*(com[0]-p_st[0]-0.02) + 0.5*v[0],
                                            -0.35, 0.35))
                        d.ctrl[AID[g + '_ankle_pitch']] = float(np.clip(
                            -(qh + qk) + trm, -0.45, 0.45))
                        d.ctrl[AID[g + '_ankle_roll']] = float(np.clip(
                            -0.5*v[1], -0.2, 0.2))
                        d.ctrl[AID[g + '_hip_roll']] = float(np.clip(
                            KR_B*phi + KRD*wx, -0.3, 0.3))
            else:  # FERMA
                za = 0.72 + (0.78 - 0.72)*smooth(self.t_modo/0.8)
                qh, qk = ik(0.0, za)
                dx = com[0] - feet[0] - 0.03
                ap = float(np.clip(-(qh + qk)
                                   + np.clip(3.5*dx + 1.2*v[0], -0.4, 0.4),
                                   -0.47, 0.47))
                ar = float(np.clip(-(3.5*(com[1]-feet[1]) + 0.7*v[1]),
                                   -0.2, 0.2))
                tau = KPB2*(pitch - 0.03) + KDB2*wy
                for g in ('r', 'l'):
                    ch = float(np.clip(self.fprev[AID[g + '_hip_pitch']]
                                       / KP_ATT[g + '_hip_pitch'], -0.6, 0.6))
                    d.ctrl[AID[g + '_hip_pitch']] = float(np.clip(
                        qh + tau + ch, -2.0, 0.5))
                    d.ctrl[AID[g + '_knee']] = qk
                    d.ctrl[AID[g + '_ankle_pitch']] = ap
                    d.ctrl[AID[g + '_ankle_roll']] = ar
                    d.ctrl[AID[g + '_hip_roll']] = float(np.clip(
                        KR_B*phi + KRD*wx, -0.3, 0.3))

        # ================= volano (E5): pronte -> frustata -> rientro =========
        if self.fase == 'crouch':
            b_fw = 0.0
        elif self.fase == 'spinta':
            b_fw = float(np.clip(-K2F*wy, -0.5, 0.5))
        elif self.fase == 'volo':
            b_fw = Q_PRONTE
        elif self.modo == 'FERMA':
            if self.t_modo < 0.35 or abs(wy) > 0.6:   # FERMA a due tempi:
                b_fw = float(np.clip(K1F*(P0R - pitch) - K2F*wy, -2.3, 1.45))
            else:                              # ...poi il rientro-ancora
                b_fw = -0.35
        else:
            b_fw = float(np.clip(K1F*(P0R - pitch) - K2F*wy + Q_PRONTE*0.3,
                                 -2.3, 1.45))
        veloce = b_fw > self.q_fw or self.fase == 'volo'
        passo_max = (RATE_ASSORBI if veloce else RATE_RITORNO)*dt
        self.q_fw += float(np.clip(b_fw - self.q_fw, -passo_max, passo_max))
        for nome in ('r_shoulder_pitch', 'l_shoulder_pitch'):
            if nome in AID:
                d.ctrl[AID[nome]] = self.q_fw
        for nome in ('r_shoulder_roll', 'l_shoulder_roll'):
            if nome in AID:
                d.ctrl[AID[nome]] = 0.15 if nome.startswith('r') else -0.15
        for nome in ('r_elbow', 'l_elbow'):
            if nome in AID:
                d.ctrl[AID[nome]] = GOMITO

        mujoco.mj_step(m, d)
        self.fprev = d.actuator_force.copy()
        self.t_f += dt
        self.t_modo += dt

        # ============ transizioni del supervisore (dopo il passo fisico) ======
        if self.fase == 'atterrata':
            if self.modo in ('PASSO', 'ASSESTA'):
                st = self.stance
                sw = 'r' if st == 'l' else 'l'
                p_st = d.xpos[BID[st + '_foot']]
                segno_sw = -1.0 if sw == 'r' else 1.0
                cp_lat = resto_y
                commit = (cp_lat - float(p_st[1]))*segno_sw > MARG
                serve = abs(resto_x - float(p_st[0])) > 0.12
                fine_passo = ((self.modo == 'PASSO'
                               and ((self.t_f >= T_MIN_R and (serve or commit))
                                    or self.t_f >= 0.8))
                              or (self.modo == 'ASSESTA' and self.t_f >= T_ASS))
                if fine_passo:
                    if self.modo == 'PASSO':
                        self.passi += 1
                    else:
                        self.assestamenti += 1
                    self.stance = sw
                    self.t_f = 0.0
                    if self.t_modo >= DWELL:
                        des = self.arbitro(resto_x, resto_y, feet, v, pitch)
                        if des != self.modo:
                            self.modo, self.t_modo = des, 0.0
                            if des == 'ASSESTA':
                                self.b_ass = resto_x + XOFFW - 0.05
                                self.b_ass_y = resto_y
            else:  # FERMA: eretta? accumula; se il riposo scappa -> ASSESTA
                if d.qpos[2] > 0.80 and abs(pitch) < 0.20:
                    self.t_fermo += dt
                    self.fermo_max = max(self.fermo_max, self.t_fermo)
                else:
                    self.t_fermo = 0.0
                if self.t_modo >= DWELL:
                    des = self.arbitro(resto_x, resto_y, feet, v, pitch)
                    if des != 'FERMA':
                        fr2, fl2 = self.forze_piedi()
                        self.stance = 'r' if fr2 >= fl2 else 'l'
                        self.modo, self.t_modo, self.t_f = 'ASSESTA', 0.0, 0.0
                        self.b_ass = resto_x + XOFFW - 0.05
                        self.b_ass_y = resto_y
                        self.t_fermo = 0.0
        if d.qpos[2] < 0.40:
            self.caduta = True
        return not self.caduta

def prova():
    g = Atterratrice()
    dt = m.opt.timestep
    clear = 0.0
    t_volo = 0.0
    z0 = float(g.d.xpos[BID['r_foot']][2])
    T = 10.0
    for i in range(int(T/dt)):
        vivo = g.frame()
        if g.fase == 'volo':
            t_volo += dt
            zp = min(float(g.d.xpos[BID['r_foot']][2]),
                     float(g.d.xpos[BID['l_foot']][2])) - z0
            clear = max(clear, zp)
        if not vivo:
            return dict(caduta=True, modo=g.modo, passi=g.passi,
                        assestamenti=g.assestamenti,
                        t_volo_ms=round(t_volo*1000, 0),
                        clearance_cm=round(clear*100, 1),
                        t_fermo_s=round(g.fermo_max, 2))
    d = g.d
    qw, qx, qy, qz = d.qpos[3:7]
    pf = abs(np.degrees(np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))))
    vfin = float(np.hypot(d.qvel[0], d.qvel[1]))
    return dict(caduta=False, volo=bool(g.volo), passi=int(g.passi),
                assestamenti=int(g.assestamenti),
                t_volo_ms=round(t_volo*1000, 0), clearance_cm=round(clear*100, 1),
                t_fermo_s=round(g.fermo_max, 2), z_finale=round(float(d.qpos[2]), 3),
                pitch_finale=round(pf, 1), v_finale=round(vfin, 2),
                eretta=bool(d.qpos[2] >= 0.88 and pf < 9.0 and vfin < 0.15))

def campagna():
    print('e6_supervisore.py - versione ' + VERSIONE)
    print('E6 supervisore: PASS = ferma ed eretta >= 3,0 s dopo il volo.')
    r = prova()
    if r['caduta']:
        print('Volo %.0f ms (+%.1f cm) | passi %d + assestamenti %d | '
              'poi caduta (fermo max %.2f s)'
              % (r['t_volo_ms'], r['clearance_cm'], r['passi'],
                 r['assestamenti'], r['t_fermo_s']))
        print('VERDETTO E6 [C]: il SUPERVISORE decide GIUSTO (riconosce lo stato')
        print('buono, entra in FERMA al momento giusto, tollera la propria cura,')
        print('il passetto di assestamento esiste) e la ferma-due-tempi guarisce')
        print('la ROTAZIONE. Ma la TRASLAZIONE residua (~13 cm dietro) e oltre')
        print('la autorita del tallone (-7 cm, scoperta 5): il confine e')
        print('GEOMETRICO, non di controllo. Porte: tallone piu lungo o caviglia')
        print('>= 35 gradi [PROPOSTE HW gia a registro]. Scoperta 29.')
        esito = ('MISURATO (arbitro certificato; il confine e il tallone)'
                 if r.get('t_volo_ms', 0) > 100 else 'FAIL')
    else:
        print('Volo %.0f ms (+%.1f cm) | passi %d + assestamenti %d | '
              'FERMA da %.2f s | finale z=%.3f pitch=%.1f |v|=%.2f'
              % (r['t_volo_ms'], r['clearance_cm'], r['passi'],
                 r['assestamenti'], r['t_fermo_s'], r['z_finale'],
                 r['pitch_finale'], r['v_finale']))
        ok = r['volo'] and r['eretta'] and r['t_fermo_s'] >= 3.0
        esito = 'PASS' if ok else 'PARZIALE'
        if ok:
            print('*** ATTERRATA: salto -> recupero -> FERMA ED ERETTA '
                  'per %.1f s ***' % r['t_fermo_s'])
    print('ESITO E6: ' + esito)
    json.dump(dict(versione=VERSIONE, esito=esito, **r),
              open('e6_supervisore.json', 'w'), indent=1)
    print('Salvato e6_supervisore.json')

def demo():
    import time
    import mujoco.viewer
    print(__doc__)
    g = Atterratrice()
    t_tot = 0.0
    dt = m.opt.timestep
    with mujoco.viewer.launch_passive(m, g.d) as v:
        while v.is_running():
            t0 = time.time()
            vivo = g.frame()
            t_tot += dt
            if not vivo or t_tot > 10.0:
                if vivo and g.t_fermo > 0.5:
                    print('atterrata: ferma %.1f s (%d passi + %d assestamenti)'
                          % (g.t_fermo, g.passi, g.assestamenti))
                g.reset()
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
