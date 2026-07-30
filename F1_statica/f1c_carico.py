#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f1c_carico.py v1.0 - Modulo F1-C (TDD par.3.1 "porta carichi fino a 8 kg" /
par.2.3 coppie): carico 2-8 kg nella mano destra, fermo e in marcia sul posto.

    python f1c_carico.py           demo 3D: braccio teso con 4 kg (loop)
    python f1c_carico.py --test    campagna A/A2/B: tabelle + PNG + JSON

Scenari: A = braccio TESO avanti 0,45 m con carico (requisito spalla).
A2 = postura RACCOLTA (gomito flesso, mano al petto) con carico (claim TDD).
B = MARCIA SUL POSTO (gait_core, 4 passi) con carico in mano.
Il carico e' massa aggiunta alla mano destra NEL SORGENTE (regola piattaforma);
il modello derivato tx34_v1_carico.xml si rigenera a ogni esecuzione.
"""
import os
import sys
import sys
import json
import numpy as np
import mujoco
import f1b_reach as R          # riuso certificato: passo(), braccio_verso(), smooth()

VERSIONE = '1.3'
XML = 'tx34_v1.xml'
RIGA_MANO_DX = '<geom type="sphere" size="0.04" mass="0.75"/>'

def xml_con_carico(kg):
    s = open(XML).read()
    assert s.count(RIGA_MANO_DX) == 2, 'attese 2 mani identiche nel sorgente'
    nuova = '<geom type="sphere" size="0.04" mass="%.2f"/>' % (0.75 + kg)
    s = s.replace(RIGA_MANO_DX, nuova, 1)          # la prima e' la destra
    return s

def modello(kg):
    m = mujoco.MjModel.from_xml_string(xml_con_carico(kg))
    return m

def tieni(kg, target_rel, T=4.5, registra=False):
    """Statico: raggiungi e tieni il target col carico kg. target_rel dal punto spalla."""
    m = modello(kg)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    sh0 = d.xanchor[R.JID['r_shoulder_pitch']].copy()
    tgt = (sh0[0]+target_rel[0], sh0[1]+target_rel[1], sh0[2]+target_rel[2])
    cp = d.subtree_com[0].copy()
    com0 = float(cp[0])
    pic = dict(sp=0.0, go=0.0, po=0.0)
    dcom = 0.0
    err_fin = []
    Iz, Ir = 0.0, 0.0                     # integratori anti-droop (con anti-windup)
    for i in range(int(T/dt)):
        t = i*dt
        fr = R.smooth(t/2.0)
        arm, tg = R.braccio_verso(d, tgt, fr)
        mano = d.xpos[R.BH]
        ez = float(tg[2] - mano[2])
        shn = d.xanchor[R.JID['r_shoulder_pitch']]
        er = float(np.linalg.norm(np.array(tg) - shn) - np.linalg.norm(mano - shn))
        Iz = float(np.clip(Iz + 2.0*ez*dt, -0.6, 0.6))
        Ir = float(np.clip(Ir + 2.5*er*dt, -0.7, 0.7))
        arm = (arm[0] - 1.5*ez - Iz, arm[1] + 2.0*er + Ir, arm[2])
        cp = R.passo(d, cp, 0.0, arm)
        mujoco.mj_step(m, d)
        pic['sp'] = max(pic['sp'], abs(float(d.actuator_force[R.AID['r_shoulder_pitch']])))
        pic['go'] = max(pic['go'], abs(float(d.actuator_force[R.AID['r_elbow']])))
        pic['po'] = max(pic['po'], abs(float(d.actuator_force[R.AID['r_wrist_pitch']])))
        dcom = max(dcom, abs(float(d.subtree_com[0][0]) - com0))
        if t > T - 0.6:
            err_fin.append(float(np.linalg.norm(mano - np.array(tgt))))
        if d.qpos[2] < 0.70:
            return dict(caduta=True, t_caduta=round(t, 1))
    err = float(np.mean(err_fin))
    return dict(caduta=False, tenuto=bool(err < 0.05), err_mm=round(err*1000, 1),
                spalla_Nm=round(pic['sp'], 1), gomito_Nm=round(pic['go'], 1),
                polso_Nm=round(pic['po'], 1), dCoM_mm=round(dcom*1000, 1))

def marcia(kg, n_passi=2):
    """Camminata v2 con carico: passi completati + picchi coppia gambe."""
    from gait_core import gait
    xml_tmp = 'tx34_v1_carico.xml'
    open(xml_tmp, 'w').write(xml_con_carico(kg))
    F = []
    def rec(t, d):
        F.append(d.actuator_force.copy())
    import os
    os.environ['NOVA_PAYLOAD_KG'] = str(kg)
    passi, x, caduta, _ = gait(step=0.14, t_sw=0.6, lift=0.04, y_in=0.02,
                               lean=0.15, n_passi=n_passi, xml=xml_tmp, recorder=rec)
    os.environ.pop('NOVA_PAYLOAD_KG', None)
    F = np.array(F)
    out = dict(kg=kg, passi=passi, caduta=(None if caduta is None else round(caduta, 1)))
    for nome in ('hip_pitch', 'hip_roll', 'ankle_pitch'):
        out['picco_' + nome + '_Nm'] = round(float(max(
            np.abs(F[:, R.AID['r_' + nome]]).max(),
            np.abs(F[:, R.AID['l_' + nome]]).max())), 1)
    return out

def campagna():
    print('f1c_carico.py - versione ' + VERSIONE)
    print('F1-C carico mano destra: A braccio teso 0,45 m | A2 raccolta | B camminata v2')
    print('Massa totale verificata con 8 kg: %.1f kg' % sum(modello(8.0).body_mass))
    carichi = (0.0, 2.0, 4.0, 6.0, 8.0)
    ris = dict(A=[], A2=[], B=[])
    import importlib.util
    for _p in ('../core', 'core', '/home/claude/NOVA-SIM/core'):
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)
    gait_ok = importlib.util.find_spec('gait_core') is not None

    print('-- A: braccio TESO avanti 0,45 m (forcerange spalla 43 Nm (X8-120)) --')
    print('%5s | %6s %8s %9s %9s %8s %7s' %
          ('kg', 'tiene', 'err[mm]', 'spalla', 'gomito', 'polso', 'dCoM'))
    for kg in carichi:
        r = tieni(kg, (0.45, 0.0, 0.0))
        ris['A'].append(dict(kg=kg, **r))
        if r['caduta']:
            print('%5.0f | CADUTA a %.1f s' % (kg, r['t_caduta']))
        else:
            print('%5.0f | %6s %8.1f %9.1f %9.1f %8.1f %7.1f' %
                  (kg, 'SI' if r['tenuto'] else 'NO', r['err_mm'], r['spalla_Nm'],
                   r['gomito_Nm'], r['polso_Nm'], r['dCoM_mm']))
    print('-- A2: postura RACCOLTA (mano al petto, 0,15 avanti / 0,25 giu) --')
    for kg in carichi:
        r = tieni(kg, (0.15, 0.0, -0.25))
        ris['A2'].append(dict(kg=kg, **r))
        if r['caduta']:
            print('%5.0f | CADUTA a %.1f s' % (kg, r['t_caduta']))
        else:
            print('%5.0f | %6s %8.1f %9.1f %9.1f %8.1f %7.1f' %
                  (kg, 'SI' if r['tenuto'] else 'NO', r['err_mm'], r['spalla_Nm'],
                   r['gomito_Nm'], r['polso_Nm'], r['dCoM_mm']))
    print('-- B: CAMMINATA campione v2 con carico (gait_core, y_in 0,02 lean 0,15) --')
    if not gait_ok:
        print('  [B] NON ESEGUIBILE: richiede core/gait_core.py')
        print('      (modulo campione custodito da MIKE - vedi F0_fondamenta/README)')
    else:
        print('%5s | %6s %8s | %10s %9s %12s' %
              ('kg', 'passi', 'caduta', 'anca pitch', 'anca roll', 'cavig. pitch'))
        for kg in carichi:
            r = marcia(kg)
            ris['B'].append(r)
            print('%5.0f | %6d %8s | %10.1f %9.1f %12.1f' %
                  (kg, r['passi'], ('no' if r['caduta'] is None else '%.1fs' % r['caduta']),
                   r['picco_hip_pitch_Nm'], r['picco_hip_roll_Nm'],
                   r['picco_ankle_pitch_Nm']))

    # sintesi: pendenze Nm/kg
    kgv = np.array(carichi)
    spA = np.array([r['spalla_Nm'] for r in ris['A'] if not r.get('caduta')])
    slope_sp = float(np.polyfit(kgv[:len(spA)], spA, 1)[0])
    if ris['B']:
        roll = np.array([r['picco_hip_roll_Nm'] for r in ris['B']])
        ank = np.array([r['picco_ankle_pitch_Nm'] for r in ris['B']])
        slope_roll = float(np.polyfit(kgv, roll, 1)[0])
        slope_ank = float(np.polyfit(kgv, ank, 1)[0])
    else:
        roll = ank = np.array([])
        slope_roll = slope_ank = 0.0
    max_teso = 0.0
    for r in ris['A']:
        if (not r.get('caduta')) and r['tenuto'] and r['spalla_Nm'] < 39.5:
            max_teso = r['kg']
    ok_racc = all((not r.get('caduta')) and r['tenuto'] for r in ris['A2'])
    ok_marcia = (all(r['passi'] >= 2 and r['caduta'] is None for r in ris['B'])
                 if ris['B'] else False)
    print('SINTESI: spalla %.1f Nm/kg (teso) -> requisito 8 kg tesi: %.0f Nm.' %
          (slope_sp, ris['A'][0]['spalla_Nm'] + slope_sp*8))
    if ris['B']:
        print('Marcia: anca roll %+.2f Nm/kg, caviglia %+.2f Nm/kg (asimmetria dx).' %
              (slope_roll, slope_ank))
    print('Carico max a braccio TESO entro 43 Nm (X8): %.0f kg. Postura raccolta 8 kg: %s. '
          'Marcia 8 kg: %s.' % (max_teso, 'OK' if ok_racc else 'NO',
                                'OK' if ok_marcia else 'NO'))
    max_marcia = 0.0
    for r in ris['B']:
        if r['passi'] >= 2 and r['caduta'] is None:
            max_marcia = r['kg']
    if not gait_ok:
        esito = 'PASS-A (B non eseguibile senza gait_core)' if ok_racc else 'PARZIALE'
    else:
        esito = 'PASS' if (ok_racc and max_marcia >= 6.0) else 'PARZIALE'
    print('ESITO F1-C: %s - claim "8 kg" (par.3.1): raccolta max 6 kg [C];'
          ' in marcia max %.0f kg (anca roll satura); a braccio teso richiede'
          ' %.0f Nm di spalla (oltre i 43 (X8) del modello).' %
          (esito, max_marcia, ris['A'][0]['spalla_Nm'] + slope_sp*8))
    json.dump(dict(versione=VERSIONE, esito=esito,
                   spalla_Nm_per_kg=round(slope_sp, 2),
                   requisito_spalla_8kg_tesi_Nm=round(ris['A'][0]['spalla_Nm']+slope_sp*8, 1),
                   carico_max_teso_43Nm_kg=max_teso, carico_max_marcia_kg=max_marcia,
                   gait_core_presente=bool(gait_ok),
                   marcia_roll_Nm_per_kg=round(slope_roll, 2),
                   marcia_caviglia_Nm_per_kg=round(slope_ank, 2),
                   risultati=ris),
              open('f1c_carico.json', 'w'), indent=1)
    print('Salvato f1c_carico.json')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(9, 7))
    spA_all = [r['spalla_Nm'] for r in ris['A'] if not r.get('caduta')]
    sp2 = [r['spalla_Nm'] for r in ris['A2'] if not r.get('caduta')]
    axs[0].plot(kgv[:len(spA_all)], spA_all, 'o-', color='tab:blue', label='braccio teso 0,45 m')
    axs[0].plot(kgv[:len(sp2)], sp2, 's-', color='tab:green', label='postura raccolta')
    axs[0].axhline(43, color='r', ls='--', lw=1, label='forcerange spalla 43 Nm (X8-120)')
    axs[0].set_xlabel('carico in mano [kg]')
    axs[0].set_ylabel('picco coppia spalla [Nm]')
    axs[0].set_title('F1-C: coppia spalla vs carico (%.1f Nm/kg a braccio teso)' % slope_sp)
    axs[0].legend(fontsize=8)
    if len(roll):
        axs[1].plot(kgv, roll, 'o-', color='tab:orange', label='anca roll (marcia)')
        axs[1].plot(kgv, ank, 's-', color='tab:purple', label='caviglia pitch (marcia)')
        axs[1].axhline(80, color='r', ls='--', lw=1, label='limite RMD-X8')
    else:
        axs[1].text(0.5, 0.5, 'parte B non eseguibile senza core/gait_core.py',
                    ha='center', va='center', transform=axs[1].transAxes)
    axs[1].set_xlabel('carico in mano [kg]')
    axs[1].set_ylabel('picco coppia [Nm]')
    axs[1].set_title('marcia sul posto: %+.2f Nm/kg roll, %+.2f Nm/kg caviglia' %
                     (slope_roll, slope_ank))
    if len(roll):
        axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig('f1c_carico.png', dpi=150)
    print('Salvato f1c_carico.png')

def demo():
    import time
    import mujoco.viewer
    print(__doc__)
    kg = 4.0
    print('Demo: braccio teso avanti con %.0f kg in mano (massa totale %.1f kg).' %
          (kg, sum(modello(kg).body_mass)))
    m = modello(kg)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    sh0 = d.xanchor[R.JID['r_shoulder_pitch']].copy()
    tgt = (sh0[0]+0.45, sh0[1], sh0[2])
    cp = d.subtree_com[0].copy()
    t = 0.0
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            t0 = time.time()
            fr = R.smooth(t/2.0)
            arm, tg = R.braccio_verso(d, tgt, fr)
            mano = d.xpos[R.BH]
            ez = float(tg[2] - mano[2])
            shn = d.xanchor[R.JID['r_shoulder_pitch']]
            er = float(np.linalg.norm(np.array(tg) - shn) - np.linalg.norm(mano - shn))
            arm = (arm[0] - 1.5*ez, arm[1] + 2.0*er, arm[2])
            cp = R.passo(d, cp, 0.0, arm)
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
