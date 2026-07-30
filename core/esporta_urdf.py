#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
esporta_urdf.py v1.0 - Export URDF (F0): tx34_v1.xml -> tx34_v1.urdf per ROS2.
Masse e inerzie dal modello compilato · range par.3.2 · effort = forcerange
par.2.3 · capsule approssimate a cilindri (visual/collision) [A] · corpi
multi-giunto scomposti in catene con link fittizi (massa 1e-3 kg [A], standard
URDF: un giunto per link). Base = pelvis (floating gestito dal consumatore ROS).
v1.1: smorzamento dei giunti esportato dal modello (<dynamics damping>) [C] e
materiale per RViz (senza, i visual rendono col rosso di default).

    python esporta_urdf.py            genera tx34_v1.urdf + verifica
    python esporta_urdf.py --test     idem (alias)

Verifica: l'URDF viene RICARICATO in MuJoCo e confrontato con l'originale
(massa totale, numero di giunti, range a campione).
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
import numpy as np
import mujoco

VERSIONE = '1.3'
import sys as _sys
XML = _sys.argv[1] if len(_sys.argv) > 1 and _sys.argv[1].endswith('.xml') else 'tx34_v1.xml'
OUT = XML.replace('.xml', '.urdf')

def quat_rpy(q):
    w, x, y, z = q
    r = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    p = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yv = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return r, p, yv

def genera():
    m = mujoco.MjModel.from_xml_path(XML)
    total_mass = m.body_mass.sum()
    print(f"[MASSA MODELLO] {total_mass:.3f} kg")
    assert 65.5 < total_mass < 67.0, f"MASSA FUORI RANGE: {total_mass:.3f} kg — verifica XML"
    nome_b = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    nome_j = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    effort = {}
    for a in range(m.nu):
        effort[nome_j[m.actuator_trnid[a][0]]] = float(abs(m.actuator_forcerange[a][1]))
    smorz = {nome_j[j]: float(m.dof_damping[m.jnt_dofadr[j]])
             for j in range(m.njnt)
             if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}

    righe = ['<?xml version="1.0"?>',
             '<robot name="TX34_NOVA">',
             '  <!-- generato da esporta_urdf.py v%s da %s -->' % (VERSIONE, XML),
             '  <material name="tx34_grigio"><color rgba="0.72 0.72 0.78 1.0"/></material>',
             '  <link name="base"/>',
             '  <joint name="base_pelvis" type="floating">',
             '    <origin xyz="0 0 0" rpy="0 0 0"/>',
             '    <parent link="base"/><child link="pelvis"/>',
             '  </joint>']

    def link(nome, massa, ipos, iquat, inerzia, geoms):
        righe.append('  <link name="%s">' % nome)
        r, p, y = quat_rpy(iquat)
        righe.append('    <inertial>')
        righe.append('      <origin xyz="%.5f %.5f %.5f" rpy="%.5f %.5f %.5f"/>'
                     % (ipos[0], ipos[1], ipos[2], r, p, y))
        righe.append('      <mass value="%.4f"/>' % massa)
        righe.append('      <inertia ixx="%.6f" iyy="%.6f" izz="%.6f" '
                     'ixy="0" ixz="0" iyz="0"/>' % tuple(inerzia))
        righe.append('    </inertial>')
        for tipo, pos, quat, size in geoms:
            r2, p2, y2 = quat_rpy(quat)
            org = '<origin xyz="%.5f %.5f %.5f" rpy="%.5f %.5f %.5f"/>' % (
                pos[0], pos[1], pos[2], r2, p2, y2)
            if tipo == mujoco.mjtGeom.mjGEOM_SPHERE:
                g = '<sphere radius="%.4f"/>' % size[0]
            elif tipo == mujoco.mjtGeom.mjGEOM_BOX:
                g = '<box size="%.4f %.4f %.4f"/>' % (2*size[0], 2*size[1], 2*size[2])
            else:  # capsule/cilindro
                g = '<cylinder radius="%.4f" length="%.4f"/>' % (size[0], 2*size[1])
            for sez in ('visual', 'collision'):
                mat = ('<material name="tx34_grigio"/>' if sez == 'visual' else '')
                righe.append('    <%s>%s<geometry>%s</geometry>%s</%s>'
                             % (sez, org, g, mat, sez))
        righe.append('  </link>')

    def link_fittizio(nome):
        righe.append('  <link name="%s">' % nome)
        righe.append('    <inertial><mass value="0.001"/>'
                     '<inertia ixx="1e-6" iyy="1e-6" izz="1e-6" '
                     'ixy="0" ixz="0" iyz="0"/></inertial>')
        righe.append('  </link>')

    def giunto(nome, tipo, padre, figlio, xyz, axis, rng, eff, dmp=0.0):
        righe.append('  <joint name="%s" type="%s">' % (nome, tipo))
        righe.append('    <origin xyz="%.5f %.5f %.5f" rpy="0 0 0"/>' % tuple(xyz))
        righe.append('    <parent link="%s"/><child link="%s"/>' % (padre, figlio))
        righe.append('    <axis xyz="%g %g %g"/>' % tuple(axis))
        righe.append('    <limit lower="%.4f" upper="%.4f" effort="%.1f" '
                     'velocity="10.0"/>' % (rng[0], rng[1], eff))
        righe.append('    <dynamics damping="%.3f" friction="0.0"/>' % dmp)
        righe.append('  </joint>')

    for b in range(1, m.nbody):
        nb = nome_b[b]
        gs = [(m.geom_type[g], m.geom_pos[g], m.geom_quat[g], m.geom_size[g])
              for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        link(nb, float(m.body_mass[b]), m.body_ipos[b], m.body_iquat[b],
             m.body_inertia[b], gs)
        ja, jn = m.body_jntadr[b], m.body_jntnum[b]
        padre = nome_b[m.body_parentid[b]]
        if jn == 0:
            righe.append('  <joint name="%s_fix" type="fixed">' % nb)
            righe.append('    <origin xyz="%.5f %.5f %.5f" rpy="0 0 0"/>'
                         % tuple(m.body_pos[b]))
            righe.append('    <parent link="%s"/><child link="%s"/></joint>'
                         % (padre, nb))
            continue
        if m.jnt_type[ja] == mujoco.mjtJoint.mjJNT_FREE:
            continue  # pelvis: radice (base flottante a cura del consumatore)
        catena_padre = padre
        origine = m.body_pos[b]
        for k in range(jn):
            j = ja + k
            ultimo = (k == jn - 1)
            figlio = nb if ultimo else nb + '_int%d' % (k + 1)
            if not ultimo:
                link_fittizio(figlio)
            giunto(nome_j[j], 'revolute', catena_padre, figlio,
                   origine if k == 0 else (0, 0, 0),
                   m.jnt_axis[j], m.jnt_range[j], effort.get(nome_j[j], 0.0),
                   smorz.get(nome_j[j], 0.0))
            catena_padre = figlio
            origine = (0, 0, 0)
    righe.append('</robot>')
    open(OUT, 'w').write('\n'.join(righe))
    print('Scritto %s (%d righe)' % (OUT, len(righe)))
    return m

def verifica(m0):
    import xml.etree.ElementTree as ET
    print("--- VERIFICA 1: lettura dell'URDF stesso (parser XML stdlib) ---")
    rob = ET.parse(OUT).getroot()
    links = rob.findall('link')
    masse = [float(i.find('mass').get('value'))
             for L in links for i in L.findall('inertial')]
    fittizi = sum(1 for v in masse if v <= 0.001)
    massa_urdf = sum(masse)
    massa0 = float(sum(m0.body_mass))
    ok_m = abs(massa_urdf - (massa0 + 0.001*fittizi)) < 0.05
    print('massa nei link: %.2f kg = originale %.2f + %d link fittizi -> %s'
          % (massa_urdf, massa0, fittizi, 'OK' if ok_m else 'DIVERSA'))
    giunti = [j for j in rob.findall('joint') if j.get('type') == 'revolute']
    attesi_n = sum(1 for j in range(m0.njnt)
                   if m0.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)
    ok_j = len(giunti) == attesi_n
    print('giunti revolute: %d (attesi %d) -> %s' % (len(giunti), attesi_n, 'OK' if ok_j else 'NO'))
    campione = ('r_knee', 'l_hip_roll', 'waist_yaw', 'r_shoulder_pitch')
    eff0 = {}
    for a in range(m0.nu):
        jn = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, m0.actuator_trnid[a][0])
        eff0[jn] = float(abs(m0.actuator_forcerange[a][1]))
    attesi = {}
    for n_ in campione:
        j0 = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, n_)
        attesi[n_] = (float(m0.jnt_range[j0][0]), float(m0.jnt_range[j0][1]),
                      eff0[n_])   # attese DERIVATE dal modello, non a memoria
    ok_r = True
    for j in giunti:
        n = j.get('name')
        if n in attesi:
            lim = j.find('limit')
            lo, hi, ef = (float(lim.get('lower')), float(lim.get('upper')),
                          float(lim.get('effort')))
            a = attesi[n]
            va = abs(lo - a[0]) < 1e-3 and abs(hi - a[1]) < 1e-3 and abs(ef - a[2]) < 0.5
            ok_r = ok_r and va
            print('  %-17s range [%+.2f %+.2f] effort %5.1f Nm -> %s'
                  % (n, lo, hi, ef, 'OK' if va else 'NO'))
    print("--- VERIFICA 2: ricarico strutturale in MuJoCo (nota: l'import a base")
    print("    fissa SALDA base+pelvis nel mondo scartandone l'inerziale: quirk")
    print("    dell'importatore, non del file - vedi Verifica 1) ---")
    m2 = mujoco.MjModel.from_xml_path(OUT)
    total_mass = m2.body_mass.sum()
    print(f"[MASSA MODELLO] {total_mass:.3f} kg")
    assert 65.5 < total_mass < 67.0, f"MASSA FUORI RANGE: {total_mass:.3f} kg — verifica XML"
    cern2 = sum(1 for j in range(m2.njnt)
                if m2.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)
    print('giunti hinge ricaricati: %d -> %s' % (cern2, 'OK' if cern2 == attesi_n else 'NO'))
    ok_d = True
    for nome in ('r_knee', 'r_ankle_pitch'):
        j0 = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, nome)
        j2 = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_JOINT, nome)
        ok = np.allclose(m0.jnt_range[j0], m2.jnt_range[j2], atol=1e-3)
        d0 = float(m0.dof_damping[m0.jnt_dofadr[j0]])
        d2 = float(m2.dof_damping[m2.jnt_dofadr[j2]])
        okd = abs(d0 - d2) < 1e-3
        ok_d = ok_d and okd
        print('  range %-14s -> %s | damping %.2f vs %.2f -> %s'
              % (nome, 'OK' if ok else 'NO', d0, d2, 'OK' if okd else 'NO'))
    finale = ok_m and ok_j and ok_r and cern2 == attesi_n and ok_d
    print('ESITO EXPORT URDF: ' + ('VERIFICATO' if finale else 'DA CORREGGERE'))

if __name__ == '__main__':
    print('esporta_urdf.py - versione ' + VERSIONE)
    verifica(genera())
