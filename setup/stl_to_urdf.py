"""
STL → URDF 자동 생성 도구
- 기능1: STL 바운딩 박스 추출
- 기능2: 관절 좌표 리스트로 단순 URDF 생성 (box)
- 기능3: STL 참조 URDF 생성 (mesh)
- 기능4: 조립된 STL → 링크별 STL 자동 분리 (연결 컴포넌트 분석)
"""

import struct
import os
import math
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────
# 하이퍼파라미터 / 상수
# ─────────────────────────────────────────
JOINT_SPECS = {
    "joint1": {"axis": "0 0 1", "lower": -3.14, "upper": 3.14},   # 베이스 Z축
    "joint2": {"axis": "0 1 0", "lower": -1.2,  "upper": 1.2},    # 어깨 Y축
    "joint3": {"axis": "0 1 0", "lower": -1.5,  "upper": 1.5},    # 팔꿈치 Y축
}
# MG996R 기반 링크 질량 (kg): 서보 55g + 3D프린트 링크 기준 추정
LINK_MASSES    = [0.12, 0.08, 0.04]
LINK_WIDTH     = 0.04              # 링크 단면 크기 (m)
# MG996R 속도: 0.14초/60° at 6V = 7.5 rad/s → 부하 고려 6.0 rad/s
SERVO_VELOCITY = 6.0               # rad/s


# ─────────────────────────────────────────
# 기능 1: STL 바운딩 박스 추출
# ─────────────────────────────────────────
def parse_stl_vertices(stl_path: str) -> np.ndarray:
    """바이너리/ASCII STL 모두 지원. 꼭짓점 좌표 배열 반환."""
    with open(stl_path, "rb") as f:
        header = f.read(80)
        try:
            num_triangles = struct.unpack("<I", f.read(4))[0]
            # 바이너리: 각 삼각형 = 법선(12) + 꼭짓점3개(36) + attr(2) = 50 bytes
            data = f.read(num_triangles * 50)
            if len(data) == num_triangles * 50:
                verts = []
                for i in range(num_triangles):
                    offset = i * 50 + 12   # 법선 건너뜀
                    for j in range(3):
                        v = struct.unpack("<fff", data[offset + j*12: offset + j*12 + 12])
                        verts.append(v)
                return np.array(verts)
        except Exception:
            pass

    # ASCII STL
    verts = []
    with open(stl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("vertex"):
                coords = list(map(float, line.split()[1:4]))
                verts.append(coords)
    return np.array(verts)


def stl_bounding_box(stl_path: str) -> dict:
    """STL 파일에서 바운딩 박스 크기 추출."""
    verts = parse_stl_vertices(stl_path)
    if len(verts) == 0:
        raise ValueError(f"STL 파싱 실패: {stl_path}")
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    sizes = maxs - mins              # [dx, dy, dz] mm 단위
    sizes_m = sizes * 0.001         # mm → m
    longest = float(sizes_m.max())
    return {"size_mm": sizes.tolist(), "size_m": sizes_m.tolist(), "length_m": longest}


# ─────────────────────────────────────────
# 기능 4: 조립된 STL → 링크별 STL 자동 분리
# ─────────────────────────────────────────
def _parse_stl_triangles(stl_path: str) -> list:
    """바이너리 STL에서 삼각형 리스트 반환. 각 원소: (normal, v0, v1, v2)"""
    triangles = []
    with open(stl_path, "rb") as f:
        f.read(80)  # header
        n = struct.unpack("<I", f.read(4))[0]
        for _ in range(n):
            normal = struct.unpack("<fff", f.read(12))
            v0     = struct.unpack("<fff", f.read(12))
            v1     = struct.unpack("<fff", f.read(12))
            v2     = struct.unpack("<fff", f.read(12))
            f.read(2)  # attribute
            triangles.append((normal, v0, v1, v2))
    return triangles


def _write_stl_binary(triangles: list, path: str) -> None:
    """삼각형 리스트를 바이너리 STL로 저장."""
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(triangles)))
        for normal, v0, v1, v2 in triangles:
            f.write(struct.pack("<fff", *normal))
            f.write(struct.pack("<fff", *v0))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(b"\x00\x00")


def split_stl_components(stl_path: str, output_dir: str,
                         link_names: list = None) -> tuple:
    """
    조립된 STL을 연결 컴포넌트별로 분리해 링크별 STL로 저장.
    Z 중심 오름차순 정렬 → base_link(최하단) … end_effector(최상단) 순.
    각 메시는 하단-중심을 원점(0,0,0)으로 정렬하여 저장 → URDF 이중 오프셋 방지.

    반환: (저장된 STL 경로 리스트, 관절 위치 리스트 [m])
           관절 위치 = 각 컴포넌트의 조립 좌표계 상 하단-중심 위치 (m 단위)
    """
    if link_names is None:
        link_names = ["base_link.stl", "link1.stl", "link2.stl", "end_effector.stl"]

    print(f"[분리] STL 파싱 중: {stl_path}")
    triangles = _parse_stl_triangles(stl_path)
    print(f"  총 삼각형: {len(triangles):,}개")

    # 정점 → 삼각형 인덱스 매핑 (소수점 2자리 반올림으로 근접 정점 병합)
    vert_to_tris = defaultdict(list)
    for ti, (_, v0, v1, v2) in enumerate(triangles):
        for v in (v0, v1, v2):
            key = (round(v[0], 2), round(v[1], 2), round(v[2], 2))
            vert_to_tris[key].append(ti)

    # Union-Find
    parent = list(range(len(triangles)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for tris in vert_to_tris.values():
        root = find(tris[0])
        for t in tris[1:]:
            parent[find(t)] = root

    # 컴포넌트별 삼각형 집합
    comps = defaultdict(list)
    for ti, tri in enumerate(triangles):
        comps[find(ti)].append(tri)

    print(f"  발견된 컴포넌트: {len(comps)}개")

    # Z 중심 오름차순 정렬 (베이스가 가장 아래)
    def z_center(tris):
        return sum((t[1][2] + t[2][2] + t[3][2]) / 3 for t in tris) / len(tris)

    sorted_comps = sorted(comps.values(), key=z_center)

    # 저장 + 메시 원점 정렬
    os.makedirs(output_dir, exist_ok=True)
    saved           = []
    attach_pts_m    = []   # 조립 좌표계에서 각 컴포넌트 하단-중심 (m 단위)

    for i, tris in enumerate(sorted_comps[:len(link_names)]):
        # 모든 정점 수집
        all_verts = np.array([v for _, v0, v1, v2 in tris for v in [v0, v1, v2]],
                             dtype=np.float64)
        # 하단-중심 = (X 평균, Y 평균, Z 최솟값) → 관절 연결점
        attach = np.array([all_verts[:, 0].mean(),
                           all_verts[:, 1].mean(),
                           all_verts[:, 2].min()], dtype=np.float64)
        attach_pts_m.append((attach * 0.001).tolist())   # mm → m

        # 정점을 하단-중심 기준으로 평행이동 (원점 정렬)
        centered_tris = []
        for normal, v0, v1, v2 in tris:
            centered_tris.append((
                normal,
                tuple(np.array(v0) - attach),
                tuple(np.array(v1) - attach),
                tuple(np.array(v2) - attach),
            ))

        name = link_names[i] if i < len(link_names) else f"part{i}.stl"
        out  = os.path.join(output_dir, name)
        _write_stl_binary(centered_tris, out)
        bb   = stl_bounding_box(out)
        print(f"  [{i}] {name}: {len(tris):,}삼각형, "
              f"크기 {[round(s,1) for s in bb['size_mm']]} mm, "
              f"조립 하단 {[round(v*1000,1) for v in attach_pts_m[-1]]} mm")
        saved.append(out)

    return saved, attach_pts_m


# ─────────────────────────────────────────
# 기능 2: 관절 좌표 → 단순 URDF 생성
# ─────────────────────────────────────────
def _link_length(p1, p2) -> float:
    return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))


def _cylinder_inertia(m, r, h) -> tuple:
    ixx = m / 12.0 * (3 * r**2 + h**2)
    iyy = ixx
    izz = m / 2.0 * r**2
    return ixx, iyy, izz


def _box_inertia(m, sx, sy, sz) -> tuple:
    ixx = m / 12.0 * (sy**2 + sz**2)
    iyy = m / 12.0 * (sx**2 + sz**2)
    izz = m / 12.0 * (sx**2 + sy**2)
    return ixx, iyy, izz


# 링크별 시각 색상 (rgba) — 발표/보고서용
_LINK_COLORS = [
    [0.25, 0.25, 0.30, 1.0],   # base_link: 다크 그레이
    [0.18, 0.52, 0.80, 1.0],   # link1:     파랑
    [0.20, 0.70, 0.40, 1.0],   # link2:     초록
    [0.90, 0.40, 0.15, 1.0],   # end_effector: 주황
]
# 관절 구 반지름
JOINT_SPHERE_R = 0.018
LINK_RADIUS    = 0.012   # 실린더 반지름


def generate_simple_urdf(joint_positions: list, output_path: str) -> None:
    """
    joint_positions: [[x0,y0,z0], [x1,y1,z1], ...]  (월드 좌표, m 단위)
    링크: 실린더 + 관절 구 조합으로 깔끔한 로봇팔 외형 생성.
    """
    n_links = len(joint_positions) - 1
    link_names  = ["base_link"] + [f"link{i+1}" for i in range(n_links - 1)] + ["end_effector"]
    joint_names = list(JOINT_SPECS.keys())   # joint1, joint2, joint3

    lines = ['<?xml version="1.0"?>', '<robot name="chess_arm">']

    # ── 링크 생성 (실린더 시각) ──
    for i, lname in enumerate(link_names):
        length = _link_length(joint_positions[i], joint_positions[i + 1]) if i < n_links else 0.03
        mass   = LINK_MASSES[i] if i < len(LINK_MASSES) else 0.03
        r      = LINK_RADIUS
        rgba   = _LINK_COLORS[i] if i < len(_LINK_COLORS) else [0.6, 0.6, 0.6, 1.0]
        rgba_s = " ".join(f"{v:.2f}" for v in rgba)
        ixx, iyy, izz = _cylinder_inertia(mass, r, length)

        # 실린더: PyBullet은 Z축 방향 기본 → origin z=length/2 로 하단 정렬
        lines += [
            f'  <link name="{lname}">',
            "    <visual>",
            "      <geometry>",
            f'        <cylinder radius="{r:.4f}" length="{length:.4f}"/>',
            "      </geometry>",
            f'      <origin xyz="0 0 {length/2:.4f}" rpy="0 0 0"/>',
            f'      <material name="color_{lname}">',
            f'        <color rgba="{rgba_s}"/>',
            "      </material>",
            "    </visual>",
            "    <collision>",
            "      <geometry>",
            f'        <cylinder radius="{r:.4f}" length="{length:.4f}"/>',
            "      </geometry>",
            f'      <origin xyz="0 0 {length/2:.4f}" rpy="0 0 0"/>',
            "    </collision>",
            "    <inertial>",
            f'      <mass value="{mass}"/>',
            f'      <origin xyz="0 0 {length/2:.4f}" rpy="0 0 0"/>',
            f'      <inertia ixx="{ixx:.6f}" ixy="0" ixz="0" iyy="{iyy:.6f}" iyz="0" izz="{izz:.6f}"/>',
            "    </inertial>",
            "  </link>",
        ]
        # 관절 위치 표시용 구 (더미 링크)
        sphere_name = f"{lname}_joint_sphere"
        s_mass = 0.001
        s_r    = JOINT_SPHERE_R
        lines += [
            f'  <link name="{sphere_name}">',
            "    <visual>",
            "      <geometry>",
            f'        <sphere radius="{s_r:.4f}"/>',
            "      </geometry>",
            '      <origin xyz="0 0 0" rpy="0 0 0"/>',
            '      <material name="joint_color">',
            '        <color rgba="0.95 0.80 0.10 1.0"/>',
            "      </material>",
            "    </visual>",
            "    <inertial>",
            f'      <mass value="{s_mass}"/>',
            '      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <inertia ixx="1e-7" ixy="0" ixz="0" iyy="1e-7" iyz="0" izz="1e-7"/>',
            "    </inertial>",
            "  </link>",
            f'  <joint name="sphere_joint_{lname}" type="fixed">',
            f'    <parent link="{lname}"/>',
            f'    <child link="{sphere_name}"/>',
            '    <origin xyz="0 0 0" rpy="0 0 0"/>',
            "  </joint>",
        ]

    # ── 관절 생성 ──
    for i in range(n_links):
        parent = link_names[i]
        child  = link_names[i + 1]
        length = _link_length(joint_positions[i], joint_positions[i + 1])

        if i < len(joint_names):
            jname = joint_names[i]
            spec  = JOINT_SPECS[jname]
            jtype = "revolute"
            axis  = spec["axis"]
            lower = spec["lower"]
            upper = spec["upper"]
        else:
            jname = f"joint{i+1}"
            jtype = "fixed"
            axis  = "0 0 1"
            lower, upper = 0.0, 0.0

        limit_line = (
            f'      <limit lower="{lower}" upper="{upper}" effort="1.27" velocity="{SERVO_VELOCITY}"/>'
            if jtype == "revolute" else ""
        )

        lines += [
            f'  <joint name="{jname}" type="{jtype}">',
            f'    <parent link="{parent}"/>',
            f'    <child link="{child}"/>',
            f'    <origin xyz="0 0 {length:.4f}" rpy="0 0 0"/>',
            f'    <axis xyz="{axis}"/>',
        ]
        if limit_line:
            lines.append(limit_line)
        lines.append("  </joint>")

    lines.append("</robot>")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[URDF] 단순 URDF 생성 완료: {output_path}")


# ─────────────────────────────────────────
# 기능 3: STL 참조 URDF 생성 (2단계용)
# ─────────────────────────────────────────
def generate_full_urdf(stl_paths: list, joint_positions: list, output_path: str,
                       mesh_centered: bool = False) -> None:
    """
    stl_paths      : STL 파일 경로 리스트 (링크 수와 동일)
    joint_positions: 관절 좌표 리스트 (절대 월드 좌표, m)
    mesh_centered  : True면 각 메시가 이미 원점 정렬되어 있음
                     → visual origin xyz="0 0 0" 사용 (이중 오프셋 방지)
    """
    n_links = len(joint_positions) - 1
    link_names  = ["base_link"] + [f"link{i+1}" for i in range(n_links - 1)] + ["end_effector"]
    joint_names = list(JOINT_SPECS.keys())

    lines = ['<?xml version="1.0"?>', '<robot name="chess_arm_full">']

    for i, lname in enumerate(link_names):
        if i < n_links:
            length = _link_length(joint_positions[i], joint_positions[i + 1])
        else:
            length = 0.05
        mass = LINK_MASSES[i] if i < len(LINK_MASSES) else 0.05
        w    = LINK_WIDTH
        ixx, iyy, izz = _box_inertia(mass, w, w, length)

        stl_file = stl_paths[i] if i < len(stl_paths) else ""
        has_mesh = bool(stl_file)
        geom_line = (
            f'        <mesh filename="{stl_file}" scale="0.001 0.001 0.001"/>'
            if has_mesh else f'        <box size="{w:.4f} {w:.4f} {length:.4f}"/>'
        )
        # 메시가 원점 정렬된 경우 추가 오프셋 불필요; 박스는 중심 오프셋 필요
        vis_origin = "0 0 0" if (has_mesh and mesh_centered) else f"0 0 {length/2:.4f}"

        lines += [
            f'  <link name="{lname}">',
            "    <visual>",
            "      <geometry>",
            geom_line,
            "      </geometry>",
            f'      <origin xyz="{vis_origin}" rpy="0 0 0"/>',
            "    </visual>",
            "    <collision>",
            "      <geometry>",
            f'        <box size="{w:.4f} {w:.4f} {length:.4f}"/>',
            "      </geometry>",
            f'      <origin xyz="0 0 {length/2:.4f}" rpy="0 0 0"/>',
            "    </collision>",
            "    <inertial>",
            f'      <mass value="{mass}"/>',
            f'      <origin xyz="0 0 {length/2:.4f}" rpy="0 0 0"/>',
            f'      <inertia ixx="{ixx:.6f}" ixy="0" ixz="0" iyy="{iyy:.6f}" iyz="0" izz="{izz:.6f}"/>',
            "    </inertial>",
            "  </link>",
        ]

    for i in range(n_links):
        parent = link_names[i]
        child  = link_names[i + 1]
        length = _link_length(joint_positions[i], joint_positions[i + 1])

        if i < len(joint_names):
            jname = joint_names[i]
            spec  = JOINT_SPECS[jname]
            jtype = "revolute"
            axis  = spec["axis"]
            lower = spec["lower"]
            upper = spec["upper"]
        else:
            jname = f"joint{i+1}"
            jtype = "fixed"
            axis  = "0 0 1"
            lower, upper = 0.0, 0.0

        lines += [
            f'  <joint name="{jname}" type="{jtype}">',
            f'    <parent link="{parent}"/>',
            f'    <child link="{child}"/>',
            f'    <origin xyz="0 0 {length:.4f}" rpy="0 0 0"/>',
            f'    <axis xyz="{axis}"/>',
        ]
        if jtype == "revolute":
            lines.append(f'    <limit lower="{lower}" upper="{upper}" effort="1.27" velocity="{SERVO_VELOCITY}"/>')
        lines.append("  </joint>")

    lines.append("</robot>")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[URDF] 풀 URDF 생성 완료: {output_path}")


# ─────────────────────────────────────────
# 단독 실행 예시
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Fusion 360에서 읽은 관절 좌표 직접 입력
    # joint1을 원점으로 잡고 L1, L2, L3만 사용
    joint_positions = [
        [0.0, 0.0, 0.000],          # joint1 (베이스 서보 축 = 원점)
        [0.0, 0.0, 0.140],          # joint2 (어깨) +140mm
        [0.0, 0.0, 0.295],          # joint3 (팔꿈치) +155mm
        [0.0, 0.0, 0.370],          # 끝단(흡착기) +75mm
    ]

    generate_simple_urdf(joint_positions, "setup/urdf/robot_simple.urdf")
    print("robot_simple.urdf 생성 성공")

    MESH_DIR       = os.path.join(os.path.dirname(__file__), "meshes")
    ASSEMBLED_STL  = os.path.join(MESH_DIR, "robot arm-1 assembly.stl")
    MESH_NAMES     = ["base_link.stl", "link1.stl", "link2.stl", "end_effector.stl"]
    MESH_PATHS     = [os.path.join(MESH_DIR, n) for n in MESH_NAMES]

    mesh_centered = False
    if os.path.exists(ASSEMBLED_STL):
        # 조립된 STL → 링크별 자동 분리 (메시 원점 정렬 포함)
        print("assembled.stl 발견 → 링크별 자동 분리 시작")
        stl_paths, attach_pts_m = split_stl_components(ASSEMBLED_STL, MESH_DIR)
        # STL에서 추출한 관절 위치로 교체 (하드코딩 값보다 우선)
        if len(attach_pts_m) >= 4:
            # 각 파트의 조립 좌표계 상 하단 위치를 Z 기준 관절 체인으로 재구성
            # attach_pts_m[0] = base 하단 → 월드 원점으로 평행이동
            base_z = attach_pts_m[0][2]
            joint_positions = [[p[0], p[1], p[2] - base_z] for p in attach_pts_m]
            joint_positions.append([joint_positions[-1][0],
                                     joint_positions[-1][1],
                                     joint_positions[-1][2] + 0.05])  # 끝단 +5cm
            print(f"  STL 기반 관절 위치 (m): {[[round(v,3) for v in p] for p in joint_positions]}")
        mesh_centered = True
    else:
        stl_paths = [p for p in MESH_PATHS if os.path.exists(p)]

    if stl_paths:
        print(f"\nSTL {len(stl_paths)}개로 메시 URDF 생성")
    else:
        print("\nsetup/meshes/ 에 STL 없음 → 박스 지오메트리로 생성")
        print("  실제 형상을 보려면 아래 중 하나:")
        print("    1) setup/meshes/robot arm-1 assembly.stl  ← 조립된 STL 하나 (자동 분리)")
        print("    2) setup/meshes/base_link.stl, link1.stl, link2.stl, end_effector.stl")

    generate_full_urdf(stl_paths, joint_positions, "setup/urdf/robot_full.urdf",
                       mesh_centered=mesh_centered)
    print("robot_full.urdf 생성 성공")
