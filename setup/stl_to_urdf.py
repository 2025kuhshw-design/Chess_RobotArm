"""
STL → URDF 자동 생성 도구
- 기능1: STL 바운딩 박스 추출
- 기능2: 관절 좌표 리스트로 단순 URDF 생성 (box)
- 기능3: STL 참조 URDF 생성 (mesh)
"""

import struct
import os
import math
import numpy as np

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
# 기능 2: 관절 좌표 → 단순 URDF 생성
# ─────────────────────────────────────────
def _link_length(p1, p2) -> float:
    return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))


def _box_inertia(m, sx, sy, sz) -> tuple:
    ixx = m / 12.0 * (sy**2 + sz**2)
    iyy = m / 12.0 * (sx**2 + sz**2)
    izz = m / 12.0 * (sx**2 + sy**2)
    return ixx, iyy, izz


def generate_simple_urdf(joint_positions: list, output_path: str) -> None:
    """
    joint_positions: [[x0,y0,z0], [x1,y1,z1], ...]  (월드 좌표, m 단위)
    인접 관절 간 거리를 링크 길이로 사용해 box 링크 URDF 생성.
    """
    n_links = len(joint_positions) - 1
    link_names  = ["base_link"] + [f"link{i+1}" for i in range(n_links - 1)] + ["end_effector"]
    joint_names = list(JOINT_SPECS.keys())   # joint1, joint2, joint3

    lines = ['<?xml version="1.0"?>', '<robot name="chess_arm">']

    # ── 링크 생성 ──
    for i, lname in enumerate(link_names):
        if i < n_links:
            length = _link_length(joint_positions[i], joint_positions[i + 1])
        else:
            length = 0.05
        mass = LINK_MASSES[i] if i < len(LINK_MASSES) else 0.05
        w    = LINK_WIDTH
        ixx, iyy, izz = _box_inertia(mass, w, w, length)

        lines += [
            f'  <link name="{lname}">',
            "    <visual>",
            "      <geometry>",
            f'        <box size="{w:.4f} {w:.4f} {length:.4f}"/>',
            "      </geometry>",
            f'      <origin xyz="0 0 {length/2:.4f}" rpy="0 0 0"/>',
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
def generate_full_urdf(stl_paths: list, joint_positions: list, output_path: str) -> None:
    """
    stl_paths: STL 파일 경로 리스트 (링크 수와 동일)
    joint_positions: 관절 좌표 리스트
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
        mesh_line = (
            f'        <mesh filename="{stl_file}" scale="0.001 0.001 0.001"/>'
            if stl_file else f'        <box size="{w:.4f} {w:.4f} {length:.4f}"/>'
        )

        lines += [
            f'  <link name="{lname}">',
            "    <visual>",
            "      <geometry>",
            mesh_line,
            "      </geometry>",
            f'      <origin xyz="0 0 {length/2:.4f}" rpy="0 0 0"/>',
            "    </visual>",
            "    <collision>",
            "      <geometry>",
            mesh_line,
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

    # setup/meshes/ 에 STL 파일이 있으면 메시 URDF, 없으면 박스 URDF 생성
    # STL 파일 순서: base_link.stl, link1.stl, link2.stl, end_effector.stl
    MESH_DIR = os.path.join(os.path.dirname(__file__), "meshes")
    MESH_NAMES = ["base_link.stl", "link1.stl", "link2.stl", "end_effector.stl"]
    stl_paths = []
    for name in MESH_NAMES:
        path = os.path.join(MESH_DIR, name)
        if os.path.exists(path):
            stl_paths.append(path)

    if stl_paths:
        print(f"STL 파일 {len(stl_paths)}개 발견 → 메시 URDF 생성")
    else:
        print("setup/meshes/ 에 STL 없음 → 박스 지오메트리로 생성")
        print("  STL 파일을 아래 이름으로 넣으면 실제 형상이 표시됩니다:")
        for name in MESH_NAMES:
            print(f"    setup/meshes/{name}")

    generate_full_urdf(stl_paths, joint_positions, "setup/urdf/robot_full.urdf")
    print("robot_full.urdf 생성 성공")
