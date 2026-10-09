"""OpenSeesPy modeling, analysis, recorder, and post-processing utilities.

The public compatibility surface is exported from :mod:`ufp`. Functions in
this module use the project's N-mm-s convention unless documented otherwise.
"""

import inspect
import math
import os
import time
from contextlib import redirect_stderr, redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops
import pandas as pd
from matplotlib.patches import Patch
from openseespy.opensees import *
from openseespy.opensees import section
from tqdm import tqdm

#==================================
# ========== Material and Section Tags ============
#==================================

def _resolve_tag_mapping(mapping, global_name):
    """Resolve an explicitly supplied tag mapping or a legacy module global."""
    if mapping is not None:
        return mapping
    resolved = globals().get(global_name)
    if resolved is None:
        raise RuntimeError(
            f"{global_name} is not defined in ufp.user_function; "
            f"pass the mapping explicitly."
        )
    return resolved


def print_all_material_tags(material_tags=None):
    """Print material tags from an explicit mapping or legacy module global."""
    material_tags = _resolve_tag_mapping(material_tags, "MaterialTags")
    print("=== 所有材料标签 ===")
    for key, (tag, desc) in material_tags.items():
        print(f"{key:12} = {tag:<2}  # {desc}")


def print_all_section_tags(section_tags=None):
    """Print section tags from an explicit mapping or legacy module global."""
    section_tags = _resolve_tag_mapping(section_tags, "SectionTags")
    print("=== 所有截面标签 ===")
    for key, (tag, desc) in section_tags.items():
        print(f"{key:20} = {tag:<2}  # {desc}")


def search_tags(keyword, material_tags=None, section_tags=None):
    """Search material and section tag mappings by key or description."""
    keyword = str(keyword)
    material_tags = _resolve_tag_mapping(material_tags, "MaterialTags")
    section_tags = _resolve_tag_mapping(section_tags, "SectionTags")

    print(f"=== 搜索包含 '{keyword}' 的标签 ===")
    found = False
    for key, (tag, desc) in material_tags.items():
        if keyword in key or keyword in desc:
            print(f"材料标签: {key:12} = {tag:<2}  # {desc}")
            found = True
    for key, (tag, desc) in section_tags.items():
        if keyword in key or keyword in desc:
            print(f"截面标签: {key:20} = {tag:<2}  # {desc}")
            found = True
    if not found:
        print("未找到匹配的材料或截面标签。")

def print_unit_table():
    """打印IK到SI单位换算表"""
    rows = [
        ("Length", "inch", "meter (m)", "1 in = 0.0254 m"),
        ("Force", "kip", "newton (N)", "1 kip = 4448.22 N"),
        ("Stress", "ksi", "MPa", "1 ksi = 6.895 MPa"),
        ("Time", "sec", "second (s)", "same"),
        ("Mass", "kip*sec^2/in", "kg", "about 14.594 kg"),
        ("Density", "kip*sec^2/in^4", "kg/m^3", "about 3.6e7 kg/m^3"),
        ("Unit weight", "kip/in^3", "N/m^3", "1 kip/in^3 = 2.527e7 N/m^3"),
        ("Acceleration", "in/sec^2", "m/s^2", "1 in/s^2 = 0.0254 m/s^2"),
        ("Stiffness", "kip/in", "N/m", "about 175126 N/m"),
        ("Moment", "kip*inch", "N*m", "1 kip*in = 113 N*m"),
        ("Energy", "kip*inch", "Joule (J)", "1 kip*in = 113 J"),
        ("Strain", "-", "-", "dimensionless"),
        ("Strain rate", "1/sec", "1/s", "same"),
    ]
    print("=== IKS (inch-kip-second) 到 SI 单位换算表 ===")
    for quantity, iks_unit, si_unit, relation in rows:
        print(f"{quantity:14} | {iks_unit:14} | {si_unit:12} | {relation}")


def GeneratePeaks(Dmax, DincrStatic=0.01, CycleType='Full', Fact=1.0):
    """
    生成位移峰值序列
    
    参数:
    Dmax: 最大位移
    DincrStatic: 位移增量
    CycleType: 循环类型 ('Full', 'HalfCycle', 或 'Push')
    Fact: 缩放因子
    
    返回: 位移步骤列表，可保存到 IDstep.txt
    """
    iDstep = [0.0]
    Disp = 0.0
    Dmax_scaled = Dmax * Fact

    if Dmax_scaled == 0:
        iDstep = [0.0]
    else:
        dx = DincrStatic if Dmax_scaled > 0 else -DincrStatic
        NstepsPeak = int(abs(Dmax_scaled) / DincrStatic)
        if NstepsPeak == 0: NstepsPeak = 1  # 确保至少一个步骤
        
        # 0 -> +peak
        for _ in range(NstepsPeak):
            Disp += dx
            iDstep.append(Disp)

        # +peak -> 0
        if CycleType in ['Full', 'HalfCycle']:
            for _ in range(NstepsPeak):
                Disp -= dx
                iDstep.append(Disp)

        # 0 -> -peak -> 0
        if CycleType == 'Full':
            # 0 -> -peak
            for _ in range(NstepsPeak):
                Disp -= dx
                iDstep.append(Disp)
            # -peak -> 0
            for _ in range(NstepsPeak):
                Disp += dx
                iDstep.append(Disp)

    return iDstep

def get_nodes_at_z(z_target, tol=1e-6):
    """获取在指定z坐标处的所有节点"""
    return [
        nd for nd in getNodeTags()
        if abs(nodeCoord(nd)[2] - z_target) < tol
    ]

def GetPeakStep(iDmax1, all_steps):
    """
    获取峰值步骤索引
    
    参数:
    iDmax1: 目标最大位移值列表
    all_steps: 所有步骤的列表，格式为 [(位移值, 其他信息), ...]
    
    返回: 峰值步骤索引列表，格式为 [(Dmax, step_index), ...]
    """
    peak_step_indices = []

    # 从all_steps中提取目标位移值
    disp_targets = [step[0] for step in all_steps]

    for Dmax in iDmax1:
        # 查找Dmax在目标位移列表中的第一次出现
        try:
            peak_index = disp_targets.index(Dmax)
        except ValueError:
            # 当精确匹配失败时，回退到最接近的位移值
            peak_index = min(
                range(len(disp_targets)),
                key=lambda i: abs(disp_targets[i] - Dmax)
            )

        peak_step_indices.append((Dmax, peak_index))

    print(f"总步数: {len(all_steps)}")
    print("\n峰值步骤:")
    for i, (dmax, step_id) in enumerate(peak_step_indices, 1):
        dmax_mm = dmax * 25.4
        print(f"第{i:2d}步，最大位移 {dmax_mm:.1f} mm，步号 {step_id:4d}")
    
    return peak_step_indices


# 静默分析函数，抑制错误输出
def silent_analyze(n=1):
    """Run ``ops.analyze`` while suppressing OpenSees stderr output."""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stderr(devnull):
            return ops.analyze(n)
    


def _coord_key(x, y, z, tol):
    return (
        int(round(float(x) / tol)),
        int(round(float(y) / tol)),
        int(round(float(z) / tol)),
    )


def build_node_bucket_map(node_tol=1e-6, node_tags=None):
    """
    从当前OpenSees模型构建可重用的节点注册表
    """
    bucket_map = {}
    tags = list(node_tags) if node_tags is not None else list(getNodeTags() or [])
    for tag in tags:
        x, y, z = nodeCoord(tag)
        key = _coord_key(x, y, z, node_tol)
        bucket_map.setdefault(key, []).append((tag, (x, y, z)))
    return bucket_map


def _find_node_in_buckets(x, y, z, bucket_map, node_tol):
    base_key = _coord_key(x, y, z, node_tol)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                key = (base_key[0] + dx, base_key[1] + dy, base_key[2] + dz)
                for tag, (xn, yn, zn) in bucket_map.get(key, []):
                    if (
                        abs(xn - x) <= node_tol
                        and abs(yn - y) <= node_tol
                        and abs(zn - z) <= node_tol
                    ):
                        return tag
    return None



def _resolve_point_from_node_or_coord(node_tag=None, point=None, label="point"):
    if node_tag is not None and point is not None:
        raise ValueError(f"{label} 不能同时提供 node_tag 和 point")
    if node_tag is None and point is None:
        raise ValueError(f"{label} 必须提供 node_tag 或 point")

    if node_tag is not None:
        x, y, z = nodeCoord(int(node_tag))
        return (float(x), float(y), float(z)), int(node_tag)

    if len(point) != 3:
        raise ValueError(f"{label} 必须是 3 个坐标值")

    x, y, z = point
    return (float(x), float(y), float(z)), None


def _unique_sorted_coords(values, tol):
    if not values:
        return []

    values = sorted(float(v) for v in values)
    merged = [values[0]]
    for value in values[1:]:
        if abs(value - merged[-1]) > tol:
            merged.append(value)
    return merged


def _resolve_base_geometry(base_i, base_j, node_tol):
    xi, yi, zi = base_i
    xj, yj, zj = base_j

    if abs(zi - zj) > node_tol:
        raise ValueError("底边两点必须在同一标高，当前函数只支持墙底边水平")

    dx = xj - xi
    dy = yj - yi

    if abs(dx) <= node_tol and abs(dy) <= node_tol:
        raise ValueError("底边两个点不能重合")

    if abs(dx) > node_tol and abs(dy) > node_tol:
        raise ValueError("当前函数只支持 XOZ 或 YOZ 平面内的剪力墙")

    if abs(dx) > node_tol:
        return {
            "plane": "XOZ",
            "span_dir": "+X" if dx > 0.0 else "-X",
            "wall_width": abs(dx),
            "anchor_point": base_i,
        }

    return {
        "plane": "YOZ",
        "span_dir": "+Y" if dy > 0.0 else "-Y",
        "wall_width": abs(dy),
        "anchor_point": base_i,
    }


FRAME_SECTION_REGISTRY = {}
FRAME_INTEGRATION_REGISTRY = {}
SHELL_SECTION_REGISTRY = {}


def register_frame_section(sec_tag, area, name=None, **properties):
    sec_tag = int(sec_tag)
    info = {
        "sec_tag": sec_tag,
        "area": float(area),
        "name": name,
    }
    info.update(properties)
    FRAME_SECTION_REGISTRY[sec_tag] = info
    return info


def register_frame_section_from_props(sec_tag, props, name=None, area_key="A", **properties):
    if props is None or area_key not in props:
        raise ValueError(f"props 中没有面积键 {area_key!r}")
    return register_frame_section(sec_tag, props[area_key], name=name, **properties)


def register_frame_integration(integration_tag, sec_tag, **properties):
    integration_tag = int(integration_tag)
    sec_tag = int(sec_tag)
    info = {
        "integration_tag": integration_tag,
        "sec_tag": sec_tag,
    }
    info.update(properties)
    FRAME_INTEGRATION_REGISTRY[integration_tag] = info
    return info


def define_frame_integration(integration_type, integration_tag, sec_tag, integration_points, *args):
    beamIntegration(integration_type, integration_tag, sec_tag, integration_points, *args)
    return register_frame_integration(
        integration_tag,
        sec_tag,
        integration_type=integration_type,
        integration_points=int(integration_points),
    )


def make_rebar_row_points(
    b,
    h,
    n,
    dia,
    row_type,
    cover=25,
    stirrup_dia=8,
    row_offset=0,
):
    """Return rebar point coordinates for one top or bottom row."""
    b = float(b)
    h = float(h)
    n = int(n)
    dia = float(dia)

    if n <= 0:
        return []

    x_left = cover + stirrup_dia + dia / 2.0
    x_right = b - cover - stirrup_dia - dia / 2.0

    if n == 1:
        xs = [b / 2.0]
    else:
        xs = [
            x_left + i * (x_right - x_left) / (n - 1)
            for i in range(n)
        ]

    if row_type == "top":
        y = h - cover - stirrup_dia - dia / 2.0 - row_offset
    elif row_type == "bottom":
        y = cover + stirrup_dia + dia / 2.0 + row_offset
    else:
        raise ValueError("row_type must be 'top' or 'bottom'")

    return [[x, y] for x in xs]


def make_rebar_side_points(
    b,
    h,
    n,
    dia,
    side,
    cover=25,
    stirrup_dia=8,
    end_offset=0,
):
    """Return rebar point coordinates for one left or right side row."""
    b = float(b)
    h = float(h)
    n = int(n)
    dia = float(dia)

    if n <= 0:
        return []

    x = cover + stirrup_dia + dia / 2.0
    if side == "right":
        x = b - x
    elif side != "left":
        raise ValueError("side must be 'left' or 'right'")

    y_bottom = cover + stirrup_dia + dia / 2.0 + end_offset
    y_top = h - cover - stirrup_dia - dia / 2.0 - end_offset

    if n == 1:
        ys = [h / 2.0]
    else:
        ys = [
            y_bottom + i * (y_top - y_bottom) / (n - 1)
            for i in range(n)
        ]

    return [[x, y] for y in ys]


def _resolve_frame_section_mat_tag(value, name, frame_level=2):
    if value is not None:
        return value

    import inspect

    frame = inspect.currentframe()
    for _ in range(frame_level):
        frame = frame.f_back
        if frame is None:
            break

    if frame is not None:
        if name in frame.f_locals:
            return frame.f_locals[name]
        if name in frame.f_globals:
            return frame.f_globals[name]

    raise ValueError(f"Please pass {name} or define {name} in the caller scope")


def _normalize_rebar_rows(rows):
    if rows is None:
        return []
    return list(rows)


def _normalize_side_bars(side_bars):
    if side_bars is None:
        return []
    if isinstance(side_bars, dict):
        return [side_bars]
    return list(side_bars)


def _normalize_perimeter_bars(perimeter_bars):
    if perimeter_bars is None:
        return []
    if isinstance(perimeter_bars, dict):
        return [perimeter_bars]
    if (
        isinstance(perimeter_bars, (list, tuple))
        and len(perimeter_bars) == 2
        and not isinstance(perimeter_bars[0], (list, tuple, dict))
    ):
        n, dia = perimeter_bars
        return [{"n": n, "dia": dia}]

    normalized = []
    for item in perimeter_bars:
        if isinstance(item, dict):
            normalized.append(item)
        else:
            n, dia = item
            normalized.append({"n": n, "dia": dia})
    return normalized


def build_rect_section(
    sec_tag,
    b,
    h,
    sec_name,
    top_bars=None,
    bottom_bars=None,
    side_bars=None,
    perimeter_bars=None,
    cover=25,
    stirrup_dia=8,
    row_spacing=35,
    top_row_spacing=None,
    bottom_row_spacing=None,
    side_end_clear=0,
    mesh_size_cover=50,
    mesh_size_core=50,
    GJ=1e13,
    cover_mat_tag=None,
    core_mat_tag=None,
    rebar_mat_tag=None,
    display_results=True,
    show=True,
    fill=False,
    show_legend=True,
    register=True,
):
    """
    Build and register a rectangular fiber section.

    top_bars and bottom_bars use [(n, dia), ...].
    side_bars uses [(n, dia), ...] and places n bars on each side.
    perimeter_bars uses (n, dia), [(n, dia), ...], or dicts with n/dia.
    """
    import opstool as opst

    cover_mat_tag = _resolve_frame_section_mat_tag(cover_mat_tag, "Tagcon")
    core_mat_tag = _resolve_frame_section_mat_tag(core_mat_tag, "Tagconcore")
    rebar_mat_tag = _resolve_frame_section_mat_tag(rebar_mat_tag, "Tagreinf")
    top_row_spacing = row_spacing if top_row_spacing is None else top_row_spacing
    bottom_row_spacing = row_spacing if bottom_row_spacing is None else bottom_row_spacing

    b = float(b)
    h = float(h)
    outlines = [[0, 0], [b, 0], [b, h], [0, h]]
    coverlines = opst.pre.section.offset(outlines, d=cover)

    cover_geo = opst.pre.section.create_polygon_patch(outlines, holes=[coverlines])
    core_geo = opst.pre.section.create_polygon_patch(coverlines)

    sec_mesh = opst.pre.section.FiberSecMesh(sec_name=sec_name)
    sec_mesh.add_patch_group({
        "cover": cover_geo,
        "core": core_geo,
    })
    sec_mesh.set_mesh_size({
        "cover": mesh_size_cover,
        "core": mesh_size_core,
    })
    sec_mesh.set_ops_mat_tag({
        "cover": cover_mat_tag,
        "core": core_mat_tag,
    })
    sec_mesh.set_mesh_color({
        "cover": "#dbb40c",
        "core": "#88b378",
    })
    if display_results or show:
        sec_mesh.mesh()
    else:
        with open(os.devnull, "w", encoding="utf-8") as _devnull:
            with redirect_stdout(_devnull), redirect_stderr(_devnull):
                sec_mesh.mesh()

    for item in _normalize_perimeter_bars(perimeter_bars):
        dia = float(item["dia"])
        offset = float(item.get("offset", dia / 2.0))
        points = opst.pre.section.offset(coverlines, d=offset)
        sec_mesh.add_rebar_line(
            points=points,
            dia=dia,
            n=int(item["n"]),
            ops_mat_tag=rebar_mat_tag,
            group_name=item.get("group_name", f"perimeter_rebar_{int(item['n'])}D{int(dia)}"),
            color=item.get("color", "#580f41"),
        )

    for row_i, bar in enumerate(_normalize_rebar_rows(top_bars)):
        n, dia = bar
        points = make_rebar_row_points(
            b=b,
            h=h,
            n=n,
            dia=dia,
            row_type="top",
            cover=cover,
            stirrup_dia=stirrup_dia,
            row_offset=row_i * top_row_spacing,
        )
        sec_mesh.add_rebar_points(
            points=points,
            dia=dia,
            ops_mat_tag=rebar_mat_tag,
            group_name=f"top_rebar_{int(n)}D{int(dia)}_row{row_i + 1}",
            color="#580f41",
        )

    for row_i, bar in enumerate(_normalize_rebar_rows(bottom_bars)):
        n, dia = bar
        points = make_rebar_row_points(
            b=b,
            h=h,
            n=n,
            dia=dia,
            row_type="bottom",
            cover=cover,
            stirrup_dia=stirrup_dia,
            row_offset=row_i * bottom_row_spacing,
        )
        sec_mesh.add_rebar_points(
            points=points,
            dia=dia,
            ops_mat_tag=rebar_mat_tag,
            group_name=f"bottom_rebar_{int(n)}D{int(dia)}_row{row_i + 1}",
            color="#580f41",
        )

    for row_i, bar in enumerate(_normalize_side_bars(side_bars)):
        if isinstance(bar, dict):
            n = bar["n"]
            dia = bar["dia"]
            end_offset = float(bar.get("end_offset", side_end_clear + row_i * row_spacing))
            color = bar.get("color", "#580f41")
        else:
            n, dia = bar
            end_offset = side_end_clear + row_i * row_spacing
            color = "#580f41"

        for side in ("left", "right"):
            points = make_rebar_side_points(
                b=b,
                h=h,
                n=n,
                dia=dia,
                side=side,
                cover=cover,
                stirrup_dia=stirrup_dia,
                end_offset=end_offset,
            )
            sec_mesh.add_rebar_points(
                points=points,
                dia=dia,
                ops_mat_tag=rebar_mat_tag,
                group_name=f"{side}_side_rebar_{int(n)}D{int(dia)}_row{row_i + 1}",
                color=color,
            )

    sec_mesh.centring()
    props = sec_mesh.get_frame_props(display_results=display_results)

    if show:
        ax = sec_mesh.view(fill=fill, show_legend=show_legend)
        if hasattr(ax, "set_title"):
            ax.set_title(sec_name)
        plt.show()

    sec_mesh.to_opspy_cmds(secTag=sec_tag, GJ=GJ)

    if register:
        register_frame_section_from_props(sec_tag, props, name=sec_name)

    return sec_mesh, props


def build_rect_retrofit_section(
    sec_tag,
    b,
    h,
    sec_name,
    retrofit_t,
    retrofit_cover_t=None,
    perimeter_bars=None,
    cover=25,
    stirrup_dia=8,
    mesh_size_cover=50,
    mesh_size_ring=50,
    mesh_size_core=50,
    GJ=1e13,
    cover_mat_tag=None,
    ring_mat_tag=None,
    core_mat_tag=None,
    rebar_mat_tag=None,
    display_results=True,
    show=True,
    fill=False,
    show_legend=True,
    register=True,
):
    """Build an OPSTOOL-backed rectangular column section with replaced outer concrete."""
    import opstool as opst

    cover_mat_tag = _resolve_frame_section_mat_tag(cover_mat_tag, "Tagcon")
    ring_mat_tag = _resolve_frame_section_mat_tag(ring_mat_tag, "Tagconcore")
    core_mat_tag = _resolve_frame_section_mat_tag(core_mat_tag, "Tagconcore")
    rebar_mat_tag = _resolve_frame_section_mat_tag(rebar_mat_tag, "Tagreinf")

    b = float(b)
    h = float(h)
    retrofit_t = float(retrofit_t)
    retrofit_cover_t = float(cover if retrofit_cover_t is None else retrofit_cover_t)
    if b <= 0.0 or h <= 0.0:
        raise ValueError("b and h must be positive")
    if retrofit_t <= retrofit_cover_t:
        raise ValueError("retrofit_t must be larger than retrofit_cover_t")
    if 2.0 * retrofit_t >= min(b, h):
        raise ValueError("retrofit_t leaves no original concrete core")

    outlines = [[0, 0], [b, 0], [b, h], [0, h]]
    coverlines = opst.pre.section.offset(outlines, d=retrofit_cover_t)
    corelines = opst.pre.section.offset(outlines, d=retrofit_t)

    cover_geo = opst.pre.section.create_polygon_patch(outlines, holes=[coverlines])
    ring_geo = opst.pre.section.create_polygon_patch(coverlines, holes=[corelines])
    core_geo = opst.pre.section.create_polygon_patch(corelines)

    sec_mesh = opst.pre.section.FiberSecMesh(sec_name=sec_name)
    sec_mesh.add_patch_group({
        "retrofit_cover": cover_geo,
        "retrofit_confined": ring_geo,
        "original_core": core_geo,
    })
    sec_mesh.set_mesh_size({
        "retrofit_cover": mesh_size_cover,
        "retrofit_confined": mesh_size_ring,
        "original_core": mesh_size_core,
    })
    sec_mesh.set_ops_mat_tag({
        "retrofit_cover": cover_mat_tag,
        "retrofit_confined": ring_mat_tag,
        "original_core": core_mat_tag,
    })
    sec_mesh.set_mesh_color({
        "retrofit_cover": "#d9c34a",
        "retrofit_confined": "#f0a33a",
        "original_core": "#88b378",
    })
    if display_results or show:
        sec_mesh.mesh()
    else:
        with open(os.devnull, "w", encoding="utf-8") as _devnull:
            with redirect_stdout(_devnull), redirect_stderr(_devnull):
                sec_mesh.mesh()

    rebar_base_lines = opst.pre.section.offset(outlines, d=float(cover))
    for item in _normalize_perimeter_bars(perimeter_bars):
        dia = float(item["dia"])
        offset = float(item.get("offset", dia / 2.0))
        points = opst.pre.section.offset(rebar_base_lines, d=offset)
        sec_mesh.add_rebar_line(
            points=points,
            dia=dia,
            n=int(item["n"]),
            ops_mat_tag=rebar_mat_tag,
            group_name=item.get("group_name", f"perimeter_rebar_{int(item['n'])}D{int(dia)}"),
            color=item.get("color", "#580f41"),
        )

    sec_mesh.centring()
    props = sec_mesh.get_frame_props(display_results=display_results)

    if show:
        ax = sec_mesh.view(fill=fill, show_legend=show_legend)
        if hasattr(ax, "set_title"):
            ax.set_title(sec_name)
        plt.show()

    sec_mesh.to_opspy_cmds(secTag=sec_tag, GJ=GJ)

    if register:
        register_frame_section_from_props(sec_tag, props, name=sec_name)

    return sec_mesh, props


def register_shell_section(sec_tag, width=None, thickness=None, area=None, name=None, **properties):
    sec_tag = int(sec_tag)
    if area is None and width is not None and thickness is not None:
        area = float(width) * float(thickness)
    info = {
        "sec_tag": sec_tag,
        "width": None if width is None else float(width),
        "thickness": None if thickness is None else float(thickness),
        "area": None if area is None else float(area),
        "name": name,
    }
    info.update(properties)
    SHELL_SECTION_REGISTRY[sec_tag] = info
    return info


def rebar_equivalent_layer_thickness(bar_dia, spacing):
    """Return the smeared rebar layer thickness A_bar / spacing in mm."""
    bar_dia = float(bar_dia)
    spacing = float(spacing)
    if bar_dia <= 0.0:
        raise ValueError("bar_dia must be positive")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    return math.pi * (bar_dia / 2.0) ** 2 / spacing


def rebar_area_from_dia(bar_dia):
    """Return one round bar area from diameter, in the same length unit squared."""
    bar_dia = float(bar_dia)
    if bar_dia <= 0.0:
        raise ValueError("bar_dia must be positive")
    return math.pi * (bar_dia / 2.0) ** 2


def wall_explicit_vertical_rebar_area(bar_dia, bars_per_line=2):
    """
    Return the truss area for one explicit wall vertical rebar line.

    In a shell wall model, one mid-surface truss line usually represents the
    two real bars on the two wall faces at the same in-plane position.
    """
    bars_per_line = float(bars_per_line)
    if bars_per_line <= 0.0:
        raise ValueError("bars_per_line must be positive")
    return bars_per_line * rebar_area_from_dia(bar_dia)


def column_mesh(
    column_height,
    anchor_node=None,
    anchor_point=None,
    z_coords=None,
    num_ele=None,
    transf_tag=1,
    integration_tag=1,
    ele_type="forceBeamColumn",
    node_tol=1e-6,
    node_map=None,
):
    """
    构建可嵌入整体结构模型的柱单元链。

    几何约定
    -------------------
    1. 柱沿全局 +Z 方向布置
    2. 锚点为柱底节点
    3. 如果锚点处已有节点，则自动复用
    4. 中间分段节点优先复用已有节点，避免重复建点
    """
    start_time = time.time()

    if column_height <= 0:
        raise ValueError("column_height 必须为正数")

    if z_coords is None:
        if num_ele is None:
            raise ValueError("提供 z_coords 或 num_ele")
        z_coords = np.linspace(0.0, column_height, int(num_ele) + 1)

    z_coords = np.asarray(z_coords, dtype=float)

    if len(z_coords) < 2:
        raise ValueError("z_coords 至少需要两个坐标")
    if abs(z_coords[0]) > node_tol:
        raise ValueError("z_coords 必须从柱底 0.0 开始")
    if np.any(np.diff(z_coords) <= 0.0):
        raise ValueError("z_coords 必须严格递增")
    if abs(z_coords[-1] - column_height) > node_tol:
        raise ValueError("z_coords[-1] 必须在容差范围内匹配 column_height")

    if anchor_node is not None and anchor_point is not None:
        raise ValueError("使用 anchor_node 或 anchor_point，不能同时使用")
    if anchor_node is not None:
        anchor_point = nodeCoord(int(anchor_node))
    elif anchor_point is None:
        anchor_point = (0.0, 0.0, 0.0)

    ox, oy, oz = [float(v) for v in anchor_point]

    if node_map is None:
        node_map = build_node_bucket_map(node_tol=node_tol)

    start_node_set = set(getNodeTags() or [])
    start_ele_set = set(getEleTags() or [])

    node_tags = []
    global_coords = []
    added_nodes = []
    reused_nodes = set()

    for z_local in z_coords:
        xg, yg, zg = ox, oy, oz + z_local
        tag, created = _get_or_create_node(xg, yg, zg, node_map, node_tol)
        node_tags.append(tag)
        global_coords.append((xg, yg, zg))
        if created:
            added_nodes.append(tag)
        else:
            reused_nodes.add(tag)

    next_ele_tag = max(getEleTags() or [0]) + 1
    added_elements = []

    for i in range(len(node_tags) - 1):
        n1 = node_tags[i]
        n2 = node_tags[i + 1]
        element(ele_type, next_ele_tag, n1, n2, transf_tag, integration_tag)
        added_elements.append(next_ele_tag)
        next_ele_tag += 1

    end_node_set = set(getNodeTags() or [])
    end_ele_set = set(getEleTags() or [])

    return {
        "anchor_node": anchor_node,
        "anchor_point": (ox, oy, oz),
        "column_height": float(column_height),
        "transf_tag": int(transf_tag),
        "integration_tag": int(integration_tag),
        "z_coords": z_coords.tolist(),
        "node_tags": node_tags,
        "global_coords": global_coords,
        "bottom_node": node_tags[0],
        "top_node": node_tags[-1],
        "added_nodes": added_nodes,
        "reused_nodes": sorted(reused_nodes),
        "added_elements": added_elements,
        "num_added_nodes": len(added_nodes),
        "num_reused_nodes": len(reused_nodes),
        "num_added_elements": len(added_elements),
        "node_map": node_map,
        "all_nodes_after_call": sorted(end_node_set),
        "all_elements_after_call": sorted(end_ele_set),
        "new_node_tags_by_diff": sorted(end_node_set - start_node_set),
        "new_element_tags_by_diff": sorted(end_ele_set - start_ele_set),
        "time_cost": time.time() - start_time,
    }

def print_model_summary():
    import inspect
    frame = inspect.currentframe().f_back
    local_vars = frame.f_locals

    print("====== 模型信息汇总 ======")
    for name, val in local_vars.items():
        if isinstance(val, dict) and "added_elements" in val:
            print(f"{name} 单元:", val["added_elements"])

    from openseespy.opensees import getNodeTags, getEleTags
    print("模型总节点数:", len(getNodeTags()))
    print("模型总单元数:", len(getEleTags()))

    

def retrofit_column_mesh(
    column_height,
    anchor_node=None,
    anchor_point=None,
    base_num_ele=None,
    base_integration_tag=1,
    retrofit_integration_tag=1,
    z_start=0.0,
    z_end=None,
    segment_coords=None,
    retrofit_num_ele=2,
    transf_tag=1,
    ele_type="forceBeamColumn",
    node_tol=1e-6,
    node_map=None,
):
    """Build one engineering column with different integrations along height."""
    import math as _math
    import time as _time
    import numpy as _np

    start_time = _time.time()
    column_height = float(column_height)
    if column_height <= 0.0:
        raise ValueError("column_height must be positive")
    z_start = float(z_start)
    z_end = column_height if z_end is None else float(z_end)
    if z_start < -node_tol or z_end > column_height + node_tol or z_end <= z_start:
        raise ValueError("invalid retrofit z_start/z_end")
    z_start = max(0.0, z_start)
    z_end = min(column_height, z_end)
    def _has_node(coords, target):
        return any(abs(float(z) - float(target)) <= node_tol for z in coords)

    if segment_coords is not None:
        if isinstance(segment_coords, str):
            raw_coords = [v.strip() for v in segment_coords.replace(",", ";").split(";") if v.strip()]
        else:
            raw_coords = list(segment_coords)
        if len(raw_coords) < 2:
            raise ValueError("segment_coords must contain at least two elevations")
        z_coords_list = [float(v) for v in raw_coords]
        if abs(z_coords_list[0]) > node_tol:
            raise ValueError("segment_coords must start at 0")
        if abs(z_coords_list[-1] - column_height) > node_tol:
            raise ValueError("segment_coords must end at column_height")
        for a, b in zip(z_coords_list, z_coords_list[1:]):
            if b <= a + node_tol:
                raise ValueError("segment_coords must be strictly increasing")
        if not _has_node(z_coords_list, z_start):
            raise ValueError("z_start must be one of segment_coords")
        if not _has_node(z_coords_list, z_end):
            raise ValueError("z_end must be one of segment_coords")
        z_coords = _np.asarray(z_coords_list, dtype=float)
    else:
        retrofit_num_ele = int(retrofit_num_ele)
        if retrofit_num_ele <= 0:
            raise ValueError("retrofit_num_ele must be positive")

        if base_num_ele is None:
            base_num_ele = max(1, retrofit_num_ele)
        base_num_ele = int(base_num_ele)
        if base_num_ele <= 0:
            raise ValueError("base_num_ele must be positive")
        target_len = column_height / base_num_ele

        def _interval_points(a, b, n):
            if b <= a + node_tol:
                return []
            return _np.linspace(a, b, int(n) + 1).tolist()

        coords = []
        if z_start > node_tol:
            n_before = max(1, int(_math.ceil((z_start - 0.0) / target_len)))
            coords.extend(_interval_points(0.0, z_start, n_before)[:-1])
        else:
            coords.append(0.0)

        coords.extend(_interval_points(z_start, z_end, retrofit_num_ele)[:-1])

        if z_end < column_height - node_tol:
            n_after = max(1, int(_math.ceil((column_height - z_end) / target_len)))
            coords.extend(_interval_points(z_end, column_height, n_after)[:-1])
        coords.append(column_height)

        z_coords_list = []
        for z in coords:
            z = float(z)
            if not z_coords_list or abs(z - z_coords_list[-1]) > node_tol:
                z_coords_list.append(z)
        z_coords = _np.asarray(z_coords_list, dtype=float)

    if anchor_node is not None and anchor_point is not None:
        raise ValueError("use anchor_node or anchor_point, not both")
    if anchor_node is not None:
        anchor_point = nodeCoord(int(anchor_node))
    elif anchor_point is None:
        anchor_point = (0.0, 0.0, 0.0)
    ox, oy, oz = [float(v) for v in anchor_point]

    if node_map is None:
        node_map = build_node_bucket_map(node_tol=node_tol)

    start_node_set = set(getNodeTags() or [])
    start_ele_set = set(getEleTags() or [])
    node_tags = []
    global_coords = []
    added_nodes = []
    reused_nodes = set()
    for z_local in z_coords:
        tag, created = _get_or_create_node(ox, oy, oz + float(z_local), node_map, node_tol)
        node_tags.append(tag)
        global_coords.append((ox, oy, oz + float(z_local)))
        if created:
            added_nodes.append(tag)
        else:
            reused_nodes.add(tag)

    next_ele_tag = max(getEleTags() or [0]) + 1
    added_elements = []
    element_integration_tags = []
    for i in range(len(node_tags) - 1):
        z_mid = 0.5 * (z_coords[i] + z_coords[i + 1])
        int_tag = retrofit_integration_tag if (z_mid >= z_start - node_tol and z_mid <= z_end + node_tol) else base_integration_tag
        element(ele_type, next_ele_tag, node_tags[i], node_tags[i + 1], transf_tag, int_tag)
        added_elements.append(next_ele_tag)
        element_integration_tags.append(int(int_tag))
        next_ele_tag += 1

    end_node_set = set(getNodeTags() or [])
    end_ele_set = set(getEleTags() or [])
    return {
        "anchor_node": anchor_node,
        "anchor_point": (ox, oy, oz),
        "column_height": float(column_height),
        "transf_tag": int(transf_tag),
        "integration_tag": int(base_integration_tag),
        "base_integration_tag": int(base_integration_tag),
        "retrofit_integration_tag": int(retrofit_integration_tag),
        "z_start": float(z_start),
        "z_end": float(z_end),
        "z_coords": z_coords.tolist(),
        "node_tags": node_tags,
        "global_coords": global_coords,
        "bottom_node": node_tags[0],
        "top_node": node_tags[-1],
        "added_nodes": added_nodes,
        "reused_nodes": sorted(reused_nodes),
        "added_elements": added_elements,
        "element_integration_tags": element_integration_tags,
        "num_added_nodes": len(added_nodes),
        "num_reused_nodes": len(reused_nodes),
        "num_added_elements": len(added_elements),
        "node_map": node_map,
        "all_nodes_after_call": sorted(end_node_set),
        "all_elements_after_call": sorted(end_ele_set),
        "new_node_tags_by_diff": sorted(end_node_set - start_node_set),
        "new_element_tags_by_diff": sorted(end_ele_set - start_ele_set),
        "time_cost": _time.time() - start_time,
    }

def beam_mesh(
    beam_length,
    anchor_node=None,
    anchor_point=None,
    axis="Y",
    span_dir=None,
    x_coords=None,
    num_ele=None,
    transf_tag=1,
    integration_tag=1,
    ele_type="forceBeamColumn",
    node_tol=1e-6,
    node_map=None,
):
    """
    构建可嵌入整体结构模型的梁单元链。

    几何约定
    -------------------
    1. 梁位于全局 X/Y 方向之一
    2. 锚点为梁起点节点
    3. 如果锚点处已有节点，则自动复用
    4. 中间分段节点优先复用已有节点，避免重复建点
    """
    start_time = time.time()

    if beam_length <= 0:
        raise ValueError("beam_length 必须为正数")

    if x_coords is None:
        if num_ele is None:
            raise ValueError("提供 x_coords 或 num_ele")
        x_coords = np.linspace(0.0, beam_length, int(num_ele) + 1)

    x_coords = np.asarray(x_coords, dtype=float)

    if len(x_coords) < 2:
        raise ValueError("x_coords 至少需要两个坐标")
    if abs(x_coords[0]) > node_tol:
        raise ValueError("x_coords 必须从梁起点 0.0 开始")
    if np.any(np.diff(x_coords) <= 0.0):
        raise ValueError("x_coords 必须严格递增")
    if abs(x_coords[-1] - beam_length) > node_tol:
        raise ValueError("x_coords[-1] 必须在容差范围内匹配 beam_length")

    axis = str(axis).upper()
    if axis not in {"X", "Y"}:
        raise ValueError("axis 必须是 'X' 或 'Y'")

    if span_dir is None:
        span_dir = f"+{axis}"
    span_dir = _normalize_axis_dir(span_dir)

    valid_dir = {"+X", "-X"} if axis == "X" else {"+Y", "-Y"}
    if span_dir not in valid_dir:
        raise ValueError(f"axis={axis} 要求 span_dir 在 {sorted(valid_dir)} 中")

    if anchor_node is not None and anchor_point is not None:
        raise ValueError("使用 anchor_node 或 anchor_point，不能同时使用")
    if anchor_node is not None:
        anchor_point = nodeCoord(int(anchor_node))
    elif anchor_point is None:
        anchor_point = (0.0, 0.0, 0.0)

    ox, oy, oz = [float(v) for v in anchor_point]

    if node_map is None:
        node_map = build_node_bucket_map(node_tol=node_tol)

    if axis == "X":
        direction = 1.0 if span_dir == "+X" else -1.0

        def local_to_global(x_local):
            return ox + direction * x_local, oy, oz

    else:
        direction = 1.0 if span_dir == "+Y" else -1.0

        def local_to_global(x_local):
            return ox, oy + direction * x_local, oz

    start_node_set = set(getNodeTags() or [])
    start_ele_set = set(getEleTags() or [])

    node_tags = []
    global_coords = []
    added_nodes = []
    reused_nodes = set()

    for x_local in x_coords:
        xg, yg, zg = local_to_global(x_local)
        tag, created = _get_or_create_node(xg, yg, zg, node_map, node_tol)
        node_tags.append(tag)
        global_coords.append((xg, yg, zg))
        if created:
            added_nodes.append(tag)
        else:
            reused_nodes.add(tag)

    next_ele_tag = max(getEleTags() or [0]) + 1
    added_elements = []

    for i in range(len(node_tags) - 1):
        n1 = node_tags[i]
        n2 = node_tags[i + 1]
        element(ele_type, next_ele_tag, n1, n2, transf_tag, integration_tag)
        added_elements.append(next_ele_tag)
        next_ele_tag += 1

    end_node_set = set(getNodeTags() or [])
    end_ele_set = set(getEleTags() or [])

    return {
        "axis": axis,
        "span_dir": span_dir,
        "anchor_node": anchor_node,
        "anchor_point": (ox, oy, oz),
        "beam_length": float(beam_length),
        "transf_tag": int(transf_tag),
        "integration_tag": int(integration_tag),
        "x_coords": x_coords.tolist(),
        "node_tags": node_tags,
        "global_coords": global_coords,
        "start_node": node_tags[0],
        "end_node": node_tags[-1],
        "added_nodes": added_nodes,
        "reused_nodes": sorted(reused_nodes),
        "added_elements": added_elements,
        "num_added_nodes": len(added_nodes),
        "num_reused_nodes": len(reused_nodes),
        "num_added_elements": len(added_elements),
        "node_map": node_map,
        "all_nodes_after_call": sorted(end_node_set),
        "all_elements_after_call": sorted(end_ele_set),
        "new_node_tags_by_diff": sorted(end_node_set - start_node_set),
        "new_element_tags_by_diff": sorted(end_ele_set - start_ele_set),
        "time_cost": time.time() - start_time,
    }

def _collect_boundary_coords(
    anchor_point,
    plane,
    span_dir,
    wall_width,
    wall_height,
    node_tol,
):
    ox, oy, oz = anchor_point

    if plane == "XOZ":
        direction = 1.0 if span_dir == "+X" else -1.0

        def global_to_local(xg, yg, zg):
            return direction * (xg - ox), yg - oy, zg - oz

    else:
        direction = 1.0 if span_dir == "+Y" else -1.0

        def global_to_local(xg, yg, zg):
            return direction * (yg - oy), xg - ox, zg - oz

    x_hits = [0.0, wall_width]
    z_hits = [0.0, wall_height]

    for tag in getNodeTags() or []:
        xg, yg, zg = nodeCoord(tag)
        x_local, normal_offset, z_local = global_to_local(xg, yg, zg)

        if abs(normal_offset) > node_tol:
            continue
        if x_local < -node_tol or x_local > wall_width + node_tol:
            continue
        if z_local < -node_tol or z_local > wall_height + node_tol:
            continue

        if abs(z_local) <= node_tol or abs(z_local - wall_height) <= node_tol:
            x_hits.append(min(max(x_local, 0.0), wall_width))

        if abs(x_local) <= node_tol or abs(x_local - wall_width) <= node_tol:
            z_hits.append(min(max(z_local, 0.0), wall_height))

    return _unique_sorted_coords(x_hits, node_tol), _unique_sorted_coords(z_hits, node_tol)


def wall_mesh_by_base_nodes(
    wall_height,
    base_i_node=None,
    base_j_node=None,
    base_i_point=None,
    base_j_point=None,
    x_coords=None,
    y_coords=None,
    z_coords=None,
    num_ele_x=None,
    num_ele_y=None,
    edge_zone_width=0.0,
    sec_tag_edge=1,
    sec_tag_wall=2,
    shell_type="ShellNLDKGQ", # ShellMITC4 ShellNLDKGQ
    node_tol=1e-6,
    node_map=None,
    match_existing_boundary_nodes=True,
    zones=None,
):
    """
    用墙底边两个点/两个节点定义剪力墙。

    参数说明
    ----------
    wall_height : float
        墙高，沿全局 +Z 方向。
    base_i_node/base_j_node : int, optional
        墙底边起点和终点对应的已有节点号。
    base_i_point/base_j_point : tuple(float, float, float), optional
        墙底边起点和终点坐标；如果该点已经在模型中存在，wall_mesh 会自动复用节点。
    match_existing_boundary_nodes : bool
        为 True 时，自动扫描墙四条边上已经存在的节点，并把它们并入 x_coords / y_coords，
        更适合墙-墙共边、墙-梁共角点的整体结构模型。

    依赖
    ----
    需要与原文件中的 wall_mesh(...) 函数放在同一模块内使用。
    """
    base_i, resolved_i_node = _resolve_point_from_node_or_coord(
        node_tag=base_i_node,
        point=base_i_point,
        label="base_i",
    )
    base_j, resolved_j_node = _resolve_point_from_node_or_coord(
        node_tag=base_j_node,
        point=base_j_point,
        label="base_j",
    )

    geom = _resolve_base_geometry(base_i, base_j, node_tol=node_tol)
    plane = geom["plane"]
    span_dir = geom["span_dir"]
    wall_width = geom["wall_width"]
    anchor_point = geom["anchor_point"]

    if z_coords is not None:
        if y_coords is not None:
            raise ValueError("y_coords 和 z_coords 不能同时提供")
        y_coords = z_coords

    if match_existing_boundary_nodes:
        auto_x, auto_z = _collect_boundary_coords(
            anchor_point=anchor_point,
            plane=plane,
            span_dir=span_dir,
            wall_width=wall_width,
            wall_height=wall_height,
            node_tol=node_tol,
        )
        if x_coords is None:
            x_coords = auto_x
        else:
            x_coords = _unique_sorted_coords(list(x_coords) + auto_x, node_tol)

        if y_coords is None:
            y_coords = auto_z
        else:
            y_coords = _unique_sorted_coords(list(y_coords) + auto_z, node_tol)

    result = wall_mesh(
        wall_width=wall_width,
        wall_height=wall_height,
        anchor_node=resolved_i_node,
        anchor_point=None if resolved_i_node is not None else anchor_point,
        plane=plane,
        span_dir=span_dir,
        x_coords=x_coords,
        y_coords=y_coords,
        num_ele_x=num_ele_x,
        num_ele_y=num_ele_y,
        edge_zone_width=edge_zone_width,
        sec_tag_edge=sec_tag_edge,
        sec_tag_wall=sec_tag_wall,
        zones=zones,
        shell_type=shell_type,
        node_tol=node_tol,
        node_map=node_map,
    )

    result.update(
        {
            "base_i_node_input": resolved_i_node,
            "base_j_node_input": resolved_j_node,
            "base_i_point": base_i,
            "base_j_point": base_j,
            "matched_existing_boundary_nodes": bool(match_existing_boundary_nodes),
        }
    )
    return result

def _get_or_create_node(x, y, z, bucket_map, node_tol):
    existing = _find_node_in_buckets(x, y, z, bucket_map, node_tol)
    if existing is not None:
        return existing, False

    new_tag = max(getNodeTags() or [0]) + 1
    node(new_tag, float(x), float(y), float(z))
    key = _coord_key(x, y, z, node_tol)
    bucket_map.setdefault(key, []).append((new_tag, (float(x), float(y), float(z))))
    return new_tag, True


def get_or_create_node(x, y, z, node_map=None, node_tol=1e-6):
    """
    用于其他组件构建器（如梁或楼板）的公共辅助函数
    """
    if node_map is None:
        node_map = build_node_bucket_map(node_tol=node_tol)
    tag, created = _get_or_create_node(x, y, z, node_map, node_tol)
    return tag, created, node_map


def _normalize_axis_dir(axis_dir):
    cleaned = str(axis_dir).replace(" ", "").upper()
    if cleaned not in {"+X", "-X", "+Y", "-Y"}:
        raise ValueError("span_dir/normal_dir 必须是 +X, -X, +Y, -Y 之一")
    return cleaned


def _resolve_span_dir(plane, span_dir=None, normal_dir=None):
    plane = str(plane).upper()
    if plane not in {"XOZ", "YOZ"}:
        raise ValueError("plane 必须是 'XOZ' 或 'YOZ'")

    if span_dir is not None:
        span_dir = _normalize_axis_dir(span_dir)
    if normal_dir is not None:
        normal_dir = _normalize_axis_dir(normal_dir)

    if plane == "XOZ":
        valid_span = {"+X", "-X"}
        valid_normal = {"+Y", "-Y"}
        normal_to_span = {"+Y": "-X", "-Y": "+X"}
    else:
        valid_span = {"+Y", "-Y"}
        valid_normal = {"+X", "-X"}
        normal_to_span = {"+X": "+Y", "-X": "-Y"}

    if span_dir is None and normal_dir is None:
        span_dir = min(valid_span)

    if span_dir is not None and span_dir not in valid_span:
        raise ValueError(f"plane={plane} 要求 span_dir 在 {sorted(valid_span)} 中")

    if normal_dir is not None and normal_dir not in valid_normal:
        raise ValueError(f"plane={plane} 要求 normal_dir 在 {sorted(valid_normal)} 中")

    if span_dir is None:
        span_dir = normal_to_span[normal_dir]

    if normal_dir is not None and normal_to_span[normal_dir] != span_dir:
        raise ValueError("span_dir 和 normal_dir 不一致")

    return plane, span_dir


def _wall_normal_from_span(plane, span_dir):
    if plane == "XOZ":
        return (0.0, -1.0, 0.0) if span_dir == "+X" else (0.0, 1.0, 0.0)
    return (1.0, 0.0, 0.0) if span_dir == "+Y" else (-1.0, 0.0, 0.0)


def _unique_wall_x_positions(values, node_tol):
    positions = sorted(float(value) for value in values)
    unique = []
    for value in positions:
        if not unique or abs(value - unique[-1]) > node_tol:
            unique.append(value)
    return unique


def _wall_zone_kind(zone):
    kind = str(zone.get("kind", zone.get("zone_type", "center"))).strip().lower()
    aliases = {
        "wall": "center",
        "center": "center",
        "center_zone": "center",
        "edge": "edge",
        "left_edge": "edge",
        "right_edge": "edge",
        "edge_zone": "edge",
    }
    if kind not in aliases:
        raise ValueError(f"Unsupported wall zone kind: {kind!r}")
    return aliases[kind]


def _normalize_wall_zones(zones, wall_width, x_coords=None, node_tol=1e-6):
    if not zones:
        raise ValueError("zones must contain at least one wall zone")

    normalized = []
    for index, zone in enumerate(zones):
        if not isinstance(zone, dict):
            raise TypeError("Each wall zone must be a dict")
        if "x0" not in zone or "x1" not in zone:
            raise ValueError("Each wall zone must define x0 and x1")
        if "sec_tag" not in zone:
            raise ValueError("Each wall zone must define sec_tag")

        x0 = float(zone["x0"])
        x1 = float(zone["x1"])
        if x1 - x0 <= node_tol:
            raise ValueError(f"Wall zone {index} has non-positive width")

        normalized.append({
            "name": str(zone.get("name", f"zone_{index + 1}")),
            "kind": _wall_zone_kind(zone),
            "x0": x0,
            "x1": x1,
            "sec_tag": int(zone["sec_tag"]),
        })

    normalized.sort(key=lambda item: item["x0"])
    if abs(normalized[0]["x0"]) > node_tol:
        raise ValueError("Wall zones must start at x=0.0")
    if abs(normalized[-1]["x1"] - float(wall_width)) > node_tol:
        raise ValueError("Wall zones must end at wall_width")

    for left, right in zip(normalized[:-1], normalized[1:]):
        if abs(left["x1"] - right["x0"]) > node_tol:
            raise ValueError("Wall zones must cover the width continuously without gaps or overlap")

    if x_coords is not None:
        x_coords = np.asarray(x_coords, dtype=float)
        boundaries = [normalized[0]["x0"]] + [zone["x1"] for zone in normalized]
        missing = [
            boundary
            for boundary in boundaries
            if not np.any(np.isclose(x_coords, boundary, atol=node_tol, rtol=0.0))
        ]
        if missing:
            missing_text = ", ".join(f"{value:g}" for value in missing)
            raise ValueError(
                "Wall zone boundaries must be present in x_coords. "
                f"Missing x positions: {missing_text}"
            )

    return normalized


def make_wall_zones(
    wall_width,
    zone_type,
    sec_tag_center=None,
    sec_tag_edge=None,
    sec_tag_left_edge=None,
    sec_tag_right_edge=None,
    left_edge_width=None,
    right_edge_width=None,
    split_x=None,
    node_tol=1e-6,
):
    """Build explicit wall-width zones for table-driven shell meshing."""
    wall_width = float(wall_width)
    zone_type = str(zone_type).strip().lower()
    aliases = {
        "center": "center_only",
        "center_only": "center_only",
        "edge_center_edge": "edge_center_edge",
        "edge+center+edge": "edge_center_edge",
        "edge_only": "edge_only",
        "all_edge": "edge_only",
        "edge_pair": "edge_pair",
        "all_edge_pair": "edge_pair",
    }
    if zone_type not in aliases:
        raise ValueError(f"Unsupported wall zone_type: {zone_type!r}")
    zone_type = aliases[zone_type]

    if wall_width <= 0.0:
        raise ValueError("wall_width must be positive")

    if zone_type == "center_only":
        if sec_tag_center is None:
            raise ValueError("sec_tag_center is required for center_only walls")
        zones = [{
            "name": "center",
            "kind": "center",
            "x0": 0.0,
            "x1": wall_width,
            "sec_tag": sec_tag_center,
        }]

    elif zone_type == "edge_only":
        edge_tag = sec_tag_edge
        if edge_tag is None:
            edge_tag = sec_tag_left_edge if sec_tag_left_edge is not None else sec_tag_right_edge
        if edge_tag is None:
            raise ValueError("sec_tag_edge is required for edge_only walls")
        zones = [{
            "name": "edge_all",
            "kind": "edge",
            "x0": 0.0,
            "x1": wall_width,
            "sec_tag": edge_tag,
        }]

    elif zone_type == "edge_pair":
        if sec_tag_left_edge is None or sec_tag_right_edge is None:
            raise ValueError("edge_pair walls require both left and right edge section tags")
        if split_x is None:
            if left_edge_width is not None:
                split_x = left_edge_width
            elif right_edge_width is not None:
                split_x = wall_width - float(right_edge_width)
            else:
                raise ValueError("edge_pair walls require split_x or one edge width")
        split_x = float(split_x)
        if split_x <= node_tol or split_x >= wall_width - node_tol:
            raise ValueError("edge_pair split_x must fall inside the wall width")
        zones = [
            {
                "name": "left_edge",
                "kind": "edge",
                "x0": 0.0,
                "x1": split_x,
                "sec_tag": sec_tag_left_edge,
            },
            {
                "name": "right_edge",
                "kind": "edge",
                "x0": split_x,
                "x1": wall_width,
                "sec_tag": sec_tag_right_edge,
            },
        ]

    else:
        if sec_tag_center is None:
            raise ValueError("sec_tag_center is required for edge_center_edge walls")
        left_edge_width = 0.0 if left_edge_width is None else float(left_edge_width)
        right_edge_width = 0.0 if right_edge_width is None else float(right_edge_width)
        if left_edge_width < 0.0 or right_edge_width < 0.0:
            raise ValueError("Edge widths cannot be negative")
        if left_edge_width + right_edge_width >= wall_width - node_tol:
            raise ValueError("edge_center_edge walls must retain a positive center-zone width")

        left_tag = sec_tag_left_edge if sec_tag_left_edge is not None else sec_tag_edge
        right_tag = sec_tag_right_edge if sec_tag_right_edge is not None else sec_tag_edge
        if left_edge_width > node_tol and left_tag is None:
            raise ValueError("A left edge section tag is required when left_edge_width is positive")
        if right_edge_width > node_tol and right_tag is None:
            raise ValueError("A right edge section tag is required when right_edge_width is positive")

        zones = []
        if left_edge_width > node_tol:
            zones.append({
                "name": "left_edge",
                "kind": "edge",
                "x0": 0.0,
                "x1": left_edge_width,
                "sec_tag": left_tag,
            })
        zones.append({
            "name": "center",
            "kind": "center",
            "x0": left_edge_width,
            "x1": wall_width - right_edge_width,
            "sec_tag": sec_tag_center,
        })
        if right_edge_width > node_tol:
            zones.append({
                "name": "right_edge",
                "kind": "edge",
                "x0": wall_width - right_edge_width,
                "x1": wall_width,
                "sec_tag": right_tag,
            })

    return _normalize_wall_zones(zones, wall_width, node_tol=node_tol)


def wall_mesh(
    wall_width,
    wall_height,
    anchor_node=None,
    anchor_point=None,
    plane="XOZ",
    span_dir=None,
    normal_dir=None,
    x_coords=None,
    y_coords=None,
    z_coords=None,
    num_ele_x=None,
    num_ele_y=None,
    edge_zone_width=0.0,
    sec_tag_edge=1,
    sec_tag_wall=2,
    shell_type="ShellNLDKGQ",
    node_tol=1e-6,
    node_map=None,
    zones=None,
):
    """
    构建可嵌入完整结构模型的剪力墙壳网格

    几何约定
    -------------------
    1. 墙体位于 XOZ 平面或 YOZ 平面
    2. 锚点是墙体的一个底角
    3. 墙体宽度从锚点沿 span_dir 增长
    4. 墙体高度始终沿全局 +Z 方向增长
    """
    start_time = time.time()

    if wall_width <= 0 or wall_height <= 0:
        raise ValueError("wall_width 和 wall_height 必须为正数")

    if x_coords is None:
        if num_ele_x is None:
            raise ValueError("提供 x_coords 或 num_ele_x")
        x_coords = np.linspace(0.0, wall_width, int(num_ele_x) + 1)

    if z_coords is not None:
        if y_coords is not None:
            raise ValueError("使用 y_coords 或 z_coords，不能同时使用")
        y_coords = z_coords

    if y_coords is None:
        if num_ele_y is None:
            raise ValueError("提供 y_coords/z_coords 或 num_ele_y")
        y_coords = np.linspace(0.0, wall_height, int(num_ele_y) + 1)

    x_coords = np.asarray(x_coords, dtype=float)
    y_coords = np.asarray(y_coords, dtype=float)

    if len(x_coords) < 2 or len(y_coords) < 2:
        raise ValueError("每个方向至少需要两个坐标")
    if abs(x_coords[0]) > node_tol:
        raise ValueError("x_coords 必须从锚角处的 0.0 开始")
    if abs(y_coords[0]) > node_tol:
        raise ValueError("y_coords/z_coords 必须从墙底的 0.0 开始")
    if np.any(np.diff(x_coords) <= 0.0):
        raise ValueError("x_coords 必须严格递增")
    if np.any(np.diff(y_coords) <= 0.0):
        raise ValueError("y_coords/z_coords 必须严格递增")
    if abs(x_coords[-1] - wall_width) > node_tol:
        raise ValueError("x_coords[-1] 必须在容差范围内匹配 wall_width")
    if abs(y_coords[-1] - wall_height) > node_tol:
        raise ValueError("y_coords/z_coords[-1] 必须在容差范围内匹配 wall_height")
    if zones is None:
        if edge_zone_width < 0.0 or edge_zone_width > wall_width / 2.0:
            raise ValueError("edge_zone_width 必须在 0 和 wall_width/2 之间")
    else:
        zones = _normalize_wall_zones(zones, wall_width, x_coords=x_coords, node_tol=node_tol)

    plane, span_dir = _resolve_span_dir(plane, span_dir=span_dir, normal_dir=normal_dir)

    if anchor_node is not None and anchor_point is not None:
        raise ValueError("使用 anchor_node 或 anchor_point，不能同时使用")
    if anchor_node is not None:
        anchor_point = nodeCoord(int(anchor_node))
    elif anchor_point is None:
        anchor_point = (0.0, 0.0, 0.0)

    ox, oy, oz = [float(v) for v in anchor_point]

    if node_map is None:
        node_map = build_node_bucket_map(node_tol=node_tol)

    if plane == "XOZ":
        direction = 1.0 if span_dir == "+X" else -1.0

        def local_to_global(x_local, z_local):
            return ox + direction * x_local, oy, oz + z_local

    else:
        direction = 1.0 if span_dir == "+Y" else -1.0

        def local_to_global(x_local, z_local):
            return ox, oy + direction * x_local, oz + z_local

    start_node_set = set(getNodeTags() or [])
    start_ele_set = set(getEleTags() or [])

    node_grid = []
    added_nodes = []
    reused_nodes = set()
    global_coords_grid = []

    for z_local in y_coords:
        row = []
        coord_row = []
        for x_local in x_coords:
            xg, yg, zg = local_to_global(x_local, z_local)
            tag, created = _get_or_create_node(xg, yg, zg, node_map, node_tol)
            row.append(tag)
            coord_row.append((xg, yg, zg))
            if created:
                added_nodes.append(tag)
            else:
                reused_nodes.add(tag)
        node_grid.append(row)
        global_coords_grid.append(coord_row)

    next_ele_tag = max(getEleTags() or [0]) + 1
    added_elements = []
    edge_elements = []
    wall_elements = []
    zone_elements = {zone["name"]: [] for zone in zones} if zones is not None else {}

    for j in range(len(y_coords) - 1):
        for i in range(len(x_coords) - 1):
            x_center = 0.5 * (x_coords[i] + x_coords[i + 1])
            if zones is not None:
                zone = next(
                    (
                        item for item in zones
                        if item["x0"] - node_tol <= x_center <= item["x1"] + node_tol
                    ),
                    None,
                )
                if zone is None:
                    raise ValueError(f"No wall zone covers x={x_center:g}")
                sec_tag = zone["sec_tag"]
                target_list = edge_elements if zone["kind"] == "edge" else wall_elements
                zone_target_list = zone_elements[zone["name"]]
            else:
                if x_center < edge_zone_width or x_center > (wall_width - edge_zone_width):
                    sec_tag = sec_tag_edge
                    target_list = edge_elements
                else:
                    sec_tag = sec_tag_wall
                    target_list = wall_elements
                zone_target_list = None

            n1 = node_grid[j][i]
            n2 = node_grid[j][i + 1]
            n3 = node_grid[j + 1][i + 1]
            n4 = node_grid[j + 1][i]

            element(shell_type, next_ele_tag, n1, n2, n3, n4, sec_tag)
            added_elements.append(next_ele_tag)
            target_list.append(next_ele_tag)
            if zone_target_list is not None:
                zone_target_list.append(next_ele_tag)
            next_ele_tag += 1

    end_node_set = set(getNodeTags() or [])
    end_ele_set = set(getEleTags() or [])

    return {
        "plane": plane,
        "span_dir": span_dir,
        "wall_normal": _wall_normal_from_span(plane, span_dir),
        "anchor_node": anchor_node,
        "anchor_point": (ox, oy, oz),
        "sec_tag_edge": int(sec_tag_edge),
        "sec_tag_wall": int(sec_tag_wall),
        "zones": None if zones is None else [dict(zone) for zone in zones],
        "shell_type": shell_type,
        "x_coords": x_coords.tolist(),
        "z_coords": y_coords.tolist(),
        "node_grid": node_grid,
        "global_coords_grid": global_coords_grid,
        "bottom_nodes": node_grid[0],
        "top_nodes": node_grid[-1],
        "left_edge_nodes": [row[0] for row in node_grid],
        "right_edge_nodes": [row[-1] for row in node_grid],
        "corner_nodes": {
            "bottom_start": node_grid[0][0],
            "bottom_end": node_grid[0][-1],
            "top_start": node_grid[-1][0],
            "top_end": node_grid[-1][-1],
        },
        "control_node": node_grid[-1][len(x_coords) // 2],
        "added_nodes": added_nodes,
        "reused_nodes": sorted(reused_nodes),
        "added_elements": added_elements,
        "edge_elements": edge_elements,
        "wall_elements": wall_elements,
        "zone_elements": zone_elements,
        "num_added_nodes": len(added_nodes),
        "num_reused_nodes": len(reused_nodes),
        "num_added_elements": len(added_elements),
        "node_map": node_map,
        "all_nodes_after_call": sorted(end_node_set),
        "all_elements_after_call": sorted(end_ele_set),
        "new_node_tags_by_diff": sorted(end_node_set - start_node_set),
        "new_element_tags_by_diff": sorted(end_ele_set - start_ele_set),
        "time_cost": time.time() - start_time,
    }


def wall_mesh_zoned(
    wall_width,
    wall_height,
    anchor_node=None,
    anchor_point=None,
    plane="XOZ",
    span_dir=None,
    normal_dir=None,
    x_coords=None,
    y_coords=None,
    z_coords=None,
    num_ele_x=None,
    num_ele_y=None,
    zones=None,
    zone_type=None,
    sec_tag_center=None,
    sec_tag_edge=None,
    sec_tag_left_edge=None,
    sec_tag_right_edge=None,
    left_edge_width=None,
    right_edge_width=None,
    split_x=None,
    shell_type="ShellNLDKGQ",
    node_tol=1e-6,
    node_map=None,
):
    """Build a wall mesh from explicit shell zones without changing wall placement."""
    if zones is None:
        if zone_type is None:
            raise ValueError("Provide zones or zone_type")
        zones = make_wall_zones(
            wall_width=wall_width,
            zone_type=zone_type,
            sec_tag_center=sec_tag_center,
            sec_tag_edge=sec_tag_edge,
            sec_tag_left_edge=sec_tag_left_edge,
            sec_tag_right_edge=sec_tag_right_edge,
            left_edge_width=left_edge_width,
            right_edge_width=right_edge_width,
            split_x=split_x,
            node_tol=node_tol,
        )

    center_tag = sec_tag_center
    edge_tag = sec_tag_edge
    if center_tag is None:
        center_tag = next(
            (zone["sec_tag"] for zone in zones if _wall_zone_kind(zone) == "center"),
            None,
        )
    if center_tag is None:
        center_tag = next(
            (zone["sec_tag"] for zone in zones if _wall_zone_kind(zone) == "edge"),
            1,
        )
    if edge_tag is None:
        edge_tag = next(
            (zone["sec_tag"] for zone in zones if _wall_zone_kind(zone) == "edge"),
            center_tag,
        )

    return wall_mesh(
        wall_width=wall_width,
        wall_height=wall_height,
        anchor_node=anchor_node,
        anchor_point=anchor_point,
        plane=plane,
        span_dir=span_dir,
        normal_dir=normal_dir,
        x_coords=x_coords,
        y_coords=y_coords,
        z_coords=z_coords,
        num_ele_x=num_ele_x,
        num_ele_y=num_ele_y,
        sec_tag_edge=edge_tag,
        sec_tag_wall=center_tag,
        zones=zones,
        shell_type=shell_type,
        node_tol=node_tol,
        node_map=node_map,
    )
_WALL_VERTICAL_REBAR_NODE_PAIRS = set()


def reset_vertical_rebar_registry():
    """Clear duplicate-check state for explicit wall vertical rebars."""
    _WALL_VERTICAL_REBAR_NODE_PAIRS.clear()


def add_vertical_rebars(node_grid, cols, area, mat_tag, start_ele_tag=None):
    """
    为 wall_mesh 生成的节点网格添加垂直钢筋（truss 单元）
    
    参数:
    node_grid: wall_mesh 返回的节点网格
    cols: 要添加钢筋的列索引列表 list[int]
    area: 钢筋面积 (m²)
    mat_tag: 材料标签
    start_ele_tag: 起始单元标签（可选）
    """
    if start_ele_tag is None:
        existing_eles = getEleTags()
        start_ele_tag = max(existing_eles) if existing_eles else 0

    ele_tag = start_ele_tag + 1
    skipped_duplicates = 0
    for col in cols:
        for row in range(len(node_grid) - 1):
            n1 = node_grid[row][col]
            n2 = node_grid[row + 1][col]
            pair_key = tuple(sorted((int(n1), int(n2))))
            if pair_key in _WALL_VERTICAL_REBAR_NODE_PAIRS:
                skipped_duplicates += 1
                continue
            element("truss", ele_tag, n1, n2, area, mat_tag)
            _WALL_VERTICAL_REBAR_NODE_PAIRS.add(pair_key)
            ele_tag += 1

    num_rebars = ele_tag - start_ele_tag - 1
    if skipped_duplicates:
        print(f"Skipped {skipped_duplicates} duplicate vertical rebar elements")
    print(f"已创建 {num_rebars} 个 truss 单元")
    return ele_tag - 1


def wall_grid_column_indices_by_x(wall_data, x_positions, node_tol=1e-6):
    """Map local wall x positions to existing wall mesh column indices."""
    if "x_coords" not in wall_data:
        raise ValueError("wall_data must contain x_coords")

    x_coords = np.asarray(wall_data["x_coords"], dtype=float)
    positions = _unique_wall_x_positions(x_positions, node_tol)
    cols = []
    missing = []
    for position in positions:
        matches = np.flatnonzero(np.isclose(x_coords, position, atol=node_tol, rtol=0.0))
        if len(matches) == 0:
            missing.append(position)
        else:
            cols.append(int(matches[0]))

    if missing:
        missing_text = ", ".join(f"{value:g}" for value in missing)
        raise ValueError(
            "Vertical rebar x positions must be wall mesh node columns. "
            f"Missing x positions: {missing_text}"
        )

    return cols


def add_vertical_rebars_by_x(
    wall_data,
    x_positions,
    area=None,
    mat_tag=None,
    start_ele_tag=None,
    node_tol=1e-6,
    bar_dia=None,
    bars_per_line=2,
):
    """
    Add vertical truss rebars on existing wall mesh columns selected by local x.

    Pass either:
    - area: explicit truss area per vertical line, preserving old behavior.
    - bar_dia: one real bar diameter. The function uses bars_per_line * A_bar,
      defaulting to two bars per line for the two wall faces represented by the
      shell mid-surface.
    """
    if "node_grid" not in wall_data:
        raise ValueError("wall_data must contain node_grid")
    if mat_tag is None:
        raise ValueError("mat_tag is required")
    if area is not None and bar_dia is not None:
        raise ValueError("Provide either area or bar_dia, not both")
    if area is None:
        if bar_dia is None:
            raise ValueError("Provide area or bar_dia")
        area = wall_explicit_vertical_rebar_area(bar_dia, bars_per_line=bars_per_line)
    area = float(area)
    if area <= 0.0:
        raise ValueError("area must be positive")

    positions = _unique_wall_x_positions(x_positions, node_tol)
    cols = wall_grid_column_indices_by_x(wall_data, positions, node_tol=node_tol)
    last_ele_tag = add_vertical_rebars(
        node_grid=wall_data["node_grid"],
        cols=cols,
        area=area,
        mat_tag=mat_tag,
        start_ele_tag=start_ele_tag,
    )
    return {
        "x_positions": positions,
        "cols": cols,
        "area": area,
        "bar_dia": None if bar_dia is None else float(bar_dia),
        "bars_per_line": None if bar_dia is None else float(bars_per_line),
        "last_ele_tag": last_ele_tag,
        "num_rebar_lines": len(cols),
    }

# ==================================
# ========== 可视化函数 ============
# ==================================
def plot_model(show_node=True, show_ele=True,
               node_size=12, ele_size=12, show_local_axes=True):
    """
    使用 opstool.vis.pyvista 可视化模型
    
    参数:
    show_node: 是否显示节点标签
    show_ele: 是否显示单元标签
    node_size: 节点标签字体大小
    ele_size: 单元标签字体大小
    show_local_axes: 是否显示局部坐标轴
    """
    from openseespy.opensees import getNodeTags, nodeCoord, getEleTags, eleNodes
    import opstool as opst

    # 设置绘图属性
    opst.vis.pyvista.set_plot_props(notebook=True)
    fig = opst.vis.pyvista.plot_model(show_local_axes=show_local_axes)

    # 添加节点标签
    if show_node:
        node_tags = getNodeTags()
        coords = np.array([nodeCoord(tag) for tag in node_tags])
        if coords.shape[1] == 2:
            coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
        fig.add_point_labels(coords,
                             [str(tag) for tag in node_tags],
                             font_size=node_size,
                             point_color="black")

    # 添加单元标签
    if show_ele:
        ele_tags = getEleTags()
        ele_centers = []
        for ele in ele_tags:
            node_ids = eleNodes(ele)
            node_coords = np.array([nodeCoord(nid) for nid in node_ids])
            if node_coords.shape[1] == 2:
                node_coords = np.hstack([node_coords, np.zeros((node_coords.shape[0], 1))])
            center = node_coords.mean(axis=0)
            ele_centers.append(center)

        fig.add_point_labels(ele_centers,
                             [str(tag) for tag in ele_tags],
                             font_size=ele_size,
                             point_color="blue")

    fig.show()





def get_shell_layer_strains(
    odb_tag,
    wall_data,
    layer=1,
    lazy_load=False,
):
    """Return eps11, eps22, and engineering gamma12 for one shell layer.

    Responses are averaged over shell Gauss points and ordered by
    ``wall_data["added_elements"]`` so they can be used directly by the wall
    cloud-plotting utilities.
    """
    import opstool as opst

    strain = opst.post.get_element_responses(
        odb_tag,
        ele_type="Shell",
        resp_type="Strains",
        lazy_load=lazy_load,
        print_info=False,
    )

    available_layers = strain.coords["fiberPoints"].to_numpy().tolist()
    if layer not in available_layers:
        raise ValueError(
            f"Shell layer {layer} is unavailable. "
            f"Available layers are {available_layers}."
        )

    strain = (
        strain
        .sel(fiberPoints=layer)
        .mean(dim="GaussPoints", skipna=True)
        .sel(eleTags=wall_data["added_elements"])
    )
    eps11 = strain.sel(stressDOFs="sigma11").to_numpy()
    eps22 = strain.sel(stressDOFs="sigma22").to_numpy()
    gamma12 = strain.sel(stressDOFs="sigma12").to_numpy()
    return eps11, eps22, gamma12


def principal_strains_2d(eps11, eps22, gamma12):
    """Calculate maximum and minimum in-plane principal strains.

    ``gamma12`` is the engineering shear strain, so ``gamma12 / 2`` is used
    in the Mohr-circle radius.
    """
    eps11 = np.asarray(eps11, dtype=float)
    eps22 = np.asarray(eps22, dtype=float)
    gamma12 = np.asarray(gamma12, dtype=float)
    if eps11.shape != eps22.shape or eps11.shape != gamma12.shape:
        raise ValueError("eps11, eps22, and gamma12 must have matching shapes.")

    average = (eps11 + eps22) / 2.0
    radius = np.sqrt(
        ((eps11 - eps22) / 2.0) ** 2
        + (gamma12 / 2.0) ** 2
    )
    return average + radius, average - radius


def _wall_element_to_node_average(wall_data, values):
    """Average row-major rectangular wall-element values to mesh nodes."""
    values = np.asarray(values, dtype=float).reshape(-1)
    nx = len(wall_data["x_coords"])
    nz = len(wall_data["z_coords"])
    expected_elements = (nx - 1) * (nz - 1)
    if values.size != expected_elements:
        raise ValueError(
            f"Wall mesh requires {expected_elements} element values, "
            f"but {values.size} were provided."
        )

    node_values = np.zeros((nz, nx), dtype=float)
    node_count = np.zeros((nz, nx), dtype=float)
    element_index = 0
    for j in range(nz - 1):
        for i in range(nx - 1):
            value = values[element_index]
            for jj, ii in ((j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i)):
                node_values[jj, ii] += value
                node_count[jj, ii] += 1.0
            element_index += 1
    return node_values / node_count


def _smooth_wall_nodal_field(wall_data, node_values, smooth_factor):
    """Bilinearly densify a rectangular wall nodal field for contour plotting."""
    x = np.asarray(wall_data["x_coords"], dtype=float)
    z = np.asarray(wall_data["z_coords"], dtype=float)
    if smooth_factor < 1:
        raise ValueError("smooth_factor must be at least 1.")

    x_smooth = np.linspace(x[0], x[-1], (len(x) - 1) * smooth_factor + 1)
    z_smooth = np.linspace(z[0], z[-1], (len(z) - 1) * smooth_factor + 1)
    values_x = np.array([np.interp(x_smooth, x, row) for row in node_values])
    values_smooth = np.array(
        [np.interp(z_smooth, z, values_x[:, i]) for i in range(len(x_smooth))]
    ).T
    X_smooth, Z_smooth = np.meshgrid(x_smooth, z_smooth)
    return X_smooth, Z_smooth, values_smooth


def plot_wall_smooth_cloud(
    wall_data,
    values,
    title,
    colorbar_label,
    minMax=False,
    min_value=None,
    max_value=None,
    levels=60,
    smooth_factor=12,
    cmap="turbo",
    figsize=(7, 6),
    save_path=None,
    dpi=600,
    show=True,
):
    """Plot a smooth contour map from rectangular wall-shell element values.

    Element values are averaged to mesh nodes and bilinearly densified for
    visualization. This smoothing must not be used to calculate extrema,
    yielded-element counts, or other quantitative indices; those calculations
    should continue to use the original element values.

    Set ``minMax=True`` with ``min_value`` and ``max_value`` to enforce a
    common color scale across specimens or loading states. ``save_path`` may
    point to PNG, TIFF, PDF, or SVG; raster formats are saved at ``dpi``.
    """
    node_values = _wall_element_to_node_average(wall_data, values)
    X_smooth, Z_smooth, values_smooth = _smooth_wall_nodal_field(
        wall_data,
        node_values,
        int(smooth_factor),
    )

    if minMax:
        if min_value is None or max_value is None:
            raise ValueError(
                "minMax=True requires both min_value and max_value."
            )
        if max_value <= min_value:
            raise ValueError("max_value must be greater than min_value.")
        contour_levels = np.linspace(min_value, max_value, int(levels) + 1)
        extend = "both"
    else:
        contour_levels = int(levels)
        extend = "neither"

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    contour = ax.contourf(
        X_smooth,
        Z_smooth,
        values_smooth,
        levels=contour_levels,
        cmap=cmap,
        extend=extend,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_title(title)
    ax.tick_params(direction="out")

    colorbar = fig.colorbar(contour, ax=ax)
    colorbar.set_label(colorbar_label)

    if save_path is not None:
        save_path = os.fspath(save_path)
        extension = os.path.splitext(save_path)[1].lower()
        save_kwargs = {"bbox_inches": "tight"}
        if extension in {".png", ".tif", ".tiff"}:
            save_kwargs["dpi"] = int(dpi)
        fig.savefig(save_path, **save_kwargs)

    if show:
        plt.show()
    return fig, ax


def plt_layershell_section(section_data):
    """
    可视化 LayeredShell 截面。
    
    参数:
    section_data: 包含截面信息的字典，格式为:
    {
        "tag": int,
        "width": float,
        "layers": [{"matTag": int, "thickness": float, "rebar": bool}, ...],
        "material_properties": {matTag: {"name": str, "color": str}, ...}
    }
    """
    layers = section_data["layers"]
    wall_width = 1000.0
    mat_props = section_data.get("material_properties", {})

    fig, ax = plt.subplots(figsize=(10, 4))
    current_y = 0
    used_tags = set()
    offset_toggle = True

    for layer in layers:
        matTag = layer["matTag"]
        thickness = layer["thickness"]
        rebar = layer.get("rebar", False)

        mat_info = mat_props.get(matTag, {})
        facecolor = mat_info.get("color", 'gray')
        mat_name = mat_info.get("name", f"Material {matTag}")
        edgecolor = facecolor if rebar else 'black'

        # 绘制矩形
        rect = plt.Rectangle(
            (0, current_y), wall_width, thickness,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linestyle='--' if rebar else '-',
            linewidth=1.5 if rebar else 0.5
        )
        ax.add_patch(rect)

        # 添加文本标签
        y_center = current_y + thickness / 2
        if rebar:
            offset = 0.002
            y_text = y_center + offset if offset_toggle else y_center - offset
            offset_toggle = not offset_toggle
        else:
            y_text = y_center

        label = f'{mat_name} ({thickness:.2f} mm)'
        ax.text(
            wall_width / 2, y_text,
            label,
            ha='center', va='center',
            fontsize=8, color='black'
        )

        used_tags.add(matTag)
        current_y += thickness

    # 设置坐标轴
    ax.set_xlim(0, wall_width)
    ax.set_ylim(0, current_y)
    ax.set_aspect('auto')
    ax.invert_yaxis()
    ax.set_xlabel("Display Width (mm)")
    ax.set_ylabel("Wall Thickness (mm)")
    section_name = section_data.get("name") or f"LayeredShell Section tag={section_data['tag']}"
    ax.set_title(f"{section_name} (tag={section_data['tag']})")
    ax.grid(True, linestyle='--', alpha=0.3)

    # 创建图例
    legend_handles = []
    for tag in sorted(list(used_tags)):
        mat_info = mat_props.get(tag, {})
        color = mat_info.get("color", "gray")
        name = mat_info.get("name", f"Material {tag}")
        legend_handles.append(Patch(facecolor=color, edgecolor='black', label=name))
        
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    plt.show()


# ============================================================
# 4. LayeredShell 截面创建函数
# ============================================================
def create_layershell(section_data, show=True):
    """
    Create a LayeredShell section and register its wall thickness.

    `section_data["width"]` and `section_data["area"]` are optional metadata.
    The shell element geometry supplies each wall component width and height.
    Set `show=False` when creating many sections without preview figures.
    """
    tag = section_data["tag"]
    layers = section_data["layers"]
    n_layers = len(layers)

    # 创建 LayeredShell 截面
    args = [tag, n_layers]
    for layer in layers:
        args.extend([layer['matTag'], layer['thickness']])
    section('LayeredShell', *args)

    thickness = section_data.get("thickness")
    if thickness is None:
        thickness = sum(float(layer["thickness"]) for layer in layers)
    register_shell_section(
        tag,
        width=section_data.get("width"),
        thickness=thickness,
        area=section_data.get("area"),
        name=section_data.get("name"),
        layers=layers,
    )
    
    if show:
        plt_layershell_section(section_data)
    
    return tag



def node_distance(node_i, node_j):
    """
    计算两个节点初始坐标之间的三维距离

    Parameters
    ----------
    node_i, node_j : int
        两个节点号

    Returns
    -------
    dict
        {
            "node_i": int,
            "node_j": int,
            "coord_i": (xi, yi, zi),
            "coord_j": (xj, yj, zj),
            "dx": float,
            "dy": float,
            "dz": float,
            "distance": float,
        }
    """
    xi, yi, zi = nodeCoord(int(node_i))
    xj, yj, zj = nodeCoord(int(node_j))

    dx = float(xj - xi)
    dy = float(yj - yi)
    dz = float(zj - zi)
    dist = math.sqrt(dx**2 + dy**2 + dz**2)

    return {
        "node_i": int(node_i),
        "node_j": int(node_j),
        "coord_i": (float(xi), float(yi), float(zi)),
        "coord_j": (float(xj), float(yj), float(zj)),
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "distance": dist,
    }



def get_component_by_name(name, frame_level=1):
    """
    按变量名获取构件字典
    适用于 notebook 里像 beam_s1_20GO、wall_s1_S11_15 这种变量
    """
    frame = inspect.currentframe()
    for _ in range(frame_level):
        frame = frame.f_back

    if name in frame.f_locals:
        obj = frame.f_locals[name]
    elif name in frame.f_globals:
        obj = frame.f_globals[name]
    else:
        raise KeyError(f"未找到名为 {name!r} 的构件变量")

    if not isinstance(obj, dict) or "added_elements" not in obj:
        raise TypeError(f"{name!r} 不是网格函数返回的构件字典")

    return obj


def show_component_info(name, frame_level=1):
    """
    打印构件基本信息
    """
    comp = get_component_by_name(name, frame_level=frame_level + 1)

    print(f"构件名: {name}")
    print(f"单元号: {comp.get('added_elements', [])}")

    if "node_tags" in comp:
        print(f"节点号: {comp['node_tags']}")
        print("节点坐标:")
        for tag, xyz in zip(comp["node_tags"], comp["global_coords"]):
            print(f"  node {tag}: {xyz}")

    elif "node_grid" in comp:
        print(f"控制节点: {comp.get('control_node')}")
        print(f"底边节点: {comp.get('bottom_nodes', [])}")
        print(f"顶边节点: {comp.get('top_nodes', [])}")
        print(f"左边节点: {comp.get('left_edge_nodes', [])}")
        print(f"右边节点: {comp.get('right_edge_nodes', [])}")


def plot_component_by_name(name, frame_level=1, show_node_tag=True):
    """
    按变量名绘制单个构件
    beam/column: 画折线
    wall: 画网格
    """
    comp = get_component_by_name(name, frame_level=frame_level + 1)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if "node_tags" in comp and "global_coords" in comp:
        coords = comp["global_coords"]
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        zs = [p[2] for p in coords]

        ax.plot(xs, ys, zs, "-o", linewidth=2)

        if show_node_tag:
            for tag, (x, y, z) in zip(comp["node_tags"], coords):
                ax.text(x, y, z, str(tag), fontsize=9)

    elif "global_coords_grid" in comp:
        grid = comp["global_coords_grid"]

        for row in grid:
            xs = [p[0] for p in row]
            ys = [p[1] for p in row]
            zs = [p[2] for p in row]
            ax.plot(xs, ys, zs, "b-", linewidth=1)

        nrow = len(grid)
        ncol = len(grid[0])
        for i in range(ncol):
            xs = [grid[j][i][0] for j in range(nrow)]
            ys = [grid[j][i][1] for j in range(nrow)]
            zs = [grid[j][i][2] for j in range(nrow)]
            ax.plot(xs, ys, zs, "b-", linewidth=1)

        if show_node_tag:
            for row_nodes, row_coords in zip(comp["node_grid"], grid):
                for tag, (x, y, z) in zip(row_nodes, row_coords):
                    ax.text(x, y, z, str(tag), fontsize=8)

    else:
        raise TypeError(f"{name!r} 不是当前支持绘制的构件类型")

    ax.set_title(name)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()







def _component_ele_centers(component_data):
    centers = []
    for ele_tag in component_data.get("added_elements", []):
        node_ids = eleNodes(ele_tag)
        node_coords = np.array([nodeCoord(nid) for nid in node_ids], dtype=float)
        if node_coords.shape[1] == 2:
            node_coords = np.hstack([node_coords, np.zeros((node_coords.shape[0], 1))])
        centers.append(node_coords.mean(axis=0))
    return centers


def _add_beam_or_column_highlight(fig, component_data, color="red", line_width=6.0):
    import pyvista as pv

    coords = np.array(component_data["global_coords"], dtype=float)
    if coords.shape[1] == 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])

    poly = pv.lines_from_points(coords, close=False)
    fig.add_mesh(poly, color=color, line_width=line_width)
    fig.add_mesh(
        pv.PolyData(coords),
        color=color,
        point_size=max(10.0, line_width * 1.5),
        render_points_as_spheres=True,
    )


def _add_wall_highlight(fig, component_data, color="red", line_width=4.0, opacity=0.20):
    import pyvista as pv

    grid = component_data["global_coords_grid"]
    nrow = len(grid)
    ncol = len(grid[0])

    points = []
    point_ids = {}
    faces = []

    for j in range(nrow):
        for i in range(ncol):
            point_ids[(j, i)] = len(points)
            points.append(grid[j][i])

    for j in range(nrow - 1):
        for i in range(ncol - 1):
            faces.extend(
                [
                    4,
                    point_ids[(j, i)],
                    point_ids[(j, i + 1)],
                    point_ids[(j + 1, i + 1)],
                    point_ids[(j + 1, i)],
                ]
            )

    surface = pv.PolyData(np.array(points, dtype=float), faces=np.array(faces))
    fig.add_mesh(surface, color=color, opacity=opacity, show_edges=False)

    for row in grid:
        coords = np.array(row, dtype=float)
        fig.add_mesh(pv.lines_from_points(coords, close=False), color=color, line_width=line_width)

    for i in range(ncol):
        coords = np.array([grid[j][i] for j in range(nrow)], dtype=float)
        fig.add_mesh(pv.lines_from_points(coords, close=False), color=color, line_width=line_width)


def highlight_component_in_model(
    component,
    show_node=False,
    show_ele=False,
    node_size=12,
    ele_size=12,
    show_local_axes=True,
    highlight_color="red",
    highlight_line_width=6,
    show_component_node_tags=False,
    show_component_ele_tags=False,
):
    """
    在整体模型中高亮显示指定构件。

    Parameters
    ----------
    component : str or dict
        构件变量名，或 beam_mesh/column_mesh/wall_mesh 返回的构件字典
    show_node, show_ele : bool
        是否显示整个模型的全部节点号、单元号
    node_size, ele_size : int
        全模型标签字体大小
    show_local_axes : bool
        是否显示整体模型局部坐标轴
    highlight_color : str
        目标构件高亮颜色
    highlight_line_width : float
        高亮线宽
    show_component_node_tags, show_component_ele_tags : bool
        是否仅对目标构件补充节点号、单元号标签
    """
    import opstool as opst

    component_data, component_name = _resolve_component_input(component, frame_level=2)

    opst.vis.pyvista.set_plot_props(notebook=False)
    fig = opst.vis.pyvista.plot_model(show_local_axes=show_local_axes)

    if show_node:
        node_tags = getNodeTags()
        coords = np.array([nodeCoord(tag) for tag in node_tags], dtype=float)
        if coords.shape[1] == 2:
            coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
        fig.add_point_labels(
            coords,
            [str(tag) for tag in node_tags],
            font_size=node_size,
            point_color="black",
        )

    if show_ele:
        ele_tags = getEleTags()
        ele_centers = []
        for ele in ele_tags:
            node_ids = eleNodes(ele)
            node_coords = np.array([nodeCoord(nid) for nid in node_ids], dtype=float)
            if node_coords.shape[1] == 2:
                node_coords = np.hstack([node_coords, np.zeros((node_coords.shape[0], 1))])
            ele_centers.append(node_coords.mean(axis=0))

        fig.add_point_labels(
            ele_centers,
            [str(tag) for tag in ele_tags],
            font_size=ele_size,
            point_color="blue",
        )

    if "global_coords" in component_data and "node_tags" in component_data:
        _add_beam_or_column_highlight(
            fig,
            component_data,
            color=highlight_color,
            line_width=float(highlight_line_width),
        )
    elif "global_coords_grid" in component_data and "node_grid" in component_data:
        _add_wall_highlight(
            fig,
            component_data,
            color=highlight_color,
            line_width=max(2.0, float(highlight_line_width) * 0.6),
            opacity=0.20,
        )
    else:
        raise TypeError("当前仅支持 beam_mesh / column_mesh / wall_mesh 返回的构件字典")

    if show_component_node_tags:
        comp_node_tags = _component_node_tags(component_data)
        comp_coords = np.array([nodeCoord(tag) for tag in comp_node_tags], dtype=float)
        if comp_coords.shape[1] == 2:
            comp_coords = np.hstack([comp_coords, np.zeros((comp_coords.shape[0], 1))])
        fig.add_point_labels(
            comp_coords,
            [str(tag) for tag in comp_node_tags],
            font_size=max(10, node_size),
            point_color=highlight_color,
            shape_opacity=0.15,
        )

    if show_component_ele_tags:
        centers = _component_ele_centers(component_data)
        fig.add_point_labels(
            centers,
            [str(tag) for tag in component_data.get("added_elements", [])],
            font_size=max(10, ele_size),
            point_color=highlight_color,
            shape_opacity=0.15,
        )

    try:
        fig.add_text(f"Highlighted: {component_name}", position="upper_left", font_size=12)
    except Exception:
        pass

    fig.show()
    return fig




def _resolve_component_input(component, frame_level=1):
    if isinstance(component, dict):
        return component, "<dict>"

    if not isinstance(component, str):
        raise TypeError("component 必须是构件变量名字字符串或构件字典")

    frame = inspect.currentframe()
    for _ in range(frame_level):
        frame = frame.f_back

    if component in frame.f_locals:
        obj = frame.f_locals[component]
    elif component in frame.f_globals:
        obj = frame.f_globals[component]
    else:
        raise KeyError(f"未找到名为 {component!r} 的构件变量")

    if not isinstance(obj, dict):
        raise TypeError(f"{component!r} 不是构件字典")

    return obj, component


def _flatten_node_grid(node_grid):
    tags = []
    seen = set()
    for row in node_grid:
        for tag in row:
            if tag not in seen:
                tags.append(tag)
                seen.add(tag)
    return tags


def _component_node_tags(component_data):
    if "node_tags" in component_data:
        return list(component_data["node_tags"])
    if "node_grid" in component_data:
        return _flatten_node_grid(component_data["node_grid"])
    return []


def _node_xyz(tag):
    xyz = nodeCoord(int(tag))
    if len(xyz) == 2:
        return np.array([float(xyz[0]), float(xyz[1]), 0.0], dtype=float)
    return np.array([float(xyz[0]), float(xyz[1]), float(xyz[2])], dtype=float)


def _coords_close(a, b, tol):
    return np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)) <= tol


def _component_kind(component_data):
    if "node_grid" in component_data and "global_coords_grid" in component_data:
        return "wall"
    if "node_tags" in component_data and "column_height" in component_data:
        return "column"
    if "node_tags" in component_data and "beam_length" in component_data:
        return "beam"
    return "unknown"


def _collect_components_from_scope(frame_level=1):
    frame = inspect.currentframe()
    for _ in range(frame_level):
        frame = frame.f_back

    components = {}
    for scope in (frame.f_globals, frame.f_locals):
        for name, obj in scope.items():
            if not isinstance(obj, dict):
                continue
            if "added_elements" not in obj:
                continue
            components[name] = obj
    return components


def component_direction_summary(component, frame_level=1):
    """
    查看单个构件的方向信息。
    """
    data, name = _resolve_component_input(component, frame_level=frame_level + 1)
    kind = _component_kind(data)

    summary = {
        "name": name,
        "kind": kind,
    }

    if kind == "wall":
        summary.update(
            {
                "plane": data.get("plane"),
                "span_dir": data.get("span_dir"),
                "wall_normal": data.get("wall_normal"),
                "anchor_point": data.get("anchor_point"),
            }
        )
    elif kind == "beam":
        summary.update(
            {
                "axis": data.get("axis"),
                "span_dir": data.get("span_dir"),
                "anchor_point": data.get("anchor_point"),
            }
        )
    elif kind == "column":
        summary.update(
            {
                "axis": "Z",
                "span_dir": "+Z",
                "anchor_point": data.get("anchor_point"),
            }
        )

    print("=== Component Direction Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary


def check_shared_nodes(component_a, component_b, frame_level=1, coord_tol=1e-6):
    """
    检查两个构件是否真正共节点，而不只是几何重合。
    """
    data_a, name_a = _resolve_component_input(component_a, frame_level=frame_level + 1)
    data_b, name_b = _resolve_component_input(component_b, frame_level=frame_level + 1)

    tags_a = _component_node_tags(data_a)
    tags_b = _component_node_tags(data_b)
    set_a = set(tags_a)
    set_b = set(tags_b)

    shared_tags = sorted(set_a & set_b)

    coords_a = {tag: _node_xyz(tag) for tag in tags_a}
    coords_b = {tag: _node_xyz(tag) for tag in tags_b}

    same_coord_different_tag = []
    for tag_a, xyz_a in coords_a.items():
        for tag_b, xyz_b in coords_b.items():
            if tag_a == tag_b:
                continue
            if _coords_close(xyz_a, xyz_b, coord_tol):
                same_coord_different_tag.append(
                    {
                        "tag_a": tag_a,
                        "tag_b": tag_b,
                        "coord": tuple(float(v) for v in xyz_a),
                    }
                )

    result = {
        "name_a": name_a,
        "name_b": name_b,
        "num_nodes_a": len(tags_a),
        "num_nodes_b": len(tags_b),
        "shared_tags": shared_tags,
        "num_shared_tags": len(shared_tags),
        "same_coord_different_tag": same_coord_different_tag,
        "num_same_coord_different_tag": len(same_coord_different_tag),
        "ok_shared": len(shared_tags) > 0,
        "warning_only_geometric_overlap": len(shared_tags) == 0 and len(same_coord_different_tag) > 0,
    }

    print("=== Shared Node Check ===")
    print(f"{name_a} nodes: {len(tags_a)}")
    print(f"{name_b} nodes: {len(tags_b)}")
    print(f"shared node tags: {len(shared_tags)}")
    if shared_tags:
        print(f"shared tags sample: {shared_tags[:10]}")
    if same_coord_different_tag:
        print("warning: 找到坐标重合但 tag 不同的节点，这说明几何上重合但没有真正连接")
        for item in same_coord_different_tag[:10]:
            print(item)

    return result


def _wall_edge_info(wall_data, edge_name):
    if edge_name == "left":
        tags = list(wall_data["left_edge_nodes"])
        coords = [row[0] for row in wall_data["global_coords_grid"]]
    elif edge_name == "right":
        tags = list(wall_data["right_edge_nodes"])
        coords = [row[-1] for row in wall_data["global_coords_grid"]]
    else:
        raise ValueError("edge_name 必须是 'left' 或 'right'")

    z_values = [float(c[2]) for c in coords]
    return {
        "tags": tags,
        "coords": coords,
        "z_values": z_values,
    }


def check_wall_corner_pair(wall_a, wall_b, frame_level=1, coord_tol=1e-6):
    """
    检查两片墙是否适合形成 L 型转角墙。
    重点检查：
    1. 是否是两片 wall_mesh 返回的墙
    2. 是否在公共边真正共节点
    3. 公共边是否沿整层高度连续
    4. plane / span_dir / wall_normal 是否清楚
    """
    data_a, name_a = _resolve_component_input(wall_a, frame_level=frame_level + 1)
    data_b, name_b = _resolve_component_input(wall_b, frame_level=frame_level + 1)

    if _component_kind(data_a) != "wall" or _component_kind(data_b) != "wall":
        raise TypeError("check_wall_corner_pair 只支持 wall_mesh 返回的墙体字典")

    edge_pairs = []
    for edge_a in ("left", "right"):
        info_a = _wall_edge_info(data_a, edge_a)
        for edge_b in ("left", "right"):
            info_b = _wall_edge_info(data_b, edge_b)

            shared_tags = sorted(set(info_a["tags"]) & set(info_b["tags"]))
            shared_z = sorted(
                {
                    round(float(nodeCoord(tag)[2]), 12)
                    for tag in shared_tags
                }
            )

            same_coord_pairs = []
            for tag_a, coord_a in zip(info_a["tags"], info_a["coords"]):
                for tag_b, coord_b in zip(info_b["tags"], info_b["coords"]):
                    if tag_a == tag_b:
                        continue
                    if _coords_close(coord_a, coord_b, coord_tol):
                        same_coord_pairs.append((tag_a, tag_b, tuple(float(v) for v in coord_a)))

            edge_pairs.append(
                {
                    "edge_a": edge_a,
                    "edge_b": edge_b,
                    "shared_tags": shared_tags,
                    "num_shared_tags": len(shared_tags),
                    "shared_z": shared_z,
                    "num_same_coord_pairs": len(same_coord_pairs),
                    "same_coord_pairs": same_coord_pairs,
                    "expected_z_count": min(len(info_a["z_values"]), len(info_b["z_values"])),
                }
            )

    best_pair = max(edge_pairs, key=lambda item: (item["num_shared_tags"], item["num_same_coord_pairs"]))

    full_height_ok = best_pair["num_shared_tags"] == best_pair["expected_z_count"]
    orthogonal_ok = data_a.get("plane") != data_b.get("plane")

    result = {
        "wall_a": name_a,
        "wall_b": name_b,
        "plane_a": data_a.get("plane"),
        "plane_b": data_b.get("plane"),
        "span_dir_a": data_a.get("span_dir"),
        "span_dir_b": data_b.get("span_dir"),
        "normal_a": data_a.get("wall_normal"),
        "normal_b": data_b.get("wall_normal"),
        "best_pair": best_pair,
        "orthogonal_ok": orthogonal_ok,
        "full_height_shared_edge_ok": full_height_ok,
        "ok": orthogonal_ok and full_height_ok and best_pair["num_same_coord_pairs"] == 0,
    }

    print("=== Wall Corner Check ===")
    print(f"{name_a}: plane={data_a.get('plane')}, span_dir={data_a.get('span_dir')}, normal={data_a.get('wall_normal')}")
    print(f"{name_b}: plane={data_b.get('plane')}, span_dir={data_b.get('span_dir')}, normal={data_b.get('wall_normal')}")
    print(f"best shared edge: {best_pair['edge_a']} <-> {best_pair['edge_b']}")
    print(f"shared tag count: {best_pair['num_shared_tags']}")
    print(f"shared z levels: {best_pair['shared_z']}")

    if not orthogonal_ok:
        print("warning: 两片墙不在不同平面上，不像标准 XOZ/YOZ 转角墙")
    if not full_height_ok:
        print("warning: 公共边没有沿全高度共节点，常见原因是 z_coords 不一致")
    if best_pair["num_same_coord_pairs"] > 0:
        print("warning: 公共边发现坐标重合但 tag 不同的节点，说明可能只是重合没有真正连接")

    return result


def scan_model_node_connectivity(frame_level=1, coord_tol=1e-6, print_limit=20):
    """
    全局扫描当前作用域中已建好的全部构件，检查节点连接问题。

    这个函数主要识别两类情况：
    1. 真正共节点：多个构件使用同一个 nodeTag
    2. 假连接/漏共节点：多个构件在相同坐标位置使用了不同 nodeTag

    Parameters
    ----------
    frame_level : int
        向上查找构件变量的层数
    coord_tol : float
        判定“坐标相同”的容差
    print_limit : int
        屏幕输出最多显示多少条问题
    """
    components = _collect_components_from_scope(frame_level=frame_level + 1)
    if not components:
        raise RuntimeError("当前作用域内未找到网格函数返回的构件字典")

    # nodeTag -> 构件名列表
    tag_to_components = {}
    # 量化坐标 -> 记录列表
    coordkey_to_records = {}

    for comp_name, comp_data in components.items():
        node_tags = _component_node_tags(comp_data)
        kind = _component_kind(comp_data)

        for tag in node_tags:
            tag_to_components.setdefault(int(tag), []).append(comp_name)

            xyz = _node_xyz(tag)
            key = tuple(int(round(float(v) / coord_tol)) for v in xyz)
            coordkey_to_records.setdefault(key, []).append(
                {
                    "component": comp_name,
                    "kind": kind,
                    "tag": int(tag),
                    "coord": tuple(float(v) for v in xyz),
                }
            )

    shared_tag_groups = []
    for tag, comp_names in tag_to_components.items():
        uniq = sorted(set(comp_names))
        if len(uniq) > 1:
            shared_tag_groups.append(
                {
                    "tag": int(tag),
                    "components": uniq,
                    "coord": tuple(float(v) for v in _node_xyz(tag)),
                }
            )

    duplicate_coord_groups = []
    for records in coordkey_to_records.values():
        tags = sorted({item["tag"] for item in records})
        comp_names = sorted({item["component"] for item in records})
        if len(tags) <= 1:
            continue

        # 再做一次实坐标复核，避免量化误差误报
        base = np.array(records[0]["coord"], dtype=float)
        if not all(_coords_close(base, item["coord"], coord_tol) for item in records[1:]):
            continue

        duplicate_coord_groups.append(
            {
                "coord": records[0]["coord"],
                "tags": tags,
                "components": comp_names,
                "records": records,
            }
        )

    duplicate_coord_groups.sort(key=lambda item: (len(item["components"]), len(item["tags"])), reverse=True)
    shared_tag_groups.sort(key=lambda item: len(item["components"]), reverse=True)

    result = {
        "num_components": len(components),
        "component_names": sorted(components.keys()),
        "num_shared_tag_groups": len(shared_tag_groups),
        "shared_tag_groups": shared_tag_groups,
        "num_duplicate_coord_groups": len(duplicate_coord_groups),
        "duplicate_coord_groups": duplicate_coord_groups,
        "ok": len(duplicate_coord_groups) == 0,
    }

    print("=== Global Node Connectivity Scan ===")
    print(f"components found: {len(components)}")
    print(f"shared-node groups (good/expected connections): {len(shared_tag_groups)}")
    print(f"duplicate-coordinate groups with different node tags (potential problems): {len(duplicate_coord_groups)}")

    if duplicate_coord_groups:
        print("\nPotential non-shared-node problems:")
        for item in duplicate_coord_groups[:print_limit]:
            print(f"coord={item['coord']}, tags={item['tags']}, components={item['components']}")
        if len(duplicate_coord_groups) > print_limit:
            print(f"... 其余 {len(duplicate_coord_groups) - print_limit} 组未显示")
    else:
        print("\n未发现“同一坐标位置使用不同节点号”的问题。")

    return result


def report_component_node_issues(frame_level=1, coord_tol=1e-6, print_limit=50):
    """
    生成按构件归类的节点问题报告。

    返回每个构件涉及了哪些“坐标重合但 tag 不同”的问题，
    便于快速定位是哪个构件最可能没有正确共节点。
    """
    scan = scan_model_node_connectivity(
        frame_level=frame_level + 1,
        coord_tol=coord_tol,
        print_limit=0,
    )

    component_issues = {}
    for issue in scan["duplicate_coord_groups"]:
        for comp in issue["components"]:
            component_issues.setdefault(comp, []).append(issue)

    sorted_items = sorted(
        component_issues.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )

    print("=== Component Node Issue Report ===")
    if not sorted_items:
        print("未发现构件层面的漏共节点问题。")
        return {
            "ok": True,
            "component_issues": {},
        }

    for comp, issues in sorted_items[:print_limit]:
        print(f"{comp}: {len(issues)} issue(s)")
        for issue in issues[:3]:
            print(f"  coord={issue['coord']}, tags={issue['tags']}, components={issue['components']}")

    if len(sorted_items) > print_limit:
        print(f"... 其余 {len(sorted_items) - print_limit} 个构件未显示")

    return {
        "ok": False,
        "component_issues": dict(sorted_items),
    }


def scan_model_near_miss_nodes(
    frame_level=1,
    near_tol=5.0,
    same_coord_tol=1e-6,
    critical_only=True,
    print_limit=30,
):
    """
    全局扫描“本来应该接上、但坐标差了一点没接上”的近邻节点问题。

    适用场景：
    - 梁柱交点
    - 梁墙交点
    - 剪力墙转角公共边
    - 墙与楼层梁、连梁交接

    Parameters
    ----------
    frame_level : int
        向上查找构件变量的层数
    near_tol : float
        认为“过近、值得怀疑”的距离阈值
    same_coord_tol : float
        认为“实际上就是同一点”的坐标容差
    critical_only : bool
        True 时只检查各构件更关键的连接节点，减少噪声
    print_limit : int
        屏幕最多显示多少条结果
    """
    components = _collect_components_from_scope(frame_level=frame_level + 1)
    if not components:
        raise RuntimeError("当前作用域内未找到网格函数返回的构件字典")

    candidates = []
    for comp_name, comp_data in components.items():
        kind = _component_kind(comp_data)
        node_tags = (
            _component_reference_nodes(comp_data)
            if critical_only
            else _component_node_tags(comp_data)
        )
        for tag in node_tags:
            candidates.append(
                {
                    "component": comp_name,
                    "kind": kind,
                    "tag": int(tag),
                    "coord": tuple(float(v) for v in _node_xyz(tag)),
                }
            )

    buckets = {}
    for item in candidates:
        key = _bucket_key(item["coord"], near_tol)
        buckets.setdefault(key, []).append(item)

    issues = []
    seen_pairs = set()

    for item in candidates:
        base_key = _bucket_key(item["coord"], near_tol)
        base_xyz = np.array(item["coord"], dtype=float)

        for neigh_key in _iter_neighbor_bucket_keys(base_key):
            for other in buckets.get(neigh_key, []):
                if other["component"] == item["component"]:
                    continue

                pair_key = tuple(
                    sorted(
                        [
                            (item["component"], item["tag"]),
                            (other["component"], other["tag"]),
                        ]
                    )
                )
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                other_xyz = np.array(other["coord"], dtype=float)
                dist = float(np.linalg.norm(base_xyz - other_xyz))

                # 完全重合的情况交给“漏共节点”检查，不在这里重复报
                if dist <= same_coord_tol:
                    continue
                if dist > near_tol:
                    continue

                issues.append(
                    {
                        "distance": dist,
                        "component_a": item["component"],
                        "kind_a": item["kind"],
                        "tag_a": item["tag"],
                        "coord_a": item["coord"],
                        "component_b": other["component"],
                        "kind_b": other["kind"],
                        "tag_b": other["tag"],
                        "coord_b": other["coord"],
                    }
                )

    issues.sort(key=lambda x: x["distance"])

    result = {
        "num_components": len(components),
        "num_candidates": len(candidates),
        "near_tol": float(near_tol),
        "critical_only": bool(critical_only),
        "num_issues": len(issues),
        "issues": issues,
        "ok": len(issues) == 0,
    }

    print("=== Global Near-Miss Node Scan ===")
    print(f"components found: {len(components)}")
    print(f"candidate nodes checked: {len(candidates)}")
    print(f"near_tol: {near_tol}")
    print(f"critical_only: {critical_only}")
    print(f"near-miss issues: {len(issues)}")

    if issues:
        print("\nPotential near-miss connection problems:")
        for item in issues[:print_limit]:
            print(
                f"d={item['distance']:.6f}, "
                f"{item['component_a']}[tag {item['tag_a']}] <-> "
                f"{item['component_b']}[tag {item['tag_b']}]"
            )
            print(f"  A: {item['coord_a']}")
            print(f"  B: {item['coord_b']}")
        if len(issues) > print_limit:
            print(f"... 其余 {len(issues) - print_limit} 条未显示")
    else:
        print("\n未发现“相距很近但没有接上”的节点问题。")

    return result


def report_near_miss_component_issues(
    frame_level=1,
    near_tol=5.0,
    same_coord_tol=1e-6,
    critical_only=True,
    print_limit=50,
):
    """
    按构件汇总近邻未连接问题。
    """
    scan = scan_model_near_miss_nodes(
        frame_level=frame_level + 1,
        near_tol=near_tol,
        same_coord_tol=same_coord_tol,
        critical_only=critical_only,
        print_limit=0,
    )

    component_issues = {}
    for issue in scan["issues"]:
        for comp_key in ("component_a", "component_b"):
            comp = issue[comp_key]
            component_issues.setdefault(comp, []).append(issue)

    sorted_items = sorted(
        component_issues.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )

    print("=== Near-Miss Component Issue Report ===")
    if not sorted_items:
        print("未发现按构件归类的近邻未连接问题。")
        return {
            "ok": True,
            "component_issues": {},
        }

    for comp, issues in sorted_items[:print_limit]:
        print(f"{comp}: {len(issues)} near-miss issue(s)")
        for issue in issues[:3]:
            if issue["component_a"] == comp:
                other_comp = issue["component_b"]
                tag = issue["tag_a"]
                other_tag = issue["tag_b"]
                coord = issue["coord_a"]
                other_coord = issue["coord_b"]
            else:
                other_comp = issue["component_a"]
                tag = issue["tag_b"]
                other_tag = issue["tag_a"]
                coord = issue["coord_b"]
                other_coord = issue["coord_a"]
            print(
                f"  d={issue['distance']:.6f}, tag {tag} near {other_comp}[tag {other_tag}]"
            )
            print(f"    this : {coord}")
            print(f"    other: {other_coord}")

    if len(sorted_items) > print_limit:
        print(f"... 其余 {len(sorted_items) - print_limit} 个构件未显示")

    return {
        "ok": False,
        "component_issues": dict(sorted_items),
    }

def _bucket_key(coord, tol):
    return tuple(int(round(float(v) / tol)) for v in coord)


def _iter_neighbor_bucket_keys(base_key):
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield (base_key[0] + dx, base_key[1] + dy, base_key[2] + dz)



def create_rigid_diaphragm_at_floor(
    z,
    floor_nodes=None,
    master_xy=None,
    master_tag=None,
    tol=1e-6,
    diaphragm_chunk_size=200,
):
    """
    在指定楼层 z 建立刚性楼板主节点，并施加 rigidDiaphragm 约束

    Parameters
    ----------
    z : float
        楼层标高
    floor_nodes : list[int], optional
        该层节点；不传则自动按 z 搜索
    master_xy : tuple(float, float), optional
        主节点平面坐标；不传则取该层节点几何中心
    master_tag : int, optional
        主节点编号；不传则自动生成
    tol : float
        标高搜索容差

    Returns
    -------
    int
        刚性楼板主节点号
    """
    z = float(z)

    if floor_nodes is None:
        floor_nodes = [
            nd for nd in ops.getNodeTags()
            if abs(ops.nodeCoord(nd)[2] - z) < tol
        ]
    else:
        floor_nodes = [int(nd) for nd in floor_nodes]

    if len(floor_nodes) == 0:
        raise ValueError(f"在 z={z} 处没有找到楼层节点")

    # 默认取该层节点几何中心
    if master_xy is None:
        xy = np.array(
            [[ops.nodeCoord(nd)[0], ops.nodeCoord(nd)[1]] for nd in floor_nodes],
            dtype=float
        )
        x_master = float(xy[:, 0].mean())
        y_master = float(xy[:, 1].mean())
    else:
        x_master = float(master_xy[0])
        y_master = float(master_xy[1])

    if master_tag is None:
        master_tag = max(ops.getNodeTags()) + 1
    master_tag = int(master_tag)

    ops.node(master_tag, x_master, y_master, z)

    # 只保留 UX, UY, RZ
    ops.fix(master_tag, 0, 0, 1, 1, 1, 0)

    slave_nodes = [nd for nd in floor_nodes if nd != master_tag]
    if len(slave_nodes) == 0:
        raise ValueError("该层没有可作为从节点的楼层节点")

    # 楼板位于 XY 平面，法向为 Z。大模型中一次传入过多节点可能导致
    # OpenSeesPy 底层不稳定，因此分块施加同一个 retained node 的约束。
    diaphragm_chunk_size = int(diaphragm_chunk_size)
    if diaphragm_chunk_size <= 0:
        ops.rigidDiaphragm(3, master_tag, *slave_nodes)
    else:
        for start in range(0, len(slave_nodes), diaphragm_chunk_size):
            ops.rigidDiaphragm(3, master_tag, *slave_nodes[start:start + diaphragm_chunk_size])

    print(f"Rigid diaphragm created at z = {z:.3f}")
    print(f"master node = {master_tag}")
    print(f"num slave nodes = {len(slave_nodes)}")
    print(f"master coord = ({x_master:.3f}, {y_master:.3f}, {z:.3f})")

    return master_tag


def _floor_nodes_at_z(z, floor_nodes=None, tol=1e-6):
    z = float(z)
    if floor_nodes is None:
        return [
            int(nd) for nd in ops.getNodeTags()
            if abs(ops.nodeCoord(nd)[2] - z) < tol
        ]
    return [int(nd) for nd in floor_nodes]


def _unique_node_tags_preserve_order(node_tags):
    result = []
    seen = set()
    for tag in node_tags:
        tag = int(tag)
        if tag in seen:
            continue
        result.append(tag)
        seen.add(tag)
    return result


def _nodes_from_component_at_z(component_data, z, tol=1e-6):
    kind = _component_kind(component_data)
    if kind == "column":
        tags = component_data.get("node_tags", [])
    elif kind == "beam":
        tags = [
            component_data.get("start_node"),
            component_data.get("end_node"),
        ]
    elif kind == "wall":
        tags = []
        for key in ("left_edge_nodes", "right_edge_nodes"):
            tags.extend(component_data.get(key, []))
        tags.extend(component_data.get("corner_nodes", {}).values())
    else:
        tags = []

    selected = []
    for tag in _unique_node_tags_preserve_order(tag for tag in tags if tag is not None):
        try:
            if abs(float(ops.nodeCoord(tag)[2]) - float(z)) <= tol:
                selected.append(tag)
        except Exception:
            continue
    return selected


def _diaphragm_nodes_at_z(
    z,
    floor_nodes=None,
    tol=1e-6,
    node_selection="frame_wall_boundary",
    components=None,
):
    if floor_nodes is not None:
        return _floor_nodes_at_z(z, floor_nodes=floor_nodes, tol=tol)

    if node_selection in (None, "all", "all_floor"):
        return _floor_nodes_at_z(z, tol=tol)

    if node_selection not in {"frame_wall_boundary", "frame"}:
        raise ValueError(
            "node_selection must be 'frame_wall_boundary', 'frame', or 'all_floor'"
        )

    selected = []
    for component_data in (components or {}).values():
        kind = _component_kind(component_data)
        if node_selection == "frame" and kind == "wall":
            continue
        selected.extend(_nodes_from_component_at_z(component_data, z, tol=tol))

    selected = _unique_node_tags_preserve_order(selected)
    if not selected:
        return _floor_nodes_at_z(z, tol=tol)
    return selected


def apply_floor_gravity_load(
    z,
    P_gravity,
    floor_nodes=None,
    exclude_nodes=None,
    tol=1e-6,
    pattern_tag=1,
    ts_tag=1,
    create_pattern=True,
    print_summary=True,
):
    """Apply vertical gravity load to floor nodes only."""
    z = float(z)
    P_gravity = float(P_gravity)
    floor_nodes = _floor_nodes_at_z(z, floor_nodes=floor_nodes, tol=tol)
    exclude_nodes = set(int(nd) for nd in (exclude_nodes or []))
    load_nodes = [nd for nd in floor_nodes if nd not in exclude_nodes]

    if len(load_nodes) == 0:
        raise ValueError(f"z={z} floor has no nodes for gravity load")

    if create_pattern:
        ops.timeSeries("Linear", ts_tag)
        ops.pattern("Plain", pattern_tag, ts_tag)

    p_each = P_gravity / len(load_nodes)
    for nd in load_nodes:
        ops.load(nd, 0.0, 0.0, -p_each, 0.0, 0.0, 0.0)

    if print_summary:
        print(f"Floor gravity load assigned at z = {z:.3f}")
        print(f"num load nodes = {len(load_nodes)}")
        print(f"gravity load each node = {p_each:.6f}")

    return {
        "z": z,
        "floor_nodes": floor_nodes,
        "load_nodes": load_nodes,
        "excluded_nodes": sorted(exclude_nodes),
        "P_gravity": P_gravity,
        "load_each_node": p_each,
    }


def assign_floor_mass_to_master(
    z,
    master_node,
    W_seismic,
    floor_nodes=None,
    g=9800.0,
    tol=1e-6,
    print_summary=True,
):
    """Assign floor translational mass and Jz to an existing diaphragm master node."""
    z = float(z)
    master_node = int(master_node)
    W_seismic = float(W_seismic)
    g = float(g)
    floor_nodes = _floor_nodes_at_z(z, floor_nodes=floor_nodes, tol=tol)

    if master_node not in ops.getNodeTags():
        raise ValueError(f"master_node={master_node} does not exist")

    master_z = ops.nodeCoord(master_node)[2]
    if abs(master_z - z) > tol:
        raise ValueError(
            f"master_node={master_node} z={master_z}, but target floor z={z}"
        )

    slave_nodes = [nd for nd in floor_nodes if nd != master_node]
    if len(slave_nodes) == 0:
        raise ValueError(f"z={z} floor has no slave nodes for mass assignment")

    m_total = W_seismic / g
    m_each = m_total / len(slave_nodes)

    x0, y0, _ = ops.nodeCoord(master_node)
    Jz = 0.0
    for nd in slave_nodes:
        x, y, _ = ops.nodeCoord(nd)
        dx = float(x - x0)
        dy = float(y - y0)
        Jz += m_each * (dx**2 + dy**2)

    ops.mass(master_node, m_total, m_total, 0.0, 0.0, 0.0, Jz)

    if print_summary:
        print(f"Floor mass assigned at z = {z:.3f}")
        print(f"master node = {master_node}")
        print(f"num slave nodes = {len(slave_nodes)}")
        print(f"total seismic mass = {m_total:.6f}")
        print(f"floor Jz = {Jz:.6f}")

    return {
        "z": z,
        "master_node": master_node,
        "slave_nodes": slave_nodes,
        "W_seismic": W_seismic,
        "m_total": m_total,
        "Jz": Jz,
    }


def assign_floor_mass_to_nodes(
    z,
    W_seismic,
    mass_nodes=None,
    master_node=None,
    g=9800.0,
    tol=1e-6,
    clear_master_mass=True,
    clear_nodes=None,
    mass_distribution="diaphragm_nodes",
    print_summary=True,
):
    """Assign floor translational mass evenly to selected floor nodes."""
    z = float(z)
    W_seismic = float(W_seismic)
    g = float(g)

    if mass_nodes is None:
        mass_nodes = _floor_nodes_at_z(z, tol=tol)
    else:
        mass_nodes = _floor_nodes_at_z(z, floor_nodes=mass_nodes, tol=tol)

    existing_nodes = set(int(nd) for nd in ops.getNodeTags())
    mass_nodes = [
        int(nd) for nd in _unique_node_tags_preserve_order(mass_nodes)
        if int(nd) in existing_nodes
    ]

    if master_node is not None:
        master_node = int(master_node)
        mass_nodes = [nd for nd in mass_nodes if nd != master_node]
        if master_node not in existing_nodes:
            raise ValueError(f"master_node={master_node} does not exist")
        master_z = ops.nodeCoord(master_node)[2]
        if abs(master_z - z) > tol:
            raise ValueError(
                f"master_node={master_node} z={master_z}, but target floor z={z}"
            )

    checked_nodes = []
    for nd in mass_nodes:
        if abs(ops.nodeCoord(nd)[2] - z) <= tol:
            checked_nodes.append(nd)
    mass_nodes = checked_nodes

    if len(mass_nodes) == 0:
        raise ValueError(f"z={z} floor has no nodes for mass assignment")

    if clear_nodes is None:
        clear_nodes = list(mass_nodes)
    else:
        clear_nodes = _floor_nodes_at_z(z, floor_nodes=clear_nodes, tol=tol)
    if clear_master_mass and master_node is not None:
        clear_nodes = list(clear_nodes) + [master_node]
    for nd in _unique_node_tags_preserve_order(clear_nodes):
        nd = int(nd)
        if nd in existing_nodes and abs(ops.nodeCoord(nd)[2] - z) <= tol:
            ops.mass(nd, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    m_total = W_seismic / g
    m_each = m_total / len(mass_nodes)

    for nd in mass_nodes:
        ops.mass(nd, m_each, m_each, 0.0, 0.0, 0.0, 0.0)

    Jz = 0.0
    if master_node is not None:
        x0, y0, _ = ops.nodeCoord(master_node)
        for nd in mass_nodes:
            x, y, _ = ops.nodeCoord(nd)
            dx = float(x - x0)
            dy = float(y - y0)
            Jz += m_each * (dx**2 + dy**2)

    if print_summary:
        print(f"Floor mass assigned at z = {z:.3f}")
        if master_node is not None:
            print(f"master node = {master_node}")
        print(f"mass distribution = {mass_distribution}")
        print(f"num mass nodes = {len(mass_nodes)}")
        print(f"mass each node = {m_each:.6f}")
        print(f"total seismic mass = {m_total:.6f}")
        if master_node is not None:
            print(f"equivalent floor Jz = {Jz:.6f}")

    return {
        "z": z,
        "master_node": master_node,
        "mass_nodes": mass_nodes,
        "slave_nodes": mass_nodes,
        "num_mass_nodes": len(mass_nodes),
        "W_seismic": W_seismic,
        "m_total": m_total,
        "m_each": m_each,
        "Jz": Jz,
        "mass_distribution": mass_distribution,
    }


def apply_floor_gravity_and_mass(
    z,
    master_node,
    P_gravity,
    W_seismic=None,
    floor_nodes=None,
    g=9800.0,
    tol=1e-6,
    pattern_tag=1,
    ts_tag=1,
    create_pattern=True,
):
    """
    对指定楼层施加：
    1. 重力荷载：分配到真实楼层节点
    2. 动力质量：集中到刚性楼板主节点

    Parameters
    ----------
    z : float
        楼层标高
    master_node : int
        刚性楼板主节点
    P_gravity : float
        该层总重力荷载
    W_seismic : float, optional
        该层总地震重量；若不传则默认等于 P_gravity
    floor_nodes : list[int], optional
        该层节点；不传则自动按 z 搜索
    g : float
        重力加速度
    tol : float
        标高搜索容差
    pattern_tag : int
        荷载工况号
    ts_tag : int
        timeSeries 号
    create_pattern : bool
        是否自动创建 Linear + Plain 荷载工况

    Returns
    -------
    dict
    """
    z = float(z)
    master_node = int(master_node)
    P_gravity = float(P_gravity)

    if W_seismic is None:
        W_seismic = P_gravity
    W_seismic = float(W_seismic)

    g = float(g)

    if floor_nodes is None:
        floor_nodes = [
            nd for nd in ops.getNodeTags()
            if abs(ops.nodeCoord(nd)[2] - z) < tol
        ]
    else:
        floor_nodes = [int(nd) for nd in floor_nodes]

    if master_node not in ops.getNodeTags():
        raise ValueError(f"master_node={master_node} 不存在")

    master_z = ops.nodeCoord(master_node)[2]
    if abs(master_z - z) > tol:
        raise ValueError(
            f"master_node={master_node} 的标高是 {master_z}，与你输入的 z={z} 不一致"
        )

    slave_nodes = [nd for nd in floor_nodes if nd != master_node]
    if len(slave_nodes) == 0:
        raise ValueError("该层没有可施加荷载的从节点")

    # 1. 重力荷载加到真实楼层节点，不加到 master
    if create_pattern:
        ops.timeSeries("Linear", ts_tag)
        ops.pattern("Plain", pattern_tag, ts_tag)

    p_each = P_gravity / len(slave_nodes)
    for nd in slave_nodes:
        ops.load(nd, 0.0, 0.0, -p_each, 0.0, 0.0, 0.0)

    # 2. 动力质量集中到 master
    m_total = W_seismic / g
    m_each = m_total / len(slave_nodes)

    x0, y0, _ = ops.nodeCoord(master_node)
    Jz = 0.0
    for nd in slave_nodes:
        x, y, _ = ops.nodeCoord(nd)
        dx = float(x - x0)
        dy = float(y - y0)
        Jz += m_each * (dx**2 + dy**2)

    # 水平地震常用：MX, MY, JZ
    ops.mass(master_node, m_total, m_total, 0.0, 0.0, 0.0, Jz)

    print(f"Floor gravity/mass assigned at z = {z:.3f}")
    print(f"num slave nodes = {len(slave_nodes)}")
    print(f"gravity load each node = {p_each:.6f}")
    print(f"total seismic mass = {m_total:.6f}")
    print(f"floor Jz = {Jz:.6f}")

    return {
        "z": z,
        "master_node": master_node,
        "slave_nodes": slave_nodes,
        "P_gravity": P_gravity,
        "W_seismic": W_seismic,
        "m_total": m_total,
        "Jz": Jz,
        "load_each_node": p_each,
    }


RC_UNIT_WEIGHT_N_PER_MM3 = 25.0e-6


def _story_number_from_component_name(name):
    import re

    match = re.search(r"_s(\d+)_", str(name))
    if match is None:
        return None
    return int(match.group(1))


def _frame_section_area_from_component(component_data):
    integration_tag = component_data.get("integration_tag")
    if integration_tag is None:
        raise ValueError("梁/柱构件字典中没有 integration_tag，无法自动反查截面面积")

    integration_tag = int(integration_tag)
    integration_info = FRAME_INTEGRATION_REGISTRY.get(integration_tag)
    if integration_info is None:
        raise ValueError(
            f"integration_tag={integration_tag} 没有注册到 FRAME_INTEGRATION_REGISTRY，"
            "请用 ufp.define_frame_integration(...) 定义梁柱积分"
        )

    sec_tag = int(integration_info["sec_tag"])
    section_info = FRAME_SECTION_REGISTRY.get(sec_tag)
    if section_info is None or section_info.get("area") is None:
        raise ValueError(
            f"sec_tag={sec_tag} 没有注册截面面积，"
            "请在截面定义后调用 ufp.register_frame_section_from_props(...)"
        )

    return float(section_info["area"]), sec_tag


def _wall_thickness_from_component(component_data):
    sec_tag = component_data.get("sec_tag_wall")
    if sec_tag is None:
        raise ValueError("墙构件字典中没有 sec_tag_wall，无法自动反查墙厚")

    sec_tag = int(sec_tag)
    section_info = SHELL_SECTION_REGISTRY.get(sec_tag)
    if section_info is None or section_info.get("thickness") is None:
        raise ValueError(
            f"sec_tag_wall={sec_tag} 没有注册墙厚，"
            "请在 section_data 中提供 thickness，或确认 ufp.create_layershell(...) 已运行"
        )

    return float(section_info["thickness"]), sec_tag


def _component_volume_mm3(component_data, kind, column_area=None, beam_area=None, wall_thickness=None):
    if kind == "column":
        sec_tag = None
        if column_area is None:
            column_area, sec_tag = _frame_section_area_from_component(component_data)
        return float(column_area) * float(component_data["column_height"])

    if kind == "beam":
        sec_tag = None
        if beam_area is None:
            beam_area, sec_tag = _frame_section_area_from_component(component_data)
        return float(beam_area) * float(component_data["beam_length"])

    if kind == "wall":
        sec_tag = None
        if wall_thickness is None:
            wall_thickness, sec_tag = _wall_thickness_from_component(component_data)
        x_coords = np.asarray(component_data["x_coords"], dtype=float)
        z_coords = np.asarray(component_data["z_coords"], dtype=float)
        wall_width = float(x_coords.max() - x_coords.min())
        wall_height = float(z_coords.max() - z_coords.min())
        return wall_width * wall_height * float(wall_thickness)

    return 0.0


def summarize_story_self_weight(
    story_heights,
    column_area=None,
    beam_area=None,
    wall_thickness=None,
    unit_weight=RC_UNIT_WEIGHT_N_PER_MM3,
    include_kinds=("column", "beam", "wall"),
    story_range=None,
    extra_floor_weights=None,
    frame_level=1,
    print_summary=True,
):
    """
    按构件变量名里的 _s1_/_s2_... 自动汇总每层构件自重。

    约定
    ----
    1. 模型单位为 N, mm, s。
    2. unit_weight 为容重，默认 25 kN/m^3 = 25e-6 N/mm^3。
    3. 第 k 层构件自重默认归到第 k 层顶楼板。
    """
    story_heights = [float(h) for h in story_heights]
    if len(story_heights) == 0:
        raise ValueError("story_heights 不能为空")

    include_kinds = set(include_kinds)
    extra_floor_weights = extra_floor_weights or {}
    components = _collect_components_from_scope(frame_level=frame_level + 1)

    if story_range is None:
        story_ids = sorted(
            {
                story
                for name in components
                for story in [_story_number_from_component_name(name)]
                if story is not None
            }
        )
    else:
        story_ids = [int(story) for story in story_range]

    story_ids = [story for story in story_ids if 1 <= story <= len(story_heights)]
    story_top_z = np.cumsum(story_heights)

    story_data = {
        story: {
            "story": story,
            "z": float(story_top_z[story - 1]),
            "components": [],
            "volume_by_kind": {kind: 0.0 for kind in ("column", "beam", "wall")},
            "weight_by_kind": {kind: 0.0 for kind in ("column", "beam", "wall")},
            "extra_weight": float(extra_floor_weights.get(story, 0.0)),
        }
        for story in story_ids
    }

    for name, component_data in components.items():
        story = _story_number_from_component_name(name)
        if story not in story_data:
            continue

        kind = _component_kind(component_data)
        if kind not in include_kinds:
            continue

        volume = _component_volume_mm3(
            component_data,
            kind,
            column_area=column_area,
            beam_area=beam_area,
            wall_thickness=wall_thickness,
        )
        weight = volume * float(unit_weight)

        item = story_data[story]
        item["components"].append(
            {
                "name": name,
                "kind": kind,
                "volume": volume,
                "weight": weight,
            }
        )
        item["volume_by_kind"][kind] += volume
        item["weight_by_kind"][kind] += weight

    for item in story_data.values():
        item["component_weight"] = float(sum(item["weight_by_kind"].values()))
        item["total_weight"] = item["component_weight"] + item["extra_weight"]
        item["num_components"] = len(item["components"])

    if print_summary:
        print("=== Story Self Weight Summary ===")
        print(f"unit_weight = {unit_weight:.6e} N/mm^3")
        for story in story_ids:
            item = story_data[story]
            print(
                f"story {story:2d}, z = {item['z']:.3f}, "
                f"components = {item['num_components']:4d}, "
                f"W = {item['total_weight']:.3f} N"
            )
            print(
                "  column = {column:.3f}, beam = {beam:.3f}, wall = {wall:.3f}, extra = {extra:.3f}".format(
                    column=item["weight_by_kind"]["column"],
                    beam=item["weight_by_kind"]["beam"],
                    wall=item["weight_by_kind"]["wall"],
                    extra=item["extra_weight"],
                )
            )

    return {
        "story_heights": story_heights,
        "unit_weight": float(unit_weight),
        "story_data": story_data,
        "total_weight": float(sum(item["total_weight"] for item in story_data.values())),
    }


def _story_data_from_summary(story_weight_summary):
    if "story_data" in story_weight_summary:
        return story_weight_summary["story_data"]
    return story_weight_summary


def _diaphragm_results_from_info(diaphragm_info):
    if diaphragm_info is None:
        return {}
    if "diaphragm_results" in diaphragm_info:
        return diaphragm_info["diaphragm_results"]
    if "floor_results" in diaphragm_info:
        return diaphragm_info["floor_results"]
    return diaphragm_info


def apply_story_self_weight_gravity_loads(
    story_weight_summary,
    story_range=None,
    diaphragm_info=None,
    pattern_tag=1,
    ts_tag=1,
    create_pattern=False,
    tol=1e-6,
    print_summary=True,
):
    """Apply only gravity loads from a story self-weight summary."""
    story_data = _story_data_from_summary(story_weight_summary)
    diaphragm_results = _diaphragm_results_from_info(diaphragm_info)

    if create_pattern:
        ops.timeSeries("Linear", ts_tag)
        ops.pattern("Plain", pattern_tag, ts_tag)

    if story_range is None:
        story_ids = sorted(story_data)
    else:
        story_ids = [int(story) for story in story_range if int(story) in story_data]

    gravity_results = {}
    for story in story_ids:
        item = story_data[story]
        dia_item = diaphragm_results.get(story, {})
        master_node = dia_item.get("master_node")
        floor_nodes = dia_item.get("floor_nodes")
        exclude_nodes = [] if master_node is None else [master_node]

        result = apply_floor_gravity_load(
            z=item["z"],
            P_gravity=item["total_weight"],
            floor_nodes=floor_nodes,
            exclude_nodes=exclude_nodes,
            tol=tol,
            pattern_tag=pattern_tag,
            ts_tag=ts_tag,
            create_pattern=False,
            print_summary=print_summary,
        )
        result["story"] = story
        result["component_weight"] = item["component_weight"]
        result["extra_weight"] = item["extra_weight"]
        result["num_components"] = item["num_components"]
        gravity_results[story] = result

    return {
        "gravity_load_results": gravity_results,
        "total_gravity_load": float(
            sum(result["P_gravity"] for result in gravity_results.values())
        ),
    }


def create_story_rigid_diaphragms(
    story_weight_summary,
    story_range=None,
    master_xy_by_story=None,
    master_tag_start=None,
    diaphragm_chunk_size=200,
    tol=1e-6,
    node_selection="frame_wall_boundary",
    frame_level=1,
):
    """Create only rigid diaphragm master nodes and constraints."""
    story_data = _story_data_from_summary(story_weight_summary)
    master_xy_by_story = master_xy_by_story or {}
    components = None
    if node_selection not in (None, "all", "all_floor"):
        components = _collect_components_from_scope(frame_level=frame_level + 1)

    if story_range is None:
        story_ids = sorted(story_data)
    else:
        story_ids = [int(story) for story in story_range if int(story) in story_data]

    diaphragm_results = {}
    next_master_tag = master_tag_start
    for story in story_ids:
        item = story_data[story]
        z = item["z"]
        all_floor_nodes = _floor_nodes_at_z(z, tol=tol)
        floor_nodes = _diaphragm_nodes_at_z(
            z,
            tol=tol,
            node_selection=node_selection,
            components=components,
        )
        master_tag = next_master_tag
        master_node = create_rigid_diaphragm_at_floor(
            z=z,
            floor_nodes=floor_nodes,
            master_xy=master_xy_by_story.get(story),
            master_tag=master_tag,
            tol=tol,
            diaphragm_chunk_size=diaphragm_chunk_size,
        )
        if next_master_tag is not None:
            next_master_tag = int(master_node) + 1

        x_master, y_master, z_master = ops.nodeCoord(master_node)
        diaphragm_results[story] = {
            "story": story,
            "z": z,
            "master_node": int(master_node),
            "master_coord": (float(x_master), float(y_master), float(z_master)),
            "floor_nodes": floor_nodes,
            "slave_nodes": floor_nodes,
            "num_slave_nodes": len(floor_nodes),
            "all_floor_node_count": len(all_floor_nodes),
            "node_selection": node_selection,
        }

    return {
        "diaphragm_results": diaphragm_results,
    }


def assign_story_mass_to_diaphragm_masters(
    story_weight_summary,
    diaphragm_info,
    story_range=None,
    g=9800.0,
    tol=1e-6,
    mass_distribution="diaphragm_nodes",
    clear_existing_mass=True,
    print_summary=True,
):
    """Assign floor masses for existing diaphragm constraints."""
    story_data = _story_data_from_summary(story_weight_summary)
    diaphragm_results = _diaphragm_results_from_info(diaphragm_info)
    mass_distribution = str(mass_distribution or "diaphragm_nodes")

    if story_range is None:
        story_ids = sorted(story_data)
    else:
        story_ids = [int(story) for story in story_range if int(story) in story_data]

    mass_results = {}
    for story in story_ids:
        if story not in diaphragm_results:
            raise KeyError(f"story {story} has no diaphragm result")
        item = story_data[story]
        dia_item = diaphragm_results[story]
        if mass_distribution == "master":
            result = assign_floor_mass_to_master(
                z=item["z"],
                master_node=dia_item["master_node"],
                W_seismic=item["total_weight"],
                floor_nodes=dia_item.get("floor_nodes"),
                g=g,
                tol=tol,
                print_summary=print_summary,
            )
            result["mass_distribution"] = "master"
        else:
            if mass_distribution in ("diaphragm_nodes", "slave_nodes", "selected_nodes"):
                mass_nodes = dia_item.get("slave_nodes") or dia_item.get("floor_nodes")
                mass_label = "diaphragm_nodes"
            elif mass_distribution in ("all_floor_nodes", "all_floor", "all"):
                mass_nodes = _floor_nodes_at_z(item["z"], tol=tol)
                mass_label = "all_floor_nodes"
            else:
                raise ValueError(
                    "mass_distribution must be 'diaphragm_nodes', "
                    "'all_floor_nodes', or 'master'"
                )
            result = assign_floor_mass_to_nodes(
                z=item["z"],
                master_node=dia_item["master_node"],
                W_seismic=item["total_weight"],
                mass_nodes=mass_nodes,
                g=g,
                tol=tol,
                clear_nodes=(
                    _floor_nodes_at_z(item["z"], tol=tol)
                    if clear_existing_mass
                    else None
                ),
                mass_distribution=mass_label,
                print_summary=print_summary,
            )
        result["story"] = story
        result["component_weight"] = item["component_weight"]
        result["extra_weight"] = item["extra_weight"]
        result["num_components"] = item["num_components"]
        mass_results[story] = result

    return {
        "mass_results": mass_results,
        "total_mass": float(sum(result["m_total"] for result in mass_results.values())),
    }


def apply_story_self_weight_gravity_and_mass(
    story_heights,
    column_area=None,
    beam_area=None,
    wall_thickness=None,
    unit_weight=RC_UNIT_WEIGHT_N_PER_MM3,
    include_kinds=("column", "beam", "wall"),
    story_range=None,
    extra_floor_weights=None,
    g=9800.0,
    pattern_tag=1,
    ts_tag=1,
    create_pattern=False,
    tol=1e-6,
    master_xy_by_story=None,
    master_tag_start=None,
    diaphragm_chunk_size=200,
    frame_level=1,
    mass_distribution="diaphragm_nodes",
    print_summary=True,
):
    """
    自动计算每层构件自重，创建刚性楼板，并施加楼层重力荷载和动力质量。
    """
    summary = summarize_story_self_weight(
        story_heights=story_heights,
        column_area=column_area,
        beam_area=beam_area,
        wall_thickness=wall_thickness,
        unit_weight=unit_weight,
        include_kinds=include_kinds,
        story_range=story_range,
        extra_floor_weights=extra_floor_weights,
        frame_level=frame_level + 1,
        print_summary=print_summary,
    )

    if create_pattern:
        ops.timeSeries("Linear", ts_tag)
        ops.pattern("Plain", pattern_tag, ts_tag)

    diaphragm_info = create_story_rigid_diaphragms(
        summary,
        story_range=story_range,
        master_xy_by_story=master_xy_by_story,
        master_tag_start=master_tag_start,
        diaphragm_chunk_size=diaphragm_chunk_size,
        tol=tol,
        frame_level=frame_level + 1,
    )
    gravity_info = apply_story_self_weight_gravity_loads(
        summary,
        story_range=story_range,
        diaphragm_info=diaphragm_info,
        pattern_tag=pattern_tag,
        ts_tag=ts_tag,
        create_pattern=False,
        tol=tol,
        print_summary=print_summary,
    )
    mass_info = assign_story_mass_to_diaphragm_masters(
        summary,
        diaphragm_info,
        story_range=story_range,
        g=g,
        tol=tol,
        mass_distribution=mass_distribution,
        print_summary=print_summary,
    )

    floor_results = {}
    for story in sorted(summary["story_data"]):
        if story_range is not None and story not in [int(s) for s in story_range]:
            continue
        floor_result = {}
        floor_result.update(diaphragm_info["diaphragm_results"].get(story, {}))
        floor_result.update(gravity_info["gravity_load_results"].get(story, {}))
        floor_result.update(mass_info["mass_results"].get(story, {}))
        floor_results[story] = floor_result

    summary["diaphragm_results"] = diaphragm_info["diaphragm_results"]
    summary["gravity_load_results"] = gravity_info["gravity_load_results"]
    summary["mass_results"] = mass_info["mass_results"]
    summary["floor_results"] = floor_results
    summary["total_mass"] = mass_info["total_mass"]
    return summary


def save_current_eigen_data_for_opstool(
    odb_tag="modal_current",
    mode_tag=1,
    interpolate_beam=False,
):
    """
    Save eigen data for opstool plotting without calling ops.eigen again.

    Run eigen(...) or set_rayleigh_damping(...) before this function. The
    function keeps opstool's ODB format and plotting workflow, but temporarily
    patches opstool's modal-property getter so save_eigen_data reuses the
    current OpenSees eigenvectors instead of recomputing them.
    """
    import numpy as _np
    import xarray as _xr
    import opstool.post.eigen_data as _eigen_data

    mode_tag = int(mode_tag)

    def _get_current_modal_properties(mode_tag_inner=1, solver="-genBandArpack"):
        modal_props = ops.modalProperties("-return")

        attrs_names = ["domainSize", "totalMass", "totalFreeMass", "centerOfMass"]
        attrs = {name: tuple(modal_props[name]) for name in attrs_names}
        for key, value in attrs.items():
            if key == "domainSize":
                value = [int(v) for v in value]
            if len(value) == 1:
                value = value[0]
            attrs[key] = value

        column_names = [name for name in modal_props if name not in attrs_names]
        columns = [modal_props[name] for name in column_names]
        data = _np.vstack(columns).transpose()[:mode_tag_inner]
        return _xr.DataArray(
            data,
            coords={
                "modeTags": _np.arange(1, mode_tag_inner + 1),
                "Properties": column_names,
            },
            dims=("modeTags", "Properties"),
            attrs=attrs,
            name="ModalProps",
        )

    old_get_modal_properties = _eigen_data._get_modal_properties
    _eigen_data._get_modal_properties = _get_current_modal_properties
    try:
        _eigen_data.save_eigen_data(
            odb_tag=odb_tag,
            mode_tag=mode_tag,
            solver="-genBandArpack",
            interpolate_beam=interpolate_beam,
        )
    finally:
        _eigen_data._get_modal_properties = old_get_modal_properties

    return {
        "odb_tag": odb_tag,
        "mode_tag": mode_tag,
        "interpolate_beam": bool(interpolate_beam),
    }





def _apply_retrofit_table_to_generated_files(model_root, selected_table=None, show_sections=False):
    """Patch generated table-driven model files using root/?????.xlsx."""
    from pathlib import Path as _Path
    import re as _re

    root = _Path(model_root).resolve()
    retrofit_table = root / "\u7f6e\u6362\u7ba1\u7406\u8868.xlsx"
    model_table = root / "\u5efa\u6a21\u7ba1\u7406\u8868.xlsx"
    section_file = root / "table_sections.py"
    if not retrofit_table.exists() or not model_table.exists() or not section_file.exists():
        return {"enabled": False, "reason": "missing retrofit/model/generated file"}

    try:
        import openpyxl as _openpyxl
    except Exception as exc:
        raise RuntimeError("openpyxl is required to read retrofit workbook") from exc

    def _is_yes(value):
        return str(value).strip() in {"\u662f", "yes", "YES", "1", "True", "true"}

    def _to_float(value, default=None):
        if value is None or value == "":
            return default
        return float(value)

    def _to_int(value, default=None):
        if value is None or value == "":
            return default
        return int(float(value))

    def _split_seq(value):
        if value is None:
            return []
        return [part.strip() for part in str(value).replace("\uff0c", ";").replace(",", ";").split(";") if part.strip()]

    def _story_no(sheet_name):
        m = _re.search(r"(\d+)", str(sheet_name))
        return int(m.group(1)) if m else None

    retrofit_wb = _openpyxl.load_workbook(retrofit_table, data_only=True)
    model_wb = _openpyxl.load_workbook(model_table, data_only=True)

    # ????? fixed columns, 0-based positions after row 4 header:
    # A type, B component name, C b, D h, E longitudinal bars, F stirrup,
    # G cover concrete, H core concrete, I steel, K cover, L mesh, N column num_ele.
    model_rows = {}
    for ws in model_wb.worksheets:
        sn = _story_no(ws.title)
        if sn is None or ws.max_row < 5:
            continue
        for row in ws.iter_rows(min_row=5, values_only=True):
            name = row[1] if len(row) > 1 else None
            if not name:
                continue
            model_rows[str(name)] = {
                "story": sn,
                "component_type": row[0] if len(row) > 0 else None,
                "name": str(name),
                "b": row[2] if len(row) > 2 else None,
                "h": row[3] if len(row) > 3 else None,
                "rebar_info": row[4] if len(row) > 4 else None,
                "stirrup_info": row[5] if len(row) > 5 else None,
                "cover_mat": row[6] if len(row) > 6 else None,
                "core_mat": row[7] if len(row) > 7 else None,
                "rebar_mat": row[8] if len(row) > 8 else None,
                "cover": row[10] if len(row) > 10 else None,
                "mesh_size": row[11] if len(row) > 11 else None,
                "column_num_ele": row[13] if len(row) > 13 else None,
            }

    column_rules = []
    wall_rules = []
    for ws in retrofit_wb.worksheets:
        sn = _story_no(ws.title)
        if sn is None or ws.max_row < 5:
            continue
        for row in ws.iter_rows(min_row=5, values_only=True):
            if not row:
                continue
            comp_type = str(row[0]).strip() if len(row) > 0 and row[0] is not None else ""
            name = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""
            if not name:
                continue
            if name not in model_rows:
                raise ValueError(f"{ws.title}: retrofit component not found in model table: {name}")
            base = model_rows[name]
            if comp_type == "\u67f1":
                segment_coords = [_to_float(v) for v in _split_seq(row[4] if len(row) > 4 else None)]
                thickness = _to_float(row[5] if len(row) > 5 else None, None)
                b0 = _to_float(base.get("b"), None)
                h0 = _to_float(base.get("h"), None)
                z_start = _to_float(row[2] if len(row) > 2 else None, 0.0)
                z_end = _to_float(row[3] if len(row) > 3 else None, None)
                full_section_override = (
                    thickness is None
                    or (b0 is not None and h0 is not None and 2.0 * float(thickness) >= min(float(b0), float(h0)))
                )
                if len(segment_coords) >= 2:
                    z_min = min(segment_coords)
                    z_max = max(segment_coords)
                    whole_height = (
                        z_start <= z_min + 1.0e-9
                        and (z_end is None or z_end >= z_max - 1.0e-9)
                    )
                else:
                    whole_height = True
                if (not full_section_override or not whole_height) and len(segment_coords) < 2:
                    raise ValueError(f"{ws.title} {name}: column segment coordinate sequence is required")
                if full_section_override:
                    column_mode = "full_section" if whole_height else "full_section_segmented"
                else:
                    column_mode = "outer_ring"
                column_rules.append({
                    "story": sn,
                    "name": name,
                    "z_start": z_start,
                    "z_end": z_end,
                    "segment_coords": segment_coords,
                    "thickness": thickness,
                    "cover_mat": str(row[6]).strip() if len(row) > 6 and row[6] is not None else None,
                    "conf_mat": str(row[7]).strip() if len(row) > 7 and row[7] is not None else None,
                    "mode": column_mode,
                    "base": base,
                })
            elif comp_type == "\u526a\u529b\u5899":
                thickness_seq = [_to_float(v) for v in _split_seq(row[8] if len(row) > 8 else None)]
                mat_seq = _split_seq(row[9] if len(row) > 9 else None)
                if len(thickness_seq) != len(mat_seq) or not thickness_seq:
                    raise ValueError(f"{ws.title} {name}: wall layer thickness/material sequence mismatch")
                wall_rules.append({"story": sn, "name": name, "thickness_seq": thickness_seq, "mat_seq": mat_seq, "base": base})

    if not column_rules and not wall_rules:
        return {"enabled": False, "reason": "no enabled retrofit rows"}

    for rule in column_rules:
        if rule["cover_mat"] is None or rule["conf_mat"] is None:
            raise ValueError(f"{rule['name']}: incomplete column retrofit rule")
        if rule["mode"] == "outer_ring" and rule["thickness"] is None:
            raise ValueError(f"{rule['name']}: incomplete column retrofit rule")

    def _fmt_num(value):
        value = float(value)
        return str(int(value)) if abs(value - int(value)) < 1e-9 else (f"{value:g}".replace(".", "p"))

    def _slug(value):
        value = str(value)
        value = _re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")
        return value or "X"

    def _mat_label(value):
        label = str(value).strip()
        for prefix in ("TagFrame", "TagWall", "TagSteel", "Tag"):
            if label.startswith(prefix):
                label = label[len(prefix):]
                break
        for suffix in ("Plane", "Plate"):
            if label.endswith(suffix):
                label = label[:-len(suffix)]
        return _slug(label)

    def _retrofit_col_section_name(b, h, thickness, cover_mat, ring_mat, core_mat, rebar_mat, bar_count, bar_dia, stirrup_dia):
        return _slug(
            f"RetrofitCol_{_fmt_num(b)}x{_fmt_num(h)}_t{_fmt_num(thickness)}_"
            f"Cvr{_mat_label(cover_mat)}_Ring{_mat_label(ring_mat)}_Core{_mat_label(core_mat)}_"
            f"Rb{_mat_label(rebar_mat)}_{int(bar_count)}D{_fmt_num(bar_dia)}_D{_fmt_num(stirrup_dia)}"
        )

    def _override_col_section_name(b, h, cover_mat, core_mat, rebar_mat, bar_count, bar_dia, stirrup_dia):
        return _slug(
            f"DefectCol_{_fmt_num(b)}x{_fmt_num(h)}_"
            f"Cvr{_mat_label(cover_mat)}_Core{_mat_label(core_mat)}_"
            f"Rb{_mat_label(rebar_mat)}_{int(bar_count)}D{_fmt_num(bar_dia)}_D{_fmt_num(stirrup_dia)}"
        )

    def _retrofit_wall_section_name(thickness_seq, mat_seq, h_bar_dia=8.0, h_bar_spacing=200.0, v_bar_dia=10.0, v_bar_spacing=200.0):
        layer_part = "_".join(f"{_fmt_num(t)}{_mat_label(m)}" for t, m in zip(thickness_seq, mat_seq))
        total_t = sum(float(t) for t in thickness_seq)
        return _slug(
            f"RetrofitWall_t{_fmt_num(total_t)}_{layer_part}_"
            f"V{_fmt_num(v_bar_dia)}_{_fmt_num(v_bar_spacing)}_H{_fmt_num(h_bar_dia)}_{_fmt_num(h_bar_spacing)}"
        )

    section_text = section_file.read_text(encoding="utf-8")
    show_default = "True" if show_sections else "False"
    if "TABLE_SECTION_SHOW =" not in section_text:
        section_text = section_text.replace("import numpy as np\n", f"import numpy as np\n\nTABLE_SECTION_SHOW = bool(globals().get('TABLE_SECTION_SHOW', {show_default}))\n", 1)
    section_text = section_text.replace("show=False", "show=TABLE_SECTION_SHOW")

    retrofit_section_code = []
    retrofit_section_code.append("\n        # -------------------- Retrofit generated sections --------------------")
    retrofit_section_code.append("        def _create_full_override_column_section(sec_tag, b, h, cover, mesh, cover_mat, core_mat, rebar_mat, bar_count, bar_dia, stirrup_dia, sec_name=None, show=False):")
    retrofit_section_code.append("            _, props = ufp.build_rect_section(sec_tag=sec_tag, b=b, h=h, sec_name=(sec_name or f'Defect_Rect_{sec_tag}'), perimeter_bars=(bar_count, bar_dia), cover=cover, stirrup_dia=stirrup_dia, mesh_size_cover=mesh, mesh_size_core=mesh, cover_mat_tag=cover_mat, core_mat_tag=core_mat, rebar_mat_tag=rebar_mat, display_results=False, show=show, register=False)")
    retrofit_section_code.append("            ufp.register_frame_section_from_props(sec_tag, props, name=(sec_name or f'Defect_Rect_{sec_tag}'))")
    retrofit_section_code.append("            return sec_tag")
    retrofit_section_code.append("        def _create_retrofit_column_section(sec_tag, b, h, cover, retrofit_t, retrofit_cover_t, mesh, cover_mat, ring_mat, core_mat, rebar_mat, bar_count, bar_dia, stirrup_dia, sec_name=None, GJ=1e13, show=False):")
    retrofit_section_code.append("            return ufp.build_rect_retrofit_section(sec_tag=sec_tag, b=b, h=h, sec_name=(sec_name or f'Retrofit_Rect_{sec_tag}'), retrofit_t=retrofit_t, retrofit_cover_t=retrofit_cover_t, perimeter_bars=(bar_count, bar_dia), cover=cover, stirrup_dia=stirrup_dia, mesh_size_cover=mesh, mesh_size_ring=mesh, mesh_size_core=mesh, GJ=GJ, cover_mat_tag=cover_mat, ring_mat_tag=ring_mat, core_mat_tag=core_mat, rebar_mat_tag=rebar_mat, display_results=False, show=show, register=True)")
    retrofit_section_code.append("        def _create_retrofit_wall_layered_section(tag, name, concrete_layers, h_bar_dia, h_bar_spacing, v_bar_dia, v_bar_spacing, concrete_names=None):")
    retrofit_section_code.append("            h_t = ufp.rebar_equivalent_layer_thickness(h_bar_dia, h_bar_spacing) if h_bar_dia and h_bar_spacing else 0.0")
    retrofit_section_code.append("            v_t = ufp.rebar_equivalent_layer_thickness(v_bar_dia, v_bar_spacing) if v_bar_dia and v_bar_spacing else 0.0")
    retrofit_section_code.append("            layers = [{'matTag': concrete_layers[0][1], 'thickness': concrete_layers[0][0]}]")
    retrofit_section_code.append("            if h_t: layers.append({'matTag': TagreinfH, 'thickness': h_t, 'rebar': True})")
    retrofit_section_code.append("            if v_t: layers.append({'matTag': TagreinfV, 'thickness': v_t, 'rebar': True})")
    retrofit_section_code.append("            for t, mat in concrete_layers[1:-1]: layers.append({'matTag': mat, 'thickness': t})")
    retrofit_section_code.append("            if v_t: layers.append({'matTag': TagreinfV, 'thickness': v_t, 'rebar': True})")
    retrofit_section_code.append("            if h_t: layers.append({'matTag': TagreinfH, 'thickness': h_t, 'rebar': True})")
    retrofit_section_code.append("            layers.append({'matTag': concrete_layers[-1][1], 'thickness': concrete_layers[-1][0]})")
    retrofit_section_code.append("            concrete_names = list(concrete_names or [])")
    retrofit_section_code.append("            props = {TagreinfH: {'name': 'HRB400 Horizontal PlateRebar', 'color': '#d62728'}, TagreinfV: {'name': 'HRB400 Vertical PlateRebar', 'color': '#3327d6'}}")
    retrofit_section_code.append("            for i, (_, mat) in enumerate(concrete_layers): props[mat] = {'name': (concrete_names[i] if i < len(concrete_names) else str(mat)), 'color': '#cccccc'}")
    retrofit_section_code.append("            return ufp.create_layershell({'tag': tag, 'name': name, 'thickness': sum(t for t, _ in concrete_layers), 'material_properties': props, 'layers': layers}, show=TABLE_SECTION_SHOW)")

    col_int_map = {}
    col_sec_map = {}
    col_section_by_key = {}
    col_sec_start = 51000
    col_int_start = 61000
    for rule in column_rules:
        base = rule["base"]
        name = rule["name"]
        rebar_info = str(base.get("rebar_info") or "")
        stirrup_info = str(base.get("stirrup_info") or "")
        m = _re.search(r"(\d+)\D+(\d+(?:\.\d+)?)", rebar_info)
        bar_count = int(m.group(1)) if m else 12
        bar_dia = float(m.group(2)) if m else 16.0
        ms = _re.search(r"(\d+(?:\.\d+)?)", stirrup_info)
        stirrup_dia = float(ms.group(1)) if ms else 8.0
        b = _to_float(base.get("b")); h = _to_float(base.get("h"))
        cover = _to_float(base.get("cover"), 25.0)
        mesh = _to_float(base.get("mesh_size"), 50.0)
        core_mat = str(base.get("core_mat") or "TagFrameC30Conf45")
        rebar_mat = str(base.get("rebar_mat") or "TagSteelHRB400")
        if rule.get("mode") in ("full_section", "full_section_segmented"):
            section_name = _override_col_section_name(b, h, rule["cover_mat"], rule["conf_mat"], rebar_mat, bar_count, bar_dia, stirrup_dia)
            key = (
                "full_section", float(b), float(h), float(cover), float(mesh),
                str(rule["cover_mat"]), str(rule["conf_mat"]), str(rebar_mat),
                int(bar_count), float(bar_dia), float(stirrup_dia), 5,
            )
        else:
            retrofit_cover_t = cover
            section_name = _retrofit_col_section_name(b, h, rule["thickness"], rule["cover_mat"], rule["conf_mat"], core_mat, rebar_mat, bar_count, bar_dia, stirrup_dia)
            key = (
                "outer_ring", float(b), float(h), float(cover), float(rule["thickness"]), float(retrofit_cover_t),
                float(mesh), str(rule["cover_mat"]), str(rule["conf_mat"]), str(core_mat), str(rebar_mat),
                int(bar_count), float(bar_dia), float(stirrup_dia), 5,
            )
        if key not in col_section_by_key:
            index = len(col_section_by_key)
            sec_tag = col_sec_start + index
            int_tag = col_int_start + index
            col_section_by_key[key] = (sec_tag, int_tag, section_name)
            retrofit_section_code.append(f"        {section_name}_SecTag = {sec_tag}")
            retrofit_section_code.append(f"        {section_name}_IntTag = {int_tag}")
            if rule.get("mode") in ("full_section", "full_section_segmented"):
                retrofit_section_code.append(f"        _create_full_override_column_section({section_name}_SecTag, b={b:g}, h={h:g}, cover={cover:g}, mesh={mesh:g}, cover_mat={rule['cover_mat']}, core_mat={rule['conf_mat']}, rebar_mat={rebar_mat}, bar_count={bar_count}, bar_dia={bar_dia:g}, stirrup_dia={stirrup_dia:g}, sec_name='{section_name}', show=TABLE_SECTION_SHOW)")
            else:
                retrofit_section_code.append(f"        _create_retrofit_column_section({section_name}_SecTag, b={b:g}, h={h:g}, cover={cover:g}, retrofit_t={rule['thickness']:g}, retrofit_cover_t={retrofit_cover_t:g}, mesh={mesh:g}, cover_mat={rule['cover_mat']}, ring_mat={rule['conf_mat']}, core_mat={core_mat}, rebar_mat={rebar_mat}, bar_count={bar_count}, bar_dia={bar_dia:g}, stirrup_dia={stirrup_dia:g}, sec_name='{section_name}', show=TABLE_SECTION_SHOW)")
            retrofit_section_code.append(f"        ufp.define_frame_integration('Lobatto', {section_name}_IntTag, {section_name}_SecTag, 5)")
        sec_tag, int_tag, _section_name = col_section_by_key[key]
        col_int_map[name] = int_tag
        col_sec_map[name] = sec_tag

    wall_tag_map = {}
    wall_section_by_key = {}
    wall_tag_start = 71000
    for rule in wall_rules:
        key = (tuple(rule["thickness_seq"]), tuple(rule["mat_seq"]))
        if key not in wall_section_by_key:
            tag = wall_tag_start + len(wall_section_by_key)
            section_name = _retrofit_wall_section_name(rule["thickness_seq"], rule["mat_seq"])
            wall_section_by_key[key] = (tag, section_name)
        wall_tag_map[rule["name"]] = wall_section_by_key[key][0]
    for key, (tag, section_name) in wall_section_by_key.items():
        thickness_seq, mat_seq = key
        pairs = ", ".join(f"({float(t):g}, {m})" for t, m in zip(thickness_seq, mat_seq))
        retrofit_section_code.append(f"        {section_name}_SecTag = {tag}")
        concrete_names = ", ".join(repr(_mat_label(m)) for m in mat_seq)
        retrofit_section_code.append(f"        _create_retrofit_wall_layered_section({section_name}_SecTag, '{section_name}', [{pairs}], h_bar_dia=8.0, h_bar_spacing=200.0, v_bar_dia=10.0, v_bar_spacing=200.0, concrete_names=[{concrete_names}])")

    full_column_int_map = {name: col_int_map[name] for name in col_int_map if next((r for r in column_rules if r["name"] == name), {}).get("mode") == "full_section"}
    segmented_column_int_map = {name: col_int_map[name] for name in col_int_map if name not in full_column_int_map}
    segmented_column_rules = [r for r in column_rules if r.get("mode") != "full_section"]
    if full_column_int_map:
        retrofit_section_code.append(f"        FRAME_INTEGRATION_BY_COMPONENT.update({repr(full_column_int_map)})")
    retrofit_section_code.append(f"        RETROFIT_COLUMN_INTEGRATION_BY_COMPONENT = {repr(segmented_column_int_map)}")
    retrofit_section_code.append(f"        RETROFIT_COLUMN_RULES_BY_COMPONENT = {repr({r['name']: {'z_start': r['z_start'], 'z_end': r['z_end'], 'segment_coords': r['segment_coords']} for r in segmented_column_rules})}")
    retrofit_section_code.append(f"        WALL_SECTION_ARGS_BY_COMPONENT.update({repr({name: {'zone_type': 'center_only', 'sec_tag_center': tag} for name, tag in wall_tag_map.items()})})")
    retrofit_section_code.append("        # -------------------- End retrofit generated sections --------------------\n")
    retrofit_block = "\n".join(retrofit_section_code)

    marker = "\n        TABLE_SECTIONS_READY = True"
    if marker not in section_text:
        raise RuntimeError("Cannot find TABLE_SECTIONS_READY marker in table_sections.py")
    section_text = section_text.replace(marker, retrofit_block + marker, 1)
    section_file.write_text(section_text, encoding="utf-8")

    for rule in column_rules:
        if rule.get("mode") == "full_section":
            continue
        story_file = root / f"model_story_{rule['story']:02d}.py"
        if not story_file.exists():
            raise FileNotFoundError(story_file)
        story_text = story_file.read_text(encoding="utf-8")
        name = rule["name"]
        pattern = _re.compile(rf"{_re.escape(name)}\s*=\s*ufp\.column_mesh\(.*?\)\s*\nnode_map\s*=\s*{_re.escape(name)}\[\"node_map\"\]", _re.S)
        match = pattern.search(story_text)
        if not match:
            raise RuntimeError(f"Cannot find generated column block for {name} in {story_file.name}")
        block = match.group(0)
        def _grab(label):
            m = _re.search(label + r"\s*=\s*(.*?),\s*\n", block)
            if not m:
                raise RuntimeError(f"Cannot parse {label} for {name}")
            return m.group(1).strip()
        column_height = _grab("column_height")
        anchor_point = _grab("anchor_point")
        replacement = f"""{name} = ufp.retrofit_column_mesh(
    column_height={column_height},
    anchor_point={anchor_point},
    base_num_ele=FRAME_MESH_ARGS_BY_COMPONENT['{name}'].get('num_ele', 1),
    base_integration_tag=FRAME_INTEGRATION_BY_COMPONENT['{name}'],
    retrofit_integration_tag=RETROFIT_COLUMN_INTEGRATION_BY_COMPONENT['{name}'],
    z_start=RETROFIT_COLUMN_RULES_BY_COMPONENT['{name}']['z_start'],
    z_end=RETROFIT_COLUMN_RULES_BY_COMPONENT['{name}']['z_end'],
    segment_coords=RETROFIT_COLUMN_RULES_BY_COMPONENT['{name}']['segment_coords'],
    transf_tag=Tagtransf_col,
    ele_type='forceBeamColumn',
    node_map=node_map,
)
node_map = {name}[\"node_map\"]"""
        story_text = story_text[:match.start()] + replacement + story_text[match.end():]
        story_file.write_text(story_text, encoding="utf-8")

    columns_by_story = {}
    walls_by_story = {}
    for rule in column_rules:
        columns_by_story[str(rule["story"])] = columns_by_story.get(str(rule["story"]), 0) + 1
    for rule in wall_rules:
        walls_by_story[str(rule["story"])] = walls_by_story.get(str(rule["story"]), 0) + 1
    full_column_names = [rule["name"] for rule in column_rules if rule.get("mode") == "full_section"]
    segmented_column_names = [rule["name"] for rule in column_rules if rule.get("mode") != "full_section"]
    return {
        "enabled": True,
        "columns": len(column_rules),
        "full_section_columns": len(full_column_names),
        "segmented_columns": len(segmented_column_names),
        "walls": len(wall_rules),
        "columns_by_story": columns_by_story,
        "walls_by_story": walls_by_story,
        "column_section_tags": col_sec_map,
        "column_integration_tags": col_int_map,
        "full_section_column_names": full_column_names,
        "segmented_column_names": segmented_column_names,
        "unique_column_section_tags": sorted(set(col_sec_map.values())),
        "unique_column_integration_tags": sorted(set(col_int_map.values())),
        "wall_section_tags": wall_tag_map,
        "unique_wall_section_tags": sorted(set(wall_tag_map.values())),
    }

def apply_model_table(root=None, table=None, show_sections=False):
    """Regenerate table-driven model files from the workbook in model root."""
    import importlib.util as _importlib_util
    import sys as _sys
    from pathlib import Path as _Path

    model_root = _Path(root).expanduser().resolve() if root is not None else _Path.cwd().resolve()
    package_root = _Path(__file__).resolve().parent
    generator_candidates = [
        model_root / "model_table_generator.py",
        package_root / "model_table_generator.py",
    ]
    generator_path = next((path for path in generator_candidates if path.exists()), None)

    if generator_path is None:
        searched = ", ".join(str(path) for path in generator_candidates)
        raise FileNotFoundError(f"Cannot find model_table_generator.py. Searched: {searched}")

    module_name = "_ufp_model_table_generator"
    spec = _importlib_util.spec_from_file_location(module_name, generator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model table generator from {generator_path}")

    module = _importlib_util.module_from_spec(spec)
    _sys.modules[module_name] = module
    spec.loader.exec_module(module)

    section_table = model_root / "\u622a\u9762\u7ba1\u7406\u8868.xlsx"
    model_table = model_root / "\u5efa\u6a21\u7ba1\u7406\u8868.xlsx"
    if table is not None:
        selected_table = _Path(table).expanduser()
        if not selected_table.is_absolute():
            selected_table = model_root / selected_table
        selected_table = selected_table.resolve()
    else:
        selected_table = model_table if model_table.exists() else section_table

    module.ROOT = model_root
    module.SECTION_TABLE = section_table
    module.MODEL_TABLE = model_table
    module.TABLE = selected_table
    module.TABLE_DRIVES_MODEL = selected_table.resolve() == model_table.resolve()
    module.SECTION_FILE = model_root / "table_sections.py"
    module.MAIN_NOTEBOOK = model_root / "Main.ipynb"

    result = module.main()
    retrofit_result = _apply_retrofit_table_to_generated_files(model_root, selected_table, show_sections=show_sections)
    if isinstance(result, dict):
        result["retrofit"] = retrofit_result
    if isinstance(retrofit_result, dict) and retrofit_result.get("enabled"):
        import json as _json
        print(_json.dumps({"retrofit": retrofit_result}, ensure_ascii=False, indent=2))
    return result


def set_rayleigh_damping(
    direction="Y",
    zeta=0.05,
    num_modes=6,
    use_initial_stiffness=True,
    eigen_solver=None,
    allow_mode_reduction=True,
    target_periods=None,
):
    """Robust Rayleigh damping setup with eigen fallback and manual-period mode."""
    import numpy as _np

    direction = str(direction).upper()
    ratio_key = {
        "X": "partiMassRatiosMX",
        "Y": "partiMassRatiosMY",
        "Z": "partiMassRatiosMZ",
    }[direction]

    def _apply_rayleigh(alpha_m, beta):
        if use_initial_stiffness:
            rayleigh(alpha_m, 0.0, beta, 0.0)
            beta_name = "betaKinit"
        else:
            rayleigh(alpha_m, beta, 0.0, 0.0)
            beta_name = "betaK"
        return beta_name

    if target_periods is not None:
        if len(target_periods) != 2:
            raise ValueError("target_periods must contain exactly two periods")
        periods = _np.array(target_periods, dtype=float)
        if _np.any(periods <= 0.0):
            raise ValueError("target_periods must be positive")
        omega = 2.0 * _np.pi / periods
        w1, w2 = omega
        alpha_m = 2.0 * zeta * w1 * w2 / (w1 + w2)
        beta = 2.0 * zeta / (w1 + w2)
        beta_name = _apply_rayleigh(alpha_m, beta)
        print(f"{direction} Rayleigh damping set from target periods:")
        print(f"T1 = {periods[0]:.6f} s, T2 = {periods[1]:.6f} s")
        print(f"zeta = {zeta:.3f}")
        print(f"alphaM = {alpha_m:.6e}")
        print(f"{beta_name} = {beta:.6e}")
        return {
            "direction": direction,
            "zeta": float(zeta),
            "target_periods": tuple(float(x) for x in periods),
            "omega": tuple(float(x) for x in omega),
            "alphaM": float(alpha_m),
            beta_name: float(beta),
            "eigen_solver": None,
            "manual_periods": True,
        }

    if eigen_solver is None:
        solver_candidates = [None, "-genBandArpack", "-fullGenLapack"]
    elif isinstance(eigen_solver, (list, tuple)):
        solver_candidates = list(eigen_solver)
    else:
        solver_candidates = [eigen_solver]

    max_modes = max(2, int(num_modes))
    mode_candidates = (
        range(max_modes, 1, -1)
        if allow_mode_reduction
        else [max_modes]
    )

    lam = None
    used_solver = None
    used_modes = None
    errors = []
    for solver in solver_candidates:
        for n_modes in mode_candidates:
            try:
                if solver in (None, "default"):
                    values = eigen(int(n_modes))
                    solver_name = "default"
                else:
                    values = eigen(str(solver), int(n_modes))
                    solver_name = str(solver)
                trial = _np.array(values, dtype=float)
                trial = trial[_np.isfinite(trial) & (trial > 0.0)]
                if len(trial) >= 2:
                    lam = trial
                    used_solver = solver_name
                    used_modes = int(n_modes)
                    break
                errors.append((solver_name, int(n_modes), "less than two positive eigenvalues"))
            except Exception as err:
                errors.append((str(solver or "default"), int(n_modes), repr(err)))
        if lam is not None:
            break

    if lam is None:
        detail = "; ".join(
            f"{solver} {n_modes} modes: {err}"
            for solver, n_modes, err in errors
        )
        raise RuntimeError(
            "OpenSees eigen analysis failed. For a quick diagnostic, try "
            "set_rayleigh_damping(..., eigen_solver='-fullGenLapack', num_modes=2). "
            "If eigen still fails, use target_periods=(T1, T2) to set damping "
            f"without modal analysis. Attempts: {detail}"
        )

    omega_all = _np.sqrt(lam)
    periods_all = 2.0 * _np.pi / omega_all

    ratios = None
    try:
        modal = modalProperties("-return")
        ratios = _np.array(modal[ratio_key], dtype=float)[:len(omega_all)]
    except Exception:
        ratios = None

    if ratios is not None and len(ratios) >= 2 and _np.any(_np.abs(ratios) > 0.0):
        idx = _np.argsort(_np.abs(ratios))[::-1][:2]
        idx = _np.sort(idx)
    else:
        idx = _np.array([0, 1], dtype=int)
        ratios = _np.full(len(omega_all), _np.nan)

    modes = idx + 1
    w1, w2 = omega_all[idx]
    T1, T2 = periods_all[idx]
    r1, r2 = ratios[idx]

    alpha_m = 2.0 * zeta * w1 * w2 / (w1 + w2)
    beta = 2.0 * zeta / (w1 + w2)
    beta_name = _apply_rayleigh(alpha_m, beta)

    print(f"{direction} Rayleigh damping set:")
    print(f"eigen solver = {used_solver}, modes requested = {used_modes}")
    print(f"mode {int(modes[0])}: T = {T1:.6f} s, mass ratio = {r1}")
    print(f"mode {int(modes[1])}: T = {T2:.6f} s, mass ratio = {r2}")
    print(f"zeta = {zeta:.3f}")
    print(f"alphaM = {alpha_m:.6e}")
    print(f"{beta_name} = {beta:.6e}")

    return {
        "direction": direction,
        "zeta": float(zeta),
        "modes": tuple(int(x) for x in modes),
        "periods": (float(T1), float(T2)),
        "omega": (float(w1), float(w2)),
        "mass_ratios": (float(r1), float(r2)),
        "alphaM": float(alpha_m),
        beta_name: float(beta),
        "eigen_solver": used_solver,
        "num_modes_used": used_modes,
        "manual_periods": False,
    }


def _run_modal_eigen(num_modes=15, eigen_solver=None, allow_mode_reduction=True):
    """Run OpenSees eigen analysis with fallback solvers."""
    import numpy as _np
    if eigen_solver is None:
        solver_candidates = [None, "-genBandArpack", "-fullGenLapack"]
    elif isinstance(eigen_solver, (list, tuple)):
        solver_candidates = list(eigen_solver)
    else:
        solver_candidates = [eigen_solver]
    max_modes = max(1, int(num_modes))
    mode_candidates = range(max_modes, 0, -1) if allow_mode_reduction else [max_modes]
    errors = []
    for solver in solver_candidates:
        for n_modes in mode_candidates:
            try:
                if solver in (None, "default"):
                    values = eigen(int(n_modes))
                    solver_name = "default"
                else:
                    values = eigen(str(solver), int(n_modes))
                    solver_name = str(solver)
                lam = _np.array(values, dtype=float)
                lam = lam[_np.isfinite(lam) & (lam > 0.0)]
                if lam.size:
                    return lam, solver_name, int(n_modes)
                errors.append((solver_name, int(n_modes), "no positive eigenvalues"))
            except Exception as err:
                errors.append((str(solver or "default"), int(n_modes), repr(err)))
    detail = "; ".join(f"{solver} {n_modes} modes: {err}" for solver, n_modes, err in errors)
    raise RuntimeError(f"OpenSees eigen analysis failed. Attempts: {detail}")


def analyze_modal_participation(num_modes=15, eigen_solver=None, allow_mode_reduction=True, print_summary=True):
    """Analyze modal periods and X/Y/RZ participation mass ratios."""
    import numpy as _np
    lam, used_solver, used_modes = _run_modal_eigen(num_modes, eigen_solver, allow_mode_reduction)
    omega = _np.sqrt(lam)
    periods = 2.0 * _np.pi / omega
    freq = 1.0 / periods
    n_actual = len(periods)
    modal_props = modalProperties("-return")
    mx = _np.asarray(modal_props.get("partiMassRatiosMX", _np.zeros(n_actual)), dtype=float)[:n_actual]
    my = _np.asarray(modal_props.get("partiMassRatiosMY", _np.zeros(n_actual)), dtype=float)[:n_actual]
    rmz = _np.asarray(modal_props.get("partiMassRatiosRMZ", _np.zeros(n_actual)), dtype=float)[:n_actual]
    has_rmz = bool(_np.any(_np.isfinite(rmz) & (_np.abs(rmz) > 0.0)))
    rows = []
    for i in range(n_actual):
        candidates = {"X translation": abs(mx[i]), "Y translation": abs(my[i])}
        if has_rmz:
            candidates["torsion"] = abs(rmz[i])
        main_type = max(candidates, key=candidates.get)
        rows.append({
            "mode": int(i + 1),
            "T": float(periods[i]),
            "freq": float(freq[i]),
            "omega": float(omega[i]),
            "MX": float(mx[i]),
            "MY": float(my[i]),
            "RMZ": float(rmz[i]) if has_rmz else float("nan"),
            "type": main_type,
        })
    x_idx = int(_np.argmax(_np.abs(mx)))
    y_idx = int(_np.argmax(_np.abs(my)))
    x_mode = rows[x_idx]
    y_mode = rows[y_idx]
    torsion_mode = rows[int(_np.argmax(_np.abs(rmz)))] if has_rmz else None
    first_mode = rows[0]
    if first_mode["type"] == "X translation":
        suggested = "X"
    elif first_mode["type"] == "Y translation":
        suggested = "Y"
    else:
        suggested = "X" if x_mode["T"] >= y_mode["T"] else "Y"
    horizontal = _np.abs(mx) + _np.abs(my)
    damping_indices = []
    for idx in (x_idx, y_idx):
        if idx not in damping_indices:
            damping_indices.append(idx)
    if len(damping_indices) < 2:
        for idx in _np.argsort(horizontal)[::-1]:
            idx = int(idx)
            if idx not in damping_indices:
                damping_indices.append(idx)
            if len(damping_indices) == 2:
                break
    damping_indices = sorted(damping_indices[:2])
    result = {
        "rows": rows,
        "lam": tuple(float(x) for x in lam),
        "periods": tuple(float(x) for x in periods),
        "omega": tuple(float(x) for x in omega),
        "freq": tuple(float(x) for x in freq),
        "MX": tuple(float(x) for x in mx),
        "MY": tuple(float(x) for x in my),
        "RMZ": tuple(float(x) for x in rmz),
        "has_rmz": has_rmz,
        "x_main_mode": x_mode,
        "y_main_mode": y_mode,
        "torsion_main_mode": torsion_mode,
        "first_mode_type": first_mode["type"],
        "suggested_main_wave_direction": suggested,
        "rayleigh_target_modes": tuple(int(i + 1) for i in damping_indices),
        "rayleigh_target_indices": tuple(int(i) for i in damping_indices),
        "eigen_solver": used_solver,
        "num_modes_used": used_modes,
    }
    if print_summary:
        print_modal_participation_summary(result)
    return result


def print_modal_participation_summary(modal_info):
    """Print a compact modal participation table."""
    rows = modal_info["rows"]
    has_rmz = modal_info.get("has_rmz", False)
    print("mode |   T(s) | freq(Hz) |    MX |    MY |   RMZ | main type")
    print("-----|--------|----------|-------|-------|-------|----------")
    for row in rows:
        rmz_text = f"{row['RMZ']:6.3f}" if has_rmz else "   N/A"
        print(
            f"{row['mode']:4d} | {row['T']:6.3f} | {row['freq']:8.3f} | "
            f"{row['MX']:5.3f} | {row['MY']:5.3f} | {rmz_text} | {row['type']}"
        )
    x_mode = modal_info["x_main_mode"]
    y_mode = modal_info["y_main_mode"]
    torsion_mode = modal_info.get("torsion_main_mode")
    print("\nModal summary:")
    print(f"X main translational mode: mode {x_mode['mode']}, T = {x_mode['T']:.4f} s, MX = {x_mode['MX']:.4f}")
    print(f"Y main translational mode: mode {y_mode['mode']}, T = {y_mode['T']:.4f} s, MY = {y_mode['MY']:.4f}")
    if torsion_mode is not None:
        print(f"Main torsional mode: mode {torsion_mode['mode']}, T = {torsion_mode['T']:.4f} s, RMZ = {torsion_mode['RMZ']:.4f}")
    else:
        print("Main torsional mode: RMZ participation is not available; use mode-shape plots for torsion check.")
    print(f"Suggested main-wave direction = {modal_info['suggested_main_wave_direction']}")


def apply_rayleigh_by_modes(modal_info, zeta=0.05, modes=None, use_initial_stiffness=True, print_summary=True):
    """Apply Rayleigh damping using two modal frequencies from modal_info."""
    if modes is None:
        modes = modal_info.get("rayleigh_target_modes")
    if modes is None or len(modes) != 2:
        raise ValueError("Exactly two Rayleigh target modes are required")
    mode_to_row = {int(row["mode"]): row for row in modal_info["rows"]}
    try:
        rows = [mode_to_row[int(mode)] for mode in modes]
    except KeyError as exc:
        raise ValueError(f"Rayleigh target mode {exc} was not found in modal_info") from exc
    w1, w2 = float(rows[0]["omega"]), float(rows[1]["omega"])
    alpha_m = 2.0 * float(zeta) * w1 * w2 / (w1 + w2)
    beta = 2.0 * float(zeta) / (w1 + w2)
    if use_initial_stiffness:
        rayleigh(alpha_m, 0.0, beta, 0.0)
        beta_name = "betaKinit"
    else:
        rayleigh(alpha_m, beta, 0.0, 0.0)
        beta_name = "betaK"
    info = {
        "direction": "XY",
        "zeta": float(zeta),
        "modes": tuple(int(mode) for mode in modes),
        "periods": tuple(float(row["T"]) for row in rows),
        "omega": (w1, w2),
        "alphaM": float(alpha_m),
        beta_name: float(beta),
        "manual_periods": False,
        "selection_rule": "dominant X and Y translational modes",
    }
    if print_summary:
        print("\nRayleigh damping target modes:")
        print(f"modes = {info['modes']}, periods = ({info['periods'][0]:.6f}, {info['periods'][1]:.6f}) s")
        print(f"zeta = {float(zeta):.3f}")
        print(f"alphaM = {alpha_m:.6e}")
        print(f"{beta_name} = {beta:.6e}")
    return info


def select_uniform_damping_frequency_band(
    modal_info,
    lower_mode=None,
    upper_mode=None,
    cumulative_mass_ratio=90.0,
    directions=("X", "Y"),
    lower_factor=0.80,
    upper_factor=1.20,
    min_freq=1.0e-6,
):
    """Select a practical frequency band for OpenSees Uniform damping."""
    import numpy as _np

    rows = list(modal_info.get("rows", []))
    if not rows:
        raise ValueError("modal_info does not contain modal rows")

    mode_to_row = {int(row["mode"]): row for row in rows}
    max_mode = int(rows[-1]["mode"])

    if lower_mode is None:
        x_mode = modal_info.get("x_main_mode", rows[0])
        y_mode = modal_info.get("y_main_mode", rows[0])
        lower_mode = min(int(x_mode["mode"]), int(y_mode["mode"]))
    lower_mode = max(1, min(int(lower_mode), max_mode))

    dirs = tuple(str(d).upper() for d in directions)
    target = float(cumulative_mass_ratio)

    if upper_mode is None:
        upper_candidates = [lower_mode]
        for direction in dirs:
            if direction in ("X", "MX"):
                key = "MX"
            elif direction in ("Y", "MY"):
                key = "MY"
            elif direction in ("RZ", "TORSION", "RMZ"):
                key = "RMZ"
            else:
                raise ValueError(f"Unknown modal mass direction: {direction}")

            values = _np.asarray([abs(float(row.get(key, 0.0))) for row in rows], dtype=float)
            if not _np.any(_np.isfinite(values)):
                continue
            cumulative = _np.nancumsum(values)
            hits = _np.where(cumulative >= target)[0]
            if hits.size:
                upper_candidates.append(int(rows[int(hits[0])]["mode"]))
            else:
                upper_candidates.append(max_mode)
        upper_mode = max(upper_candidates)

    upper_mode = max(lower_mode, min(int(upper_mode), max_mode))
    selected_rows = [mode_to_row[mode] for mode in range(lower_mode, upper_mode + 1) if mode in mode_to_row]
    if not selected_rows:
        raise ValueError(f"No modal rows found in mode range {lower_mode}..{upper_mode}")

    freq_values = [float(row["freq"]) for row in selected_rows if float(row["freq"]) > 0.0]
    if not freq_values:
        raise ValueError("Selected modes do not contain positive frequencies")

    freq_low = max(float(min_freq), float(lower_factor) * min(freq_values))
    freq_high = float(upper_factor) * max(freq_values)
    if freq_high <= freq_low:
        raise ValueError(f"Invalid Uniform damping frequency band: {freq_low} to {freq_high} Hz")

    return {
        "lower_mode": int(lower_mode),
        "upper_mode": int(upper_mode),
        "modes": tuple(int(row["mode"]) for row in selected_rows),
        "freq_low": float(freq_low),
        "freq_high": float(freq_high),
        "lower_factor": float(lower_factor),
        "upper_factor": float(upper_factor),
        "cumulative_mass_ratio": target,
        "directions": dirs,
    }


def apply_uniform_damping(
    modal_info=None,
    zeta=0.05,
    damping_tag=900001,
    region_tag=910001,
    freq_low=None,
    freq_high=None,
    lower_mode=None,
    upper_mode=None,
    cumulative_mass_ratio=90.0,
    directions=("X", "Y"),
    lower_factor=0.80,
    upper_factor=1.20,
    ele_tags=None,
    chunk_size=1000,
    clear_rayleigh=True,
    print_summary=True,
):
    """
    Define OpenSees Uniform damping and assign it to model elements by regions.

    If freq_low/freq_high are not supplied, modal_info is used to choose a band
    from the dominant X/Y translational modes to the mode where cumulative
    participation reaches cumulative_mass_ratio.
    """
    if not hasattr(ops, "damping"):
        raise RuntimeError("This OpenSeesPy build does not support ops.damping(). Use the opensees environment.")

    if freq_low is None or freq_high is None:
        band = select_uniform_damping_frequency_band(
            modal_info,
            lower_mode=lower_mode,
            upper_mode=upper_mode,
            cumulative_mass_ratio=cumulative_mass_ratio,
            directions=directions,
            lower_factor=lower_factor,
            upper_factor=upper_factor,
        )
        freq_low = band["freq_low"]
        freq_high = band["freq_high"]
    else:
        band = {
            "lower_mode": None,
            "upper_mode": None,
            "modes": (),
            "freq_low": float(freq_low),
            "freq_high": float(freq_high),
            "lower_factor": None,
            "upper_factor": None,
            "cumulative_mass_ratio": None,
            "directions": tuple(str(d).upper() for d in directions),
        }

    freq_low = float(freq_low)
    freq_high = float(freq_high)
    if freq_low <= 0.0 or freq_high <= freq_low:
        raise ValueError(f"Invalid Uniform damping frequency band: {freq_low} to {freq_high} Hz")

    if ele_tags is None:
        ele_tags = ops.getEleTags()
    ele_tags = sorted({int(tag) for tag in ele_tags})
    if not ele_tags:
        raise ValueError("No element tags are available for Uniform damping assignment")

    if clear_rayleigh:
        ops.rayleigh(0.0, 0.0, 0.0, 0.0)

    actual_damping_tag = None
    start_tag = int(damping_tag)
    for candidate in range(start_tag, start_tag + 1000):
        try:
            ops.damping("Uniform", int(candidate), float(zeta), freq_low, freq_high)
            actual_damping_tag = int(candidate)
            break
        except Exception:
            continue
    if actual_damping_tag is None:
        raise RuntimeError(f"Could not create Uniform damping tag starting from {start_tag}")

    chunk_size = max(1, int(chunk_size))
    region_tags = []
    base_region_tag = int(region_tag) + (actual_damping_tag - start_tag) * 1000
    for i in range(0, len(ele_tags), chunk_size):
        chunk = ele_tags[i:i + chunk_size]
        current_region_tag = base_region_tag + len(region_tags)
        ops.region(int(current_region_tag), "-ele", *chunk, "-damp", int(actual_damping_tag))
        region_tags.append(int(current_region_tag))

    info = {
        "type": "Uniform",
        "zeta": float(zeta),
        "damping_tag": int(actual_damping_tag),
        "requested_damping_tag": int(damping_tag),
        "region_tags": tuple(region_tags),
        "element_count": int(len(ele_tags)),
        "freq_low": float(freq_low),
        "freq_high": float(freq_high),
        "band": band,
        "clear_rayleigh": bool(clear_rayleigh),
    }
    if print_summary:
        print("\nUniform damping:")
        print(f"damping tag = {info['damping_tag']}, zeta = {info['zeta']:.3f}")
        print(f"frequency band = {freq_low:.6f} to {freq_high:.6f} Hz")
        if band.get("lower_mode") is not None:
            print(f"mode range used for band = {band['lower_mode']} to {band['upper_mode']}")
            print(f"selection target = cumulative {band['directions']} mass >= {band['cumulative_mass_ratio']:.1f}%")
        print(f"assigned elements = {info['element_count']}, regions = {len(region_tags)}")
        if clear_rayleigh:
            print("Rayleigh coefficients were reset to zero before assigning Uniform damping.")
    return info


def setup_bidirectional_modal_rayleigh(
    num_modes=15,
    zeta=0.05,
    use_initial_stiffness=True,
    eigen_solver=None,
    allow_mode_reduction=True,
    save_opstool=True,
    odb_tag="modal_current",
    plot=True,
    opsvis_module=None,
    plot_scale=1,
    show_outline=False,
    show_origin=False,
    interpolate_beam=False,
    plot_mode_tags=None,
    print_summary=True,
):
    """Analyze modal participation, set XY Rayleigh damping, and optionally plot modes."""
    modal_info = analyze_modal_participation(
        num_modes=num_modes,
        eigen_solver=eigen_solver,
        allow_mode_reduction=allow_mode_reduction,
        print_summary=print_summary,
    )
    ray_info = apply_rayleigh_by_modes(
        modal_info,
        zeta=zeta,
        modes=modal_info["rayleigh_target_modes"],
        use_initial_stiffness=use_initial_stiffness,
        print_summary=print_summary,
    )
    eigen_odb_tag = None
    if plot_mode_tags is None:
        mode_tags_to_plot = [1, len(modal_info["rows"])]
    else:
        mode_tags_to_plot = sorted({int(mode) for mode in plot_mode_tags})
        max_mode = len(modal_info["rows"])
        bad_modes = [mode for mode in mode_tags_to_plot if mode < 1 or mode > max_mode]
        if bad_modes:
            raise ValueError(f"plot_mode_tags contains modes outside 1..{max_mode}: {bad_modes}")
        if len(mode_tags_to_plot) == 1:
            mode_tags_to_plot = [mode_tags_to_plot[0], mode_tags_to_plot[0]]
        else:
            mode_tags_to_plot = [mode_tags_to_plot[0], mode_tags_to_plot[-1]]
    if save_opstool or plot:
        eigen_odb_tag = str(odb_tag)
        save_current_eigen_data_for_opstool(
            odb_tag=eigen_odb_tag,
            mode_tag=len(modal_info["rows"]),
            interpolate_beam=interpolate_beam,
        )
    fig = None
    if plot:
        if opsvis_module is None:
            import opstool.vis.plotly as _opsvis_module
            opsvis_module = _opsvis_module
        fig = opsvis_module.plot_eigen(
            mode_tags=mode_tags_to_plot,
            odb_tag=eigen_odb_tag,
            scale=plot_scale,
            show_outline=show_outline,
            show_origin=show_origin,
            interpolate_beam=interpolate_beam,
        )
        if hasattr(fig, "show"):
            fig.show()
    if print_summary:
        print(f"Mode range plotted = {mode_tags_to_plot[0]} to {mode_tags_to_plot[1]}")
    return {
        "modal": modal_info,
        "rayleigh": ray_info,
        "odb_tag": eigen_odb_tag,
        "mode_tags_to_plot": tuple(int(x) for x in mode_tags_to_plot),
        "figure": fig,
    }

def read_ground_motion(
    file_path,
    dt=None,
    output_file=None,
    input_unit="auto",
    model_g=9800.0,
    accel_column="auto",
    delimiter=None,
    return_array=True,
    overwrite=True,
    use_readrecord=True,
):
    """
    Read a ground-motion file and create a one-column acceleration file for OpenSees.

    Supported input:
        - .at2: PEER/AT2-style file. Values are treated as g by default and are
          converted to model acceleration units by multiplying model_g.
        - .txt/.dat/.csv: numeric text file. By default values are treated as
          already in model acceleration units. One-column files are read directly.
          For two-or-more-column files, accel_column='auto' uses the last column
          and tries to infer dt from the first column.

    Parameters
    ----------
    file_path : str or path-like
        Input ground-motion file.
    dt : float, optional
        Time step. Required for one-column text files unless it can be parsed
        from an AT2 header or inferred from a time column.
    output_file : str or path-like, optional
        Clean one-column output file. If None, a sibling file named
        '<stem>_opensees_clean.dat' is written.
    input_unit : {'auto', 'g', 'model'}, default 'auto'
        Unit of input acceleration values. For .at2, auto means 'g'. For text,
        auto means 'model'.
    model_g : float, default 9800.0
        Gravity acceleration in model units. Use 9800.0 for N-mm-s models.
    accel_column : {'auto', int}, default 'auto'
        Acceleration column for text files. Integer uses zero-based indexing.
        'auto' uses column 0 for one-column files and the last column otherwise.
    delimiter : str, optional
        Passed to numpy.loadtxt for text files. None handles whitespace.
    return_array : bool, default True
        Whether to include the acceleration numpy array in the returned dict.
    overwrite : bool, default True
        Whether to overwrite output_file if it exists.
    use_readrecord : bool, default True
        For .at2 files, try to use ReadRecord.py from the input-file folder
        before falling back to the built-in parser.

    Returns
    -------
    dict
        Keys include path, clean_file, dt, n_step, accel, source_type,
        input_unit, factor, peak_abs_accel.

    Example
    -------
    gm = ufp.read_ground_motion(r"E:\\openseespy\\KJ\\elCentro.at2")
    dt = gm["dt"]
    n_step = gm["n_step"]
    timeSeries("Path", 2, "-filePath", gm["clean_file"], "-dt", dt)
    """
    from pathlib import Path as _Path
    import re as _re
    import numpy as _np

    path = _Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Ground-motion file not found: {path}")

    suffix = path.suffix.lower()
    if output_file is None:
        output_path = path.with_name(f"{path.stem}_opensees_clean.dat")
    else:
        output_path = _Path(output_file).expanduser()
        if not output_path.is_absolute():
            output_path = path.parent / output_path

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    def _numeric_tokens(text):
        pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
        return [float(x) for x in _re.findall(pattern, text)]

    def _parse_at2_dt(text):
        patterns = [
            r"NPTS\s*=\s*\d+\s*,?\s*DT\s*=\s*([0-9.Ee+-]+)",
            r"(\d+)\s+points\s+at\s+equal\s+spacing\s+of\s+([0-9.Ee+-]+)\s+sec",
            r"DT\s*=\s*([0-9.Ee+-]+)",
        ]
        for pat in patterns:
            match = _re.search(pat, text, flags=_re.IGNORECASE)
            if match:
                return float(match.groups()[-1])
        return None

    if suffix == ".at2":
        text = path.read_text(encoding="utf-8", errors="ignore")
        parsed_dt = _parse_at2_dt(text)
        values = None

        readrecord_file = path.parent / "ReadRecord.py"
        if use_readrecord and readrecord_file.exists():
            try:
                import importlib.util as _importlib_util
                import uuid as _uuid

                module_name = f"_ufp_ReadRecord_{_uuid.uuid4().hex}"
                spec = _importlib_util.spec_from_file_location(module_name, readrecord_file)
                module = _importlib_util.module_from_spec(spec)
                spec.loader.exec_module(module)

                raw_output = output_path.with_name(f"{output_path.stem}_readrecord_raw.dat")
                rr_dt, _rr_npts = module.ReadRecord(str(path), str(raw_output))
                if dt is None and rr_dt:
                    dt = float(rr_dt)
                raw_text = raw_output.read_text(encoding="utf-8", errors="ignore")
                values = _np.asarray(_numeric_tokens(raw_text), dtype=float)
                try:
                    raw_output.unlink()
                except Exception:
                    pass
            except Exception as err:
                print(f"ReadRecord.py parsing failed, using built-in AT2 parser: {err}")

        if values is None:
            if dt is None:
                dt = parsed_dt
            lines = text.splitlines()
            begin_line = None
            for i, line in enumerate(lines):
                line_lower = line.lower()
                if "begin" in line_lower and "data" in line_lower:
                    begin_line = i
                    break
            if begin_line is not None:
                data_lines = lines[begin_line + 1:]
            else:
                # Common AT2 files use four header lines before numeric data.
                data_lines = lines[4:]

            vals = []
            for line in data_lines:
                line_lower = line.lower()
                if "end" in line_lower and "data" in line_lower:
                    break
                if "npts" in line_lower or "dt" in line_lower:
                    continue
                vals.extend(_numeric_tokens(line))
            values = _np.asarray(vals, dtype=float)

        if dt is None:
            raise ValueError("dt was not supplied and could not be parsed from the AT2 header.")

        source_type = "at2"
        actual_input_unit = "g" if input_unit == "auto" else str(input_unit).lower()
    else:
        raw = _np.loadtxt(path, delimiter=delimiter)
        raw = _np.asarray(raw, dtype=float)
        if raw.ndim == 0:
            values = raw.reshape(1)
        elif raw.ndim == 1:
            values = raw
        else:
            if accel_column == "auto":
                col = raw.shape[1] - 1
            else:
                col = int(accel_column)
            values = raw[:, col]
            if dt is None and raw.shape[1] >= 2:
                time_col = raw[:, 0]
                dts = _np.diff(time_col)
                dts = dts[_np.isfinite(dts) & (_np.abs(dts) > 0.0)]
                if dts.size:
                    dt = float(_np.median(dts))

        if dt is None:
            raise ValueError("dt must be supplied for a one-column text ground-motion file.")
        source_type = suffix.lstrip(".") or "text"
        actual_input_unit = "model" if input_unit == "auto" else str(input_unit).lower()

    values = values[_np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"No numeric acceleration values were found in {path}")

    if actual_input_unit in ("g", "gravity"):
        factor = float(model_g)
    elif actual_input_unit in ("model", "model_unit", "model_units", "accel", "acceleration"):
        factor = 1.0
    else:
        raise ValueError("input_unit must be 'auto', 'g', or 'model'")

    accel = values * factor
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _np.savetxt(output_path, accel, fmt="%.10e")

    info = {
        "path": str(path),
        "clean_file": str(output_path),
        "dt": float(dt),
        "n_step": int(accel.size),
        "source_type": source_type,
        "input_unit": actual_input_unit,
        "factor": float(factor),
        "peak_abs_accel": float(_np.max(_np.abs(accel))),
        "min_accel": float(_np.min(accel)),
        "max_accel": float(_np.max(accel)),
    }
    if return_array:
        info["accel"] = accel
    return info


def _scaled_ground_motion_for_plot(info, scale_factor=1.0):
    """Return scaled acceleration history and PGA information from read_ground_motion output."""
    import numpy as _np

    accel = _np.loadtxt(info["clean_file"], dtype=float)
    accel = _np.asarray(accel, dtype=float).reshape(-1) * float(scale_factor)
    time = _np.arange(accel.size, dtype=float) * float(info["dt"])
    pga_index = int(_np.argmax(_np.abs(accel)))
    pga_value = float(accel[pga_index])
    return {
        "time": time,
        "accel": accel,
        "pga_index": pga_index,
        "pga_value": pga_value,
        "pga_abs": abs(pga_value),
    }


def plot_bidirectional_ground_motion(
    gm_x_info,
    gm_y_info,
    record_x="X",
    record_y="Y",
    scale_x=1.0,
    scale_y=1.0,
    mm_per_gal=10.0,
    figsize=(11, 4.8),
    title="Bidirectional ground-motion acceleration histories",
    show=True,
    ax=None,
):
    """
    Plot X/Y ground-motion acceleration histories in one axes after scaling.

    Parameters
    ----------
    gm_x_info, gm_y_info : dict
        Dictionaries returned by read_ground_motion(...).
    record_x, record_y : str
        Record names shown in the legend.
    scale_x, scale_y : float
        Scale factors used in OpenSees Path timeSeries.
    mm_per_gal : float, default 10.0
        Unit conversion for N-mm-s models: 1 gal = 10 mm/s^2.
    figsize : tuple, default (11, 4.8)
        Figure size when ax is not supplied.
    title : str
        Figure title.
    show : bool, default True
        Whether to call plt.show().
    ax : matplotlib axes, optional
        Existing axes to draw on.

    Returns
    -------
    dict
        Contains fig, ax, ax_gal, and X/Y PGA summary.
    """
    import matplotlib.pyplot as _plt

    x = _scaled_ground_motion_for_plot(gm_x_info, scale_x)
    y = _scaled_ground_motion_for_plot(gm_y_info, scale_y)

    if ax is None:
        fig, ax = _plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    line_data = [
        (x, "X", str(record_x), float(scale_x), "#1f77b4"),
        (y, "Y", str(record_y), float(scale_y), "#2ca02c"),
    ]

    summary = {}
    for item, direction, record_name, scale_factor, color in line_data:
        pga_gal = item["pga_abs"] / float(mm_per_gal)
        summary[direction] = {
            "record": record_name,
            "scale": scale_factor,
            "pga_mm_s2": item["pga_abs"],
            "pga_gal": pga_gal,
            "pga_g": pga_gal / 980.0,
            "pga_time": float(item["time"][item["pga_index"]]),
            "pga_signed_mm_s2": item["pga_value"],
        }
        ax.plot(
            item["time"],
            item["accel"],
            lw=0.8,
            color=color,
            label=f"{direction}: {record_name}, scale={scale_factor:g}, PGA={pga_gal:.2f} gal",
        )
        ax.scatter(
            [item["time"][item["pga_index"]]],
            [item["pga_value"]],
            s=42,
            color="#d62728",
            edgecolors="white",
            linewidths=0.6,
            zorder=4,
        )

    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (mm/s^2)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", frameon=True)

    ax_gal = ax.twinx()
    y1, y2 = ax.get_ylim()
    ax_gal.set_ylim(y1 / float(mm_per_gal), y2 / float(mm_per_gal))
    ax_gal.set_ylabel("Acceleration (gal)")

    fig.tight_layout()
    if show:
        _plt.show()

    return {
        "fig": fig,
        "ax": ax,
        "ax_gal": ax_gal,
        "summary": summary,
    }


def _newmark_response_spectrum(accel, dt, periods, damping_ratio=0.05):
    """Compute linear SDOF response spectra with average-acceleration Newmark."""
    import numpy as _np

    accel = _np.asarray(accel, dtype=float).reshape(-1)
    periods = _np.asarray(periods, dtype=float).reshape(-1)
    if accel.size < 2:
        raise ValueError("Ground-motion acceleration must contain at least two points")
    if float(dt) <= 0.0:
        raise ValueError("dt must be positive")
    if not 0.0 <= float(damping_ratio) < 1.0:
        raise ValueError("damping_ratio must be between 0 and 1")
    if periods.size == 0 or _np.any(periods < 0.0):
        raise ValueError("periods must contain non-negative values")

    sd = _np.zeros(periods.size, dtype=float)
    psv = _np.zeros(periods.size, dtype=float)
    psa = _np.zeros(periods.size, dtype=float)
    sa_abs = _np.zeros(periods.size, dtype=float)
    zero_mask = periods <= 0.0
    sa_abs[zero_mask] = float(_np.max(_np.abs(accel)))
    psa[zero_mask] = sa_abs[zero_mask]

    active = _np.where(~zero_mask)[0]
    if active.size == 0:
        return {"sd": sd, "psv": psv, "psa": psa, "sa_abs": sa_abs}

    omega = 2.0 * _np.pi / periods[active]
    stiffness = omega ** 2
    damping = 2.0 * float(damping_ratio) * omega
    beta = 0.25
    gamma = 0.50
    record_dt = float(dt)
    min_positive_period = float(_np.min(periods[active]))
    substeps = max(1, int(_np.ceil(record_dt / (min_positive_period / 20.0))))
    dt = record_dt / substeps
    a0 = 1.0 / (beta * dt ** 2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    effective_stiffness = stiffness + a0 + a1 * damping

    disp = _np.zeros(active.size, dtype=float)
    vel = _np.zeros(active.size, dtype=float)
    rel_accel = _np.full(active.size, -accel[0], dtype=float)
    max_disp = _np.abs(disp)
    max_abs_accel = _np.abs(rel_accel + accel[0])

    previous_ground_accel = float(accel[0])
    for target_ground_accel in accel[1:]:
        delta_ground_accel = float(target_ground_accel) - previous_ground_accel
        for substep in range(1, substeps + 1):
            ground_accel = previous_ground_accel + delta_ground_accel * substep / substeps
            effective_load = (
                -ground_accel
                + a0 * disp + a2 * vel + a3 * rel_accel
                + damping * (a1 * disp + a4 * vel + a5 * rel_accel)
            )
            new_disp = effective_load / effective_stiffness
            new_rel_accel = a0 * (new_disp - disp) - a2 * vel - a3 * rel_accel
            new_vel = vel + dt * ((1.0 - gamma) * rel_accel + gamma * new_rel_accel)

            max_disp = _np.maximum(max_disp, _np.abs(new_disp))
            max_abs_accel = _np.maximum(max_abs_accel, _np.abs(new_rel_accel + ground_accel))
            disp, vel, rel_accel = new_disp, new_vel, new_rel_accel
        previous_ground_accel = float(target_ground_accel)

    sd[active] = max_disp
    psv[active] = omega * max_disp
    psa[active] = stiffness * max_disp
    sa_abs[active] = max_abs_accel
    return {"sd": sd, "psv": psv, "psa": psa, "sa_abs": sa_abs}


def plot_bidirectional_response_spectrum(
    gm_x_info,
    gm_y_info,
    record_x="X",
    record_y="Y",
    scale_x=1.0,
    scale_y=1.0,
    damping_ratio=0.05,
    periods=None,
    period_min=0.02,
    period_max=6.0,
    period_step=0.02,
    model_g=9800.0,
    output_file=None,
    figsize=(8.2, 5.2),
    title=None,
    show=True,
    ax=None,
):
    """
    Plot scaled X/Y pseudo-acceleration spectra and their geometric mean.

    Scale factors are applied exactly once to the clean acceleration files,
    matching the OpenSees Path timeSeries. The geometric-mean spectrum is
    sqrt(Sa_X * Sa_Y) at each common period. Spectral acceleration uses g.
    """
    from pathlib import Path as _Path
    import matplotlib.pyplot as _plt
    import numpy as _np

    if abs(float(gm_x_info["dt"]) - float(gm_y_info["dt"])) > 1.0e-12:
        raise ValueError(
            f"X/Y ground-motion dt mismatch: {gm_x_info['dt']} and {gm_y_info['dt']}"
        )
    if periods is None:
        if float(period_min) <= 0.0 or float(period_max) < float(period_min):
            raise ValueError("period_min and period_max must define a positive range")
        if float(period_step) <= 0.0:
            raise ValueError("period_step must be positive")
        positive_periods = _np.arange(
            float(period_min),
            float(period_max) + 0.5 * float(period_step),
            float(period_step),
        )
        periods = _np.concatenate(([0.0], positive_periods))
    else:
        periods = _np.asarray(periods, dtype=float).reshape(-1)

    x = _scaled_ground_motion_for_plot(gm_x_info, scale_x)
    y = _scaled_ground_motion_for_plot(gm_y_info, scale_y)
    spectrum_x = _newmark_response_spectrum(
        x["accel"], gm_x_info["dt"], periods, damping_ratio=damping_ratio
    )
    spectrum_y = _newmark_response_spectrum(
        y["accel"], gm_y_info["dt"], periods, damping_ratio=damping_ratio
    )
    sa_x_g = spectrum_x["psa"] / float(model_g)
    sa_y_g = spectrum_y["psa"] / float(model_g)
    sa_geomean_g = _np.sqrt(_np.maximum(sa_x_g, 0.0) * _np.maximum(sa_y_g, 0.0))

    table = _np.column_stack((periods, sa_x_g, sa_y_g, sa_geomean_g))
    output_path = None
    if output_file is not None:
        output_path = _Path(output_file).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"5%-damped pseudo-acceleration response spectra; "
            f"damping_ratio={float(damping_ratio):.6f}\n"
            f"X={record_x}; scale_x={float(scale_x):.10g}; "
            f"Y={record_y}; scale_y={float(scale_y):.10g}\n"
            "Period_s  Sa_X_g  Sa_Y_g  Sa_GeometricMean_g"
        )
        _np.savetxt(output_path, table, fmt="%.8e", header=header)

    if ax is None:
        fig, ax = _plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    damping_percent = 100.0 * float(damping_ratio)
    ax.plot(periods, sa_x_g, color="#4C78A8", lw=1.1, alpha=0.72, label=f"X: {record_x}")
    ax.plot(periods, sa_y_g, color="#59A14F", lw=1.1, alpha=0.72, label=f"Y: {record_y}")
    ax.plot(periods, sa_geomean_g, color="#111111", lw=2.1, label="Geometric mean")
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Pseudo-spectral acceleration, Sa (g)")
    ax.set_title(title or f"Bidirectional response spectra ({damping_percent:g}% damping)")
    ax.set_xlim(float(_np.min(periods)), float(_np.max(periods)))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", alpha=0.32)
    ax.legend(frameon=False)
    fig.tight_layout()
    if output_path is not None:
        print(f"Geometric-mean response spectrum saved: {output_path}")
    if show:
        _plt.show()

    return {
        "fig": fig,
        "ax": ax,
        "periods": periods,
        "sa_x_g": sa_x_g,
        "sa_y_g": sa_y_g,
        "sa_geomean_g": sa_geomean_g,
        "sd_x_mm": spectrum_x["sd"],
        "sd_y_mm": spectrum_y["sd"],
        "sa_abs_x_g": spectrum_x["sa_abs"] / float(model_g),
        "sa_abs_y_g": spectrum_y["sa_abs"] / float(model_g),
        "damping_ratio": float(damping_ratio),
        "output_file": str(output_path) if output_path is not None else None,
    }


def _native_direction_dof(direction):
    text = str(direction).strip().upper()
    aliases = {
        "X": ("UX", 1),
        "UX": ("UX", 1),
        "1": ("UX", 1),
        "Y": ("UY", 2),
        "UY": ("UY", 2),
        "2": ("UY", 2),
    }
    if text not in aliases:
        raise ValueError(f"Unsupported recorder direction: {direction!r}. Use UX/UY.")
    return aliases[text]


def _story_lookup_item(data, story):
    if story in data:
        return data[story]
    story_int = int(story)
    if story_int in data:
        return data[story_int]
    story_str = str(story_int)
    if story_str in data:
        return data[story_str]
    raise KeyError(f"Story {story} not found")


def _component_story_from_name(name, component_data=None):
    story = _story_number_from_component_name(name)
    if story is not None:
        return int(story)
    if isinstance(component_data, dict):
        for key in ("story", "floor", "story_id"):
            if component_data.get(key) is not None:
                return int(component_data[key])
    return None


def collect_native_component_metadata(model_vars=None):
    """
    Collect frame and wall element metadata from table-driven component dicts.

    The function scans variables produced by beam_mesh/column_mesh/wall_mesh and
    returns three DataFrames: all components, frame elements, and wall elements.
    """
    if model_vars is None:
        components = _collect_components_from_scope(frame_level=2)
    else:
        components = {
            name: obj for name, obj in dict(model_vars).items()
            if isinstance(obj, dict) and "added_elements" in obj
        }

    rows = []
    seen_ele = set()
    for name, comp in sorted(components.items(), key=lambda item: str(item[0])):
        kind = _component_kind(comp)
        if kind not in ("beam", "column", "wall"):
            continue
        story = _component_story_from_name(name, comp)
        for local_index, ele_tag in enumerate(comp.get("added_elements", []), start=1):
            ele_tag = int(ele_tag)
            if ele_tag in seen_ele:
                continue
            seen_ele.add(ele_tag)
            try:
                nodes = [int(nd) for nd in ops.eleNodes(ele_tag)]
                coords = np.asarray([ops.nodeCoord(nd) for nd in nodes], dtype=float)
                if coords.ndim == 1:
                    coords = coords.reshape(1, -1)
                if coords.shape[1] == 2:
                    coords = np.column_stack([coords, np.zeros(coords.shape[0])])
                center = coords.mean(axis=0)
                xyz_min = coords.min(axis=0)
                xyz_max = coords.max(axis=0)
            except Exception:
                nodes = []
                center = np.asarray([np.nan, np.nan, np.nan], dtype=float)
                xyz_min = np.asarray([np.nan, np.nan, np.nan], dtype=float)
                xyz_max = np.asarray([np.nan, np.nan, np.nan], dtype=float)

            rows.append({
                "eleTag": ele_tag,
                "story": story,
                "component": str(name),
                "kind": kind,
                "component_ele_index": int(local_index),
                "node_tags": ";".join(str(nd) for nd in nodes),
                "x_mid": float(center[0]),
                "y_mid": float(center[1]),
                "z_mid": float(center[2]),
                "x_min": float(xyz_min[0]),
                "x_max": float(xyz_max[0]),
                "y_min": float(xyz_min[1]),
                "y_max": float(xyz_max[1]),
                "z_min": float(xyz_min[2]),
                "z_max": float(xyz_max[2]),
                "integration_tag": comp.get("integration_tag"),
                "sec_tag_wall": comp.get("sec_tag_wall"),
            })

    columns = [
        "eleTag", "story", "component", "kind", "component_ele_index",
        "node_tags", "x_mid", "y_mid", "z_mid", "x_min", "x_max",
        "y_min", "y_max", "z_min", "z_max", "integration_tag",
        "sec_tag_wall",
    ]
    all_df = pd.DataFrame(rows, columns=columns)
    if all_df.empty:
        frame_df = pd.DataFrame(columns=columns)
        wall_df = pd.DataFrame(columns=columns)
    else:
        frame_df = all_df[all_df["kind"].isin(["beam", "column"])].copy()
        wall_df = all_df[all_df["kind"].eq("wall")].copy()
    return {
        "components": all_df,
        "frame": frame_df,
        "wall": wall_df,
    }


def select_floor_corner_nodes(elevations, node_tol=1e-6):
    """Select four plan-corner nodes at each floor elevation for torsion checks."""
    corner_rows = []
    for story, z in enumerate(elevations, start=1):
        nodes = [int(nd) for nd in ops.getNodeTags() if abs(float(ops.nodeCoord(nd)[2]) - float(z)) <= node_tol]
        if not nodes:
            continue
        coords = []
        for nd in nodes:
            x, y, zc = ops.nodeCoord(nd)
            coords.append((nd, float(x), float(y), float(zc)))
        labels = {
            "minX_minY": lambda item: (item[1], item[2]),
            "minX_maxY": lambda item: (item[1], -item[2]),
            "maxX_minY": lambda item: (-item[1], item[2]),
            "maxX_maxY": lambda item: (-item[1], -item[2]),
        }
        used = set()
        for label, keyfunc in labels.items():
            available = [item for item in coords if item[0] not in used]
            if not available:
                available = coords
            nd, x, y, zc = min(available, key=keyfunc)
            used.add(nd)
            corner_rows.append({
                "story": int(story),
                "corner": label,
                "node": int(nd),
                "x": float(x),
                "y": float(y),
                "z": float(zc),
            })
    return pd.DataFrame(corner_rows, columns=["story", "corner", "node", "x", "y", "z"])


def _write_native_metadata_csv(path, df):
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _recorder_args_with_optional_dt(recorder_dt):
    if recorder_dt is None:
        return []
    return ["-dT", float(recorder_dt)]


def setup_full_native_dynamic_recorders(
    out_dir,
    story_weight_summary,
    diaphragm_info,
    model_vars=None,
    directions=("UX", "UY"),
    recorder_dt=None,
    precision=12,
    g=9800.0,
    base_z=0.0,
    node_tol=1e-6,
    ground_motion_info=None,
    ground_motion_scales=None,
    record_initial=True,
    record_story_drift=True,
    record_floor_corners=True,
    record_frame_local_force=True,
    record_frame_basic=True,
    record_frame_sections=True,
    record_wall_forces=True,
    record_wall_stresses=True,
    frame_section_i=1,
    frame_section_j=5,
    print_summary=True,
):
    """
    Define native OpenSees recorders for dynamic response indicators.

    By default recorder_dt is None, so OpenSees records every converged analysis
    step. Fiber/material-point recorders are intentionally not included here.
    """
    import json as _json
    from pathlib import Path as _Path

    out_dir = _Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ops.remove("recorders")
    except Exception:
        pass

    story_data = _story_data_from_summary(story_weight_summary)
    diaphragm_results = _diaphragm_results_from_info(diaphragm_info)
    story_ids = sorted(int(story) for story in story_data.keys())

    story_rows = []
    floor_rows = []
    floor_control_nodes = []
    elevations = []
    for story in story_ids:
        item = _story_lookup_item(story_data, story)
        dia = _story_lookup_item(diaphragm_results, story)
        master_node = int(dia["master_node"])
        z = float(item.get("z", ops.nodeCoord(master_node)[2]))
        height = float(z - (elevations[-1] if elevations else 0.0))
        weight = float(item.get("total_weight", item.get("W_seismic", 0.0)))
        mass = float(dia.get("m_total", weight / float(g)))
        floor_nodes = [int(nd) for nd in dia.get("floor_nodes", [])]

        elevations.append(z)
        floor_control_nodes.append(master_node)
        master_coord = ops.nodeCoord(master_node)
        story_rows.append({
            "story": int(story),
            "elevation": z,
            "story_height": height,
            "mass": mass,
            "weight": weight,
            "control_node": master_node,
            "control_x": float(master_coord[0]),
            "control_y": float(master_coord[1]),
            "control_z": float(master_coord[2]),
        })
        floor_rows.append({
            "story": int(story),
            "control_node": master_node,
            "node_group": "diaphragm_master",
            "floor_nodes": ";".join(str(nd) for nd in floor_nodes),
        })

    story_df = pd.DataFrame(story_rows)
    floor_df = pd.DataFrame(floor_rows)
    base_nodes = sorted(int(nd) for nd in get_nodes_at_z(float(base_z), tol=node_tol))
    base_df = pd.DataFrame({
        "node": base_nodes,
        "x": [float(ops.nodeCoord(nd)[0]) for nd in base_nodes],
        "y": [float(ops.nodeCoord(nd)[1]) for nd in base_nodes],
        "z": [float(ops.nodeCoord(nd)[2]) for nd in base_nodes],
    })

    metadata = collect_native_component_metadata(model_vars=model_vars)
    comp_df = metadata["components"]
    frame_df = metadata["frame"]
    wall_df = metadata["wall"]

    corner_df = select_floor_corner_nodes(elevations, node_tol=node_tol) if record_floor_corners else pd.DataFrame()

    _write_native_metadata_csv(out_dir / "meta_story.csv", story_df)
    _write_native_metadata_csv(out_dir / "meta_floor_nodes.csv", floor_df)
    _write_native_metadata_csv(out_dir / "meta_base_nodes.csv", base_df)
    _write_native_metadata_csv(out_dir / "meta_components.csv", comp_df)
    _write_native_metadata_csv(out_dir / "meta_frame_elements.csv", frame_df)
    _write_native_metadata_csv(out_dir / "meta_wall_elements.csv", wall_df)
    if record_floor_corners:
        _write_native_metadata_csv(out_dir / "meta_floor_corner_nodes.csv", corner_df)

    ground_motion_info = ground_motion_info or {}
    ground_motion_scales = ground_motion_scales or {}
    for direction in directions:
        label, _dof = _native_direction_dof(direction)
        gm = ground_motion_info.get(label) or ground_motion_info.get(label[-1]) or ground_motion_info.get(direction)
        if gm is None:
            continue
        scale = ground_motion_scales.get(label, ground_motion_scales.get(label[-1], ground_motion_scales.get(direction, 1.0)))
        accel = np.loadtxt(gm["clean_file"], dtype=float).reshape(-1) * float(scale)
        time_values = np.arange(accel.size, dtype=float) * float(gm["dt"])
        gm_df = pd.DataFrame({"time": time_values, "accel": accel})
        _write_native_metadata_csv(out_dir / f"ground_motion_{label}.csv", gm_df)

    dt_args = _recorder_args_with_optional_dt(recorder_dt)
    response_shapes = {
        "floor_control_nodes": floor_control_nodes,
        "base_nodes": base_nodes,
        "frame_ele_tags": frame_df["eleTag"].astype(int).tolist() if not frame_df.empty else [],
        "wall_ele_tags": wall_df["eleTag"].astype(int).tolist() if not wall_df.empty else [],
        "directions": [],
        "recorder_dt": recorder_dt,
    }

    for direction in directions:
        label, dof = _native_direction_dof(direction)
        response_shapes["directions"].append(label)
        if floor_control_nodes:
            ops.recorder(
                "Node", "-file", str(out_dir / f"node_floor_disp_{label}.out"),
                "-time", *dt_args, "-precision", int(precision),
                "-node", *floor_control_nodes, "-dof", int(dof), "disp",
            )
            ops.recorder(
                "Node", "-file", str(out_dir / f"node_floor_accel_{label}.out"),
                "-time", *dt_args, "-precision", int(precision),
                "-node", *floor_control_nodes, "-dof", int(dof), "accel",
            )
        if base_nodes:
            ops.recorder(
                "Node", "-file", str(out_dir / f"node_base_reaction_{label}.out"),
                "-time", *dt_args, "-precision", int(precision),
                "-node", *base_nodes, "-dof", int(dof), "reaction",
            )
        if record_story_drift and len(floor_control_nodes) >= 1:
            lower_nodes = []
            upper_nodes = []
            ground_node = None
            if base_nodes:
                x0, y0, _ = ops.nodeCoord(floor_control_nodes[0])
                ground_node = min(
                    base_nodes,
                    key=lambda nd: (float(ops.nodeCoord(nd)[0]) - float(x0)) ** 2
                    + (float(ops.nodeCoord(nd)[1]) - float(y0)) ** 2,
                )
            for i, upper in enumerate(floor_control_nodes):
                lower_nodes.append(int(ground_node if i == 0 and ground_node is not None else floor_control_nodes[i - 1]))
                upper_nodes.append(int(upper))
            drift_df = pd.DataFrame({
                "story": story_ids,
                "iNode": lower_nodes,
                "jNode": upper_nodes,
                "direction": label,
            })
            _write_native_metadata_csv(out_dir / f"meta_story_drift_nodes_{label}.csv", drift_df)
            try:
                ops.recorder(
                    "Drift", "-file", str(out_dir / f"story_drift_{label}.out"),
                    "-time", *dt_args, "-precision", int(precision),
                    "-iNode", *lower_nodes, "-jNode", *upper_nodes,
                    "-dof", int(dof), "-perpDirn", 3,
                )
            except Exception:
                if print_summary:
                    print(f"Drift recorder is not available in this OpenSeesPy build; {label} drift will be postprocessed from floor displacement.")
        if record_floor_corners and not corner_df.empty:
            corner_nodes = corner_df["node"].astype(int).tolist()
            ops.recorder(
                "Node", "-file", str(out_dir / f"node_floor_corner_disp_{label}.out"),
                "-time", *dt_args, "-precision", int(precision),
                "-node", *corner_nodes, "-dof", int(dof), "disp",
            )
            ops.recorder(
                "Node", "-file", str(out_dir / f"node_floor_corner_accel_{label}.out"),
                "-time", *dt_args, "-precision", int(precision),
                "-node", *corner_nodes, "-dof", int(dof), "accel",
            )

    frame_ele_tags = response_shapes["frame_ele_tags"]
    wall_ele_tags = response_shapes["wall_ele_tags"]
    if frame_ele_tags and record_frame_local_force:
        ops.recorder(
            "Element", "-file", str(out_dir / "frame_localForce.out"),
            "-time", *dt_args, "-precision", int(precision),
            "-ele", *frame_ele_tags, "localForce",
        )
    if frame_ele_tags and record_frame_basic:
        ops.recorder(
            "Element", "-file", str(out_dir / "frame_basicForce.out"),
            "-time", *dt_args, "-precision", int(precision),
            "-ele", *frame_ele_tags, "basicForce",
        )
        ops.recorder(
            "Element", "-file", str(out_dir / "frame_basicDeformation.out"),
            "-time", *dt_args, "-precision", int(precision),
            "-ele", *frame_ele_tags, "basicDeformation",
        )
    if frame_ele_tags and record_frame_sections:
        for sec_label, sec_num in (("i", int(frame_section_i)), ("j", int(frame_section_j))):
            ops.recorder(
                "Element", "-file", str(out_dir / f"frame_section_force_{sec_label}.out"),
                "-time", *dt_args, "-precision", int(precision),
                "-ele", *frame_ele_tags, "section", int(sec_num), "force",
            )
            ops.recorder(
                "Element", "-file", str(out_dir / f"frame_section_deformation_{sec_label}.out"),
                "-time", *dt_args, "-precision", int(precision),
                "-ele", *frame_ele_tags, "section", int(sec_num), "deformation",
            )
    if wall_ele_tags and record_wall_forces:
        ops.recorder(
            "Element", "-file", str(out_dir / "wall_forces.out"),
            "-time", *dt_args, "-precision", int(precision),
            "-ele", *wall_ele_tags, "forces",
        )
    wall_stress_recorded = False
    if wall_ele_tags and record_wall_stresses:
        try:
            test_resp = ops.eleResponse(int(wall_ele_tags[0]), "stresses")
            if test_resp is not None and len(test_resp) > 0:
                ops.recorder(
                    "Element", "-file", str(out_dir / "wall_stresses.out"),
                    "-time", *dt_args, "-precision", int(precision),
                    "-ele", *wall_ele_tags, "stresses",
                )
                wall_stress_recorded = True
        except Exception:
            wall_stress_recorded = False

    response_shapes["wall_stresses_recorded"] = bool(wall_stress_recorded)
    with (out_dir / "response_shapes.json").open("w", encoding="utf-8") as f:
        _json.dump(response_shapes, f, ensure_ascii=False, indent=2)

    info = {
        "out_dir": str(out_dir),
        "story_count": int(len(story_rows)),
        "floor_control_nodes": floor_control_nodes,
        "base_nodes": base_nodes,
        "frame_ele_tags": frame_ele_tags,
        "wall_ele_tags": wall_ele_tags,
        "directions": tuple(response_shapes["directions"]),
        "recorder_dt": recorder_dt,
        "records_every_step": recorder_dt is None,
        "wall_stresses_recorded": bool(wall_stress_recorded),
    }
    with (out_dir / "recorder_info.json").open("w", encoding="utf-8") as f:
        _json.dump(info, f, ensure_ascii=False, indent=2)

    if record_initial:
        try:
            ops.record()
        except Exception:
            pass

    if print_summary:
        print("\nNative OpenSees recorders:")
        print(f"output dir = {out_dir}")
        print(f"directions = {tuple(response_shapes['directions'])}")
        print("record interval = every converged analysis step" if recorder_dt is None else f"record interval dT = {recorder_dt}")
        print(f"floor nodes = {len(floor_control_nodes)}, base nodes = {len(base_nodes)}")
        print(f"frame elements = {len(frame_ele_tags)}, wall elements = {len(wall_ele_tags)}")
        print(f"wall stresses recorded = {wall_stress_recorded}")
        print("fiber/material-point recorders = off")
    return info


def _read_native_recorder(path):
    path = os.fspath(path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None, None
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1:]


def _native_ground_accel(out_dir, direction, time_values):
    path = os.path.join(os.fspath(out_dir), f"ground_motion_{direction}.csv")
    if not os.path.exists(path):
        return np.zeros_like(time_values, dtype=float)
    gm = pd.read_csv(path)
    if gm.empty:
        return np.zeros_like(time_values, dtype=float)
    return np.interp(time_values, gm["time"].to_numpy(float), gm["accel"].to_numpy(float), left=0.0, right=0.0)


def _write_native_summary_csv(path, df):
    os.makedirs(os.path.dirname(os.fspath(path)), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _native_element_envelope(out_file, meta_df, output_csv, response_name):
    time_values, values = _read_native_recorder(out_file)
    if time_values is None or values is None or values.size == 0 or meta_df.empty:
        return pd.DataFrame()
    ele_tags = meta_df["eleTag"].astype(int).tolist()
    n_ele = len(ele_tags)
    if n_ele == 0:
        return pd.DataFrame()
    n_comp = values.shape[1] // n_ele
    if n_comp <= 0:
        return pd.DataFrame()
    values = values[:, :n_ele * n_comp].reshape(values.shape[0], n_ele, n_comp)
    peak = np.nanmax(np.abs(values), axis=0)
    rows = []
    for i, ele_tag in enumerate(ele_tags):
        base = meta_df.iloc[i].to_dict()
        base["response"] = response_name
        base["num_components"] = int(n_comp)
        base["max_abs"] = float(np.nanmax(peak[i, :]))
        if n_comp >= 12 and response_name == "localForce":
            base["max_abs_Fx"] = float(np.nanmax(np.abs(values[:, i, [0, 6]])))
            base["max_abs_Fy"] = float(np.nanmax(np.abs(values[:, i, [1, 7]])))
            base["max_abs_Fz"] = float(np.nanmax(np.abs(values[:, i, [2, 8]])))
            base["max_abs_Mx"] = float(np.nanmax(np.abs(values[:, i, [3, 9]])))
            base["max_abs_My"] = float(np.nanmax(np.abs(values[:, i, [4, 10]])))
            base["max_abs_Mz"] = float(np.nanmax(np.abs(values[:, i, [5, 11]])))
        for comp in range(n_comp):
            base[f"max_abs_c{comp + 1}"] = float(peak[i, comp])
        rows.append(base)
    df = pd.DataFrame(rows)
    _write_native_summary_csv(output_csv, df)
    return df



def _native_story_plan_extents(out_dir, story_df, base_df, frame_df, wall_df):
    """Return the common plan extents of every pair of adjacent floors."""
    element_frames = [df for df in (frame_df, wall_df) if not df.empty]
    if element_frames:
        element_meta = pd.concat(element_frames, ignore_index=True)
        floor_extents = element_meta.groupby("story").agg(
            x_min=("x_min", "min"),
            x_max=("x_max", "max"),
            y_min=("y_min", "min"),
            y_max=("y_max", "max"),
        )
    else:
        corner_path = os.path.join(os.fspath(out_dir), "meta_floor_corner_nodes.csv")
        if not os.path.exists(corner_path):
            raise FileNotFoundError("No element or floor-corner metadata is available")
        corner_df = pd.read_csv(corner_path)
        floor_extents = corner_df.groupby("story").agg(
            x_min=("x", "min"),
            x_max=("x", "max"),
            y_min=("y", "min"),
            y_max=("y", "max"),
        )

    base_extent = {
        "x_min": float(base_df["x"].min()),
        "x_max": float(base_df["x"].max()),
        "y_min": float(base_df["y"].min()),
        "y_max": float(base_df["y"].max()),
    }
    common_extents = []
    story_numbers = story_df["story"].astype(int).tolist()
    for story_col, story in enumerate(story_numbers):
        if story not in floor_extents.index:
            raise RuntimeError(f"No floor plan metadata for story {story}")
        upper = floor_extents.loc[story]
        lower = base_extent if story_col == 0 else floor_extents.loc[story_numbers[story_col - 1]]
        extent = {
            "x_min": max(float(upper["x_min"]), float(lower["x_min"])),
            "x_max": min(float(upper["x_max"]), float(lower["x_max"])),
            "y_min": max(float(upper["y_min"]), float(lower["y_min"])),
            "y_max": min(float(upper["y_max"]), float(lower["y_max"])),
        }
        if extent["x_min"] > extent["x_max"] or extent["y_min"] > extent["y_max"]:
            raise RuntimeError(f"Adjacent floors have no common plan area at story {story}")
        common_extents.append(extent)
    return common_extents


def _native_floor_torsion_kinematics(out_dir, story_df, base_df, frame_df, wall_df):
    """Recover rigid-diaphragm rotation and reference coordinates from native recorders."""
    corner_path = os.path.join(os.fspath(out_dir), "meta_floor_corner_nodes.csv")
    if not os.path.exists(corner_path):
        raise FileNotFoundError("meta_floor_corner_nodes.csv is required for torsional drift")
    corner_df = pd.read_csv(corner_path)

    time_ux, floor_ux = _read_native_recorder(os.path.join(os.fspath(out_dir), "node_floor_disp_UX.out"))
    time_uy, floor_uy = _read_native_recorder(os.path.join(os.fspath(out_dir), "node_floor_disp_UY.out"))
    time_cux, corner_ux = _read_native_recorder(os.path.join(os.fspath(out_dir), "node_floor_corner_disp_UX.out"))
    time_cuy, corner_uy = _read_native_recorder(os.path.join(os.fspath(out_dir), "node_floor_corner_disp_UY.out"))
    arrays = (time_ux, floor_ux, time_uy, floor_uy, time_cux, corner_ux, time_cuy, corner_uy)
    if any(value is None for value in arrays):
        raise FileNotFoundError("UX/UY floor and corner displacement recorders are required")
    if not (
        np.array_equal(time_ux, time_uy)
        and np.array_equal(time_ux, time_cux)
        and np.array_equal(time_ux, time_cuy)
    ):
        raise RuntimeError("Floor and corner recorder time columns do not match")

    n_time, n_story = floor_ux.shape
    if n_story != len(story_df) or floor_uy.shape[1] != n_story:
        raise RuntimeError("Floor recorder column count does not match story metadata")
    if corner_ux.shape[1] != len(corner_df) or corner_uy.shape[1] != len(corner_df):
        raise RuntimeError("Corner recorder column count does not match corner metadata")

    theta = np.zeros((n_time, n_story), dtype=float)
    master_xy = np.zeros((n_story, 2), dtype=float)
    has_saved_master_xy = {"control_x", "control_y"}.issubset(story_df.columns)

    for story_col, story in enumerate(story_df["story"].astype(int)):
        rows = corner_df[corner_df["story"].astype(int).eq(story)]
        cols = rows.index.to_numpy(dtype=int)
        x = rows["x"].to_numpy(dtype=float)
        y = rows["y"].to_numpy(dtype=float)
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        denominator = float(np.sum(x_centered**2 + y_centered**2))
        if denominator <= 0.0:
            raise RuntimeError(f"Cannot infer diaphragm rotation at story {story}")

        ux = corner_ux[:, cols]
        uy = corner_uy[:, cols]
        theta[:, story_col] = (
            np.sum(-(ux - np.mean(ux, axis=1, keepdims=True)) * y_centered, axis=1)
            + np.sum((uy - np.mean(uy, axis=1, keepdims=True)) * x_centered, axis=1)
        ) / denominator

        if has_saved_master_xy:
            master_xy[story_col] = [
                float(story_df.iloc[story_col]["control_x"]),
                float(story_df.iloc[story_col]["control_y"]),
            ]
            continue

        story_theta = theta[:, story_col]
        rotation_energy = float(len(rows) * np.sum(story_theta**2))
        if rotation_energy <= np.finfo(float).eps:
            raise RuntimeError(f"Insufficient torsional response to locate master node at story {story}")
        delta_ux = ux - floor_ux[:, [story_col]]
        delta_uy = uy - floor_uy[:, [story_col]]
        master_y = np.sum(
            story_theta[:, None] * (delta_ux + story_theta[:, None] * y[None, :])
        ) / rotation_energy
        master_x = -np.sum(
            story_theta[:, None] * (delta_uy - story_theta[:, None] * x[None, :])
        ) / rotation_energy
        master_xy[story_col] = [master_x, master_y]

    return {
        "time": time_ux,
        "theta": theta,
        "master_xy": master_xy,
        "common_extents": _native_story_plan_extents(
            out_dir, story_df, base_df, frame_df, wall_df
        ),
    }


def _native_torsion_story_drift(time_values, disp_values, story_heights, direction, kinematics):
    """Calculate maximum same-vertical-line drift at the two governing plan edges."""
    kin_time = np.asarray(kinematics["time"], dtype=float)
    theta = np.asarray(kinematics["theta"], dtype=float)
    if not np.array_equal(kin_time, time_values):
        interp_theta = np.empty((len(time_values), theta.shape[1]), dtype=float)
        for story_col in range(theta.shape[1]):
            interp_theta[:, story_col] = np.interp(time_values, kin_time, theta[:, story_col])
        theta = interp_theta
    master_xy = np.asarray(kinematics["master_xy"], dtype=float)
    extents = kinematics["common_extents"]
    n_time, n_story = disp_values.shape
    edge_drift = np.empty((n_time, n_story, 2), dtype=float)
    edge_coordinates = np.empty((n_story, 2), dtype=float)

    for story_col in range(n_story):
        if story_col == 0:
            lower_disp = np.zeros(n_time, dtype=float)
            lower_theta = np.zeros(n_time, dtype=float)
            lower_x = lower_y = 0.0
        else:
            lower_disp = disp_values[:, story_col - 1]
            lower_theta = theta[:, story_col - 1]
            lower_x, lower_y = master_xy[story_col - 1]
        upper_disp = disp_values[:, story_col]
        upper_theta = theta[:, story_col]
        upper_x, upper_y = master_xy[story_col]
        extent = extents[story_col]

        if str(direction).upper() == "UX":
            coordinates = np.asarray([extent["y_min"], extent["y_max"]], dtype=float)
            edge_delta = np.column_stack([
                upper_disp - upper_theta * (coordinate - upper_y)
                - (lower_disp - lower_theta * (coordinate - lower_y))
                for coordinate in coordinates
            ])
        else:
            coordinates = np.asarray([extent["x_min"], extent["x_max"]], dtype=float)
            edge_delta = np.column_stack([
                upper_disp + upper_theta * (coordinate - upper_x)
                - (lower_disp + lower_theta * (coordinate - lower_x))
                for coordinate in coordinates
            ])
        edge_drift[:, story_col, :] = edge_delta / float(story_heights[story_col])
        edge_coordinates[story_col, :] = coordinates

    max_ratio = np.empty(n_story, dtype=float)
    residual_ratio = np.empty(n_story, dtype=float)
    peak_time = np.empty(n_story, dtype=float)
    governing_coordinate = np.empty(n_story, dtype=float)
    dynamic_edge_ratio = np.empty(n_story, dtype=float)
    for story_col in range(n_story):
        values = edge_drift[:, story_col, :]
        flat_index = int(np.nanargmax(np.abs(values)))
        time_index, edge_index = np.unravel_index(flat_index, values.shape)
        edge_abs_at_peak = np.abs(values[time_index, :])
        edge_average = float(np.mean(edge_abs_at_peak))
        max_ratio[story_col] = float(edge_abs_at_peak[edge_index])
        residual_ratio[story_col] = float(values[-1, np.argmax(np.abs(values[-1, :]))])
        peak_time[story_col] = float(time_values[time_index])
        governing_coordinate[story_col] = float(edge_coordinates[story_col, edge_index])
        dynamic_edge_ratio[story_col] = (
            float(np.max(edge_abs_at_peak) / edge_average) if edge_average > 0.0 else np.nan
        )

    return {
        "max_ratio": max_ratio,
        "residual_ratio": residual_ratio,
        "peak_time": peak_time,
        "governing_coordinate": governing_coordinate,
        "dynamic_edge_ratio_at_peak": dynamic_edge_ratio,
    }


def postprocess_native_dynamic_recorders(
    out_dir,
    analysis_label=None,
    directions=("UX", "UY"),
    g=9800.0,
    plot=True,
    print_summary=True,
    process_element_responses=True,
):
    """Postprocess native recorder files into story/global/frame/wall summaries.

    Set process_element_responses=False to refresh floor response metrics without
    rereading large frame and wall response files. Existing element summaries
    are loaded when available.
    """
    from pathlib import Path as _Path

    out_dir = _Path(out_dir).expanduser().resolve()
    post_dir = out_dir / "postprocess"
    post_dir.mkdir(parents=True, exist_ok=True)

    story_df = pd.read_csv(out_dir / "meta_story.csv")
    base_df = pd.read_csv(out_dir / "meta_base_nodes.csv")
    frame_df = pd.read_csv(out_dir / "meta_frame_elements.csv") if (out_dir / "meta_frame_elements.csv").exists() else pd.DataFrame()
    wall_df = pd.read_csv(out_dir / "meta_wall_elements.csv") if (out_dir / "meta_wall_elements.csv").exists() else pd.DataFrame()
    story_ids = story_df["story"].astype(int).to_numpy()
    story_heights = story_df["story_height"].to_numpy(float)
    total_weight = float(story_df["weight"].sum()) if "weight" in story_df else np.nan

    torsion_kinematics = None
    torsion_warning = None
    try:
        torsion_kinematics = _native_floor_torsion_kinematics(
            out_dir, story_df, base_df, frame_df, wall_df
        )
    except Exception as err:
        torsion_warning = f"{type(err).__name__}: {err}"
        if print_summary:
            print(f"Torsional story drift is unavailable: {torsion_warning}")

    story_summary_frames = []
    global_rows = []
    time_by_direction = {}
    roof_disp_by_direction = {}
    base_shear_by_direction = {}

    for direction in directions:
        label, _dof = _native_direction_dof(direction)
        time_disp, disp_values = _read_native_recorder(out_dir / f"node_floor_disp_{label}.out")
        time_acc, accel_values = _read_native_recorder(out_dir / f"node_floor_accel_{label}.out")
        time_reac, reaction_values = _read_native_recorder(out_dir / f"node_base_reaction_{label}.out")
        if time_disp is None or disp_values is None:
            continue

        time_values = time_disp
        lower_disp = np.column_stack([np.zeros(disp_values.shape[0]), disp_values[:, :-1]])
        drift = (disp_values - lower_disp) / story_heights.reshape(1, -1)
        max_disp = np.nanmax(np.abs(disp_values), axis=0)
        residual_disp = disp_values[-1, :]
        max_drift = np.nanmax(np.abs(drift), axis=0)
        residual_drift = drift[-1, :]
        torsion_drift = None
        if torsion_kinematics is not None:
            try:
                torsion_drift = _native_torsion_story_drift(
                    time_values,
                    disp_values,
                    story_heights,
                    label,
                    torsion_kinematics,
                )
            except Exception as err:
                if print_summary:
                    print(f"{label} torsional story drift skipped: {type(err).__name__}: {err}")

        if accel_values is not None:
            if len(time_acc) != len(time_values) or not np.allclose(time_acc, time_values):
                accel_interp = np.empty((len(time_values), accel_values.shape[1]), dtype=float)
                for i in range(accel_values.shape[1]):
                    accel_interp[:, i] = np.interp(time_values, time_acc, accel_values[:, i])
                accel_values = accel_interp
            ground_accel = _native_ground_accel(out_dir, label, time_values)
            abs_accel = accel_values + ground_accel.reshape(-1, 1)
            pfa = np.nanmax(np.abs(abs_accel), axis=0)
            pga = float(np.nanmax(np.abs(ground_accel))) if ground_accel.size else np.nan
            accel_amp = pfa / pga if pga and np.isfinite(pga) and pga > 0.0 else np.full_like(pfa, np.nan)
        else:
            pfa = np.full(len(story_ids), np.nan)
            pga = np.nan
            accel_amp = np.full(len(story_ids), np.nan)

        if reaction_values is not None and reaction_values.size:
            base_shear = reaction_values.sum(axis=1)
            if len(time_reac) != len(time_values) or not np.allclose(time_reac, time_values):
                base_shear = np.interp(time_values, time_reac, base_shear)
        else:
            base_shear = np.full(len(time_values), np.nan)

        if accel_values is not None:
            mass = story_df["mass"].to_numpy(float)
            story_abs_accel = accel_values + _native_ground_accel(out_dir, label, time_values).reshape(-1, 1)
            story_inertia = story_abs_accel * mass.reshape(1, -1)
            story_shear = np.flip(np.cumsum(np.flip(story_inertia, axis=1), axis=1), axis=1)
            max_story_shear = np.nanmax(np.abs(story_shear), axis=0)
        else:
            max_story_shear = np.full(len(story_ids), np.nan)

        direction_story = story_df[["story", "elevation", "story_height", "control_node"]].copy()
        direction_story["direction"] = label
        direction_story["max_abs_disp_mm"] = max_disp
        direction_story["residual_disp_mm"] = residual_disp
        # Legacy max_drift columns remain aliases of the diaphragm-master method.
        direction_story["max_drift_ratio"] = max_drift
        direction_story["max_drift_percent"] = max_drift * 100.0
        direction_story["max_drift_master_ratio"] = max_drift
        direction_story["max_drift_master_percent"] = max_drift * 100.0
        direction_story["residual_drift_ratio"] = residual_drift
        direction_story["residual_drift_percent"] = residual_drift * 100.0
        direction_story["residual_drift_master_ratio"] = residual_drift
        direction_story["residual_drift_master_percent"] = residual_drift * 100.0
        if torsion_drift is not None:
            direction_story["max_drift_torsion_ratio"] = torsion_drift["max_ratio"]
            direction_story["max_drift_torsion_percent"] = torsion_drift["max_ratio"] * 100.0
            direction_story["residual_drift_torsion_ratio"] = torsion_drift["residual_ratio"]
            direction_story["residual_drift_torsion_percent"] = torsion_drift["residual_ratio"] * 100.0
            direction_story["torsion_peak_time_s"] = torsion_drift["peak_time"]
            direction_story["torsion_governing_coordinate_mm"] = torsion_drift["governing_coordinate"]
            direction_story["torsion_dynamic_edge_ratio_at_peak"] = torsion_drift["dynamic_edge_ratio_at_peak"]
            direction_story["torsion_drift_amplification"] = np.divide(
                torsion_drift["max_ratio"],
                max_drift,
                out=np.full_like(max_drift, np.nan),
                where=max_drift > 0.0,
            )
        else:
            for column in (
                "max_drift_torsion_ratio",
                "max_drift_torsion_percent",
                "residual_drift_torsion_ratio",
                "residual_drift_torsion_percent",
                "torsion_peak_time_s",
                "torsion_governing_coordinate_mm",
                "torsion_dynamic_edge_ratio_at_peak",
                "torsion_drift_amplification",
            ):
                direction_story[column] = np.nan
        direction_story["pfa_mm_s2"] = pfa
        direction_story["pfa_gal"] = pfa / 10.0
        direction_story["pfa_over_pga"] = accel_amp
        direction_story["max_abs_story_inertia_shear_N"] = max_story_shear
        story_summary_frames.append(direction_story)

        roof_disp = disp_values[:, -1]
        roof_disp_by_direction[label] = (time_values, roof_disp)
        base_shear_by_direction[label] = (time_values, base_shear)
        time_by_direction[label] = time_values
        roof_idx = int(np.nanargmax(np.abs(roof_disp)))
        drift_idx = int(np.nanargmax(max_drift))
        if torsion_drift is not None and np.any(np.isfinite(torsion_drift["max_ratio"])):
            torsion_drift_idx = int(np.nanargmax(torsion_drift["max_ratio"]))
        else:
            torsion_drift_idx = None
        global_row = {
            "analysis_label": analysis_label,
            "direction": label,
            "recorded_steps": int(len(time_values)),
            "roof_story": int(story_ids[-1]),
            "peak_abs_roof_disp_mm": float(np.nanmax(np.abs(roof_disp))),
            "peak_roof_disp_time_s": float(time_values[roof_idx]),
            "peak_drift_percent": float(np.nanmax(max_drift) * 100.0),
            "peak_drift_story": int(story_ids[drift_idx]),
            "peak_drift_master_percent": float(np.nanmax(max_drift) * 100.0),
            "peak_drift_master_story": int(story_ids[drift_idx]),
            "peak_drift_torsion_percent": (
                float(torsion_drift["max_ratio"][torsion_drift_idx] * 100.0)
                if torsion_drift_idx is not None else np.nan
            ),
            "peak_drift_torsion_story": (
                int(story_ids[torsion_drift_idx]) if torsion_drift_idx is not None else np.nan
            ),
            "peak_drift_torsion_time_s": (
                float(torsion_drift["peak_time"][torsion_drift_idx])
                if torsion_drift_idx is not None else np.nan
            ),
            "peak_pfa_gal": float(np.nanmax(pfa) / 10.0),
            "pga_gal": float(pga / 10.0) if np.isfinite(pga) else np.nan,
            "peak_abs_base_shear_N": float(np.nanmax(np.abs(base_shear))) if np.any(np.isfinite(base_shear)) else np.nan,
            "base_shear_coefficient": float(np.nanmax(np.abs(base_shear)) / total_weight) if total_weight and np.isfinite(total_weight) else np.nan,
        }
        global_rows.append(global_row)

    story_summary = pd.concat(story_summary_frames, ignore_index=True) if story_summary_frames else pd.DataFrame()
    global_summary = pd.DataFrame(global_rows)
    _write_native_summary_csv(post_dir / "story_response_summary.csv", story_summary)
    _write_native_summary_csv(post_dir / "global_response_summary.csv", global_summary)

    def _read_existing_post_csv(name):
        existing_path = post_dir / name
        return pd.read_csv(existing_path) if existing_path.exists() else pd.DataFrame()

    if process_element_responses:
        frame_local = _native_element_envelope(out_dir / "frame_localForce.out", frame_df, post_dir / "frame_localForce_envelope.csv", "localForce")
        frame_basic_force = _native_element_envelope(out_dir / "frame_basicForce.out", frame_df, post_dir / "frame_basicForce_envelope.csv", "basicForce")
        frame_basic_deformation = _native_element_envelope(out_dir / "frame_basicDeformation.out", frame_df, post_dir / "frame_basicDeformation_envelope.csv", "basicDeformation")
        frame_sec_force_i = _native_element_envelope(out_dir / "frame_section_force_i.out", frame_df, post_dir / "frame_section_force_i_envelope.csv", "sectionForce_i")
        frame_sec_force_j = _native_element_envelope(out_dir / "frame_section_force_j.out", frame_df, post_dir / "frame_section_force_j_envelope.csv", "sectionForce_j")
        frame_sec_defo_i = _native_element_envelope(out_dir / "frame_section_deformation_i.out", frame_df, post_dir / "frame_section_deformation_i_envelope.csv", "sectionDeformation_i")
        frame_sec_defo_j = _native_element_envelope(out_dir / "frame_section_deformation_j.out", frame_df, post_dir / "frame_section_deformation_j_envelope.csv", "sectionDeformation_j")
        wall_forces = _native_element_envelope(out_dir / "wall_forces.out", wall_df, post_dir / "wall_forces_envelope.csv", "wallForces")
        wall_stresses = _native_element_envelope(out_dir / "wall_stresses.out", wall_df, post_dir / "wall_stresses_envelope.csv", "wallStresses")
        shear_share = {}
        try:
            for direction in directions:
                label, _dof = _native_direction_dof(direction)
                shear_share[label] = postprocess_frame_wall_shear_share(
                    out_dir, direction=label, print_summary=False
                )
        except Exception as err:
            if print_summary:
                print(f"Frame-wall shear share postprocess skipped: {err}")
        if not frame_local.empty and "max_abs_Fy" in frame_local:
            frame_story = (
                frame_local.groupby(["story", "kind"], dropna=False)["max_abs_Fy"]
                .agg(["max", "mean", lambda s: s.quantile(0.95)])
                .reset_index()
                .rename(columns={"max": "max_Fy", "mean": "mean_Fy", "<lambda_0>": "p95_Fy"})
            )
            _write_native_summary_csv(post_dir / "frame_story_Fy_envelope.csv", frame_story)
        else:
            frame_story = pd.DataFrame()
    else:
        frame_local = _read_existing_post_csv("frame_localForce_envelope.csv")
        frame_basic_force = _read_existing_post_csv("frame_basicForce_envelope.csv")
        frame_basic_deformation = _read_existing_post_csv("frame_basicDeformation_envelope.csv")
        frame_sec_force_i = _read_existing_post_csv("frame_section_force_i_envelope.csv")
        frame_sec_force_j = _read_existing_post_csv("frame_section_force_j_envelope.csv")
        frame_sec_defo_i = _read_existing_post_csv("frame_section_deformation_i_envelope.csv")
        frame_sec_defo_j = _read_existing_post_csv("frame_section_deformation_j_envelope.csv")
        wall_forces = _read_existing_post_csv("wall_forces_envelope.csv")
        wall_stresses = _read_existing_post_csv("wall_stresses_envelope.csv")
        frame_story = _read_existing_post_csv("frame_story_Fy_envelope.csv")
        shear_share = {}
        for direction in directions:
            label, _dof = _native_direction_dof(direction)
            share_path = post_dir / f"frame_wall_shear_share_{label}.csv"
            if share_path.exists():
                shear_share[label] = pd.read_csv(share_path)

    if plot and not story_summary.empty:
        for label in sorted(story_summary["direction"].unique()):
            subset = story_summary[story_summary["direction"].eq(label)]
            fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
            axes[0].plot(subset["max_abs_disp_mm"], subset["story"], marker="o")
            axes[0].set_xlabel("Max floor disp (mm)")
            axes[0].set_ylabel("Story")
            axes[1].plot(
                subset["max_drift_master_percent"], subset["story"],
                marker="o", linestyle="--", label="Master node",
            )
            if subset["max_drift_torsion_percent"].notna().any():
                axes[1].plot(
                    subset["max_drift_torsion_percent"], subset["story"],
                    marker="o", label="Floor maximum (torsion)",
                )
            axes[1].set_xlabel("Max drift (%)")
            axes[1].legend(frameon=False)
            axes[2].plot(subset["pfa_gal"], subset["story"], marker="o")
            axes[2].set_xlabel("PFA (gal)")
            for ax in axes:
                ax.grid(True, linestyle="--", alpha=0.35)
            fig.suptitle(f"{analysis_label or ''} {label} story response envelope".strip())
            fig.tight_layout()
            fig.savefig(post_dir / f"story_envelope_{label}.png", dpi=200)
            plt.show()

        for label, (time_values, roof_disp) in roof_disp_by_direction.items():
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(time_values, roof_disp, lw=0.9, label=f"{label} roof displacement")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Roof displacement (mm)")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend()
            fig.tight_layout()
            fig.savefig(post_dir / f"roof_disp_time_{label}.png", dpi=200)
            plt.show()

    if print_summary:
        print("\nNative recorder postprocess:")
        print(f"input dir = {out_dir}")
        print(f"post dir = {post_dir}")
        if not global_summary.empty:
            print(global_summary.to_string(index=False))

    return {
        "out_dir": str(out_dir),
        "post_dir": str(post_dir),
        "story_summary": story_summary,
        "global_summary": global_summary,
        "frame_local": frame_local,
        "frame_basic_force": frame_basic_force,
        "frame_basic_deformation": frame_basic_deformation,
        "frame_section_force_i": frame_sec_force_i,
        "frame_section_force_j": frame_sec_force_j,
        "frame_section_deformation_i": frame_sec_defo_i,
        "frame_section_deformation_j": frame_sec_defo_j,
        "frame_story_Fy": frame_story,
        "wall_forces": wall_forces,
        "wall_stresses": wall_stresses,
        "shear_share": shear_share,
    }


def _native_postprocess_paths(out_dir):
    from pathlib import Path as _Path

    out_dir = _Path(out_dir).expanduser().resolve()
    return out_dir, out_dir / "postprocess"


def load_native_postprocess_results(out_dir):
    """Load postprocess CSV files from a completed native-recorder analysis."""
    out_dir, post_dir = _native_postprocess_paths(out_dir)

    def read_csv(name):
        path = post_dir / name
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

    shear_share = {}
    for direction in ("UX", "UY"):
        path = post_dir / f"frame_wall_shear_share_{direction}.csv"
        if path.exists():
            shear_share[direction] = pd.read_csv(path)

    return {
        "out_dir": str(out_dir),
        "post_dir": str(post_dir),
        "story_summary": read_csv("story_response_summary.csv"),
        "global_summary": read_csv("global_response_summary.csv"),
        "frame_local": read_csv("frame_localForce_envelope.csv"),
        "frame_basic_force": read_csv("frame_basicForce_envelope.csv"),
        "frame_basic_deformation": read_csv("frame_basicDeformation_envelope.csv"),
        "frame_section_force_i": read_csv("frame_section_force_i_envelope.csv"),
        "frame_section_force_j": read_csv("frame_section_force_j_envelope.csv"),
        "frame_section_deformation_i": read_csv("frame_section_deformation_i_envelope.csv"),
        "frame_section_deformation_j": read_csv("frame_section_deformation_j_envelope.csv"),
        "frame_story_Fy": read_csv("frame_story_Fy_envelope.csv"),
        "wall_forces": read_csv("wall_forces_envelope.csv"),
        "wall_stresses": read_csv("wall_stresses_envelope.csv"),
        "shear_share": shear_share,
    }


def _native_results_from_input(results_or_out_dir):
    if isinstance(results_or_out_dir, dict) and "story_summary" in results_or_out_dir:
        return results_or_out_dir
    return load_native_postprocess_results(results_or_out_dir)


def _native_plot_data_dir_from_results(results):
    from pathlib import Path as _Path

    post_dir = results.get("post_dir")
    if post_dir is None:
        out_dir = results.get("out_dir")
        if out_dir is None:
            return None
        post_dir = _Path(out_dir) / "postprocess"
    plot_data_dir = _Path(post_dir).expanduser().resolve() / "plot_data"
    plot_data_dir.mkdir(parents=True, exist_ok=True)
    return plot_data_dir


def _native_write_plot_data(results, filename, df, print_path=True, source_path=None):
    plot_data_dir = _native_plot_data_dir_from_results(results)
    if plot_data_dir is None:
        return None
    path = plot_data_dir / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    if print_path:
        if source_path is not None:
            print(f"Source data = {source_path}")
        print(f"Plot data = {path}")
    return path


def plot_native_story_metric(
    results_or_out_dir,
    metric,
    direction="UY",
    x_label=None,
    title=None,
    marker="o",
    ax=None,
    show=True,
    save_plot_data=True,
    print_data_path=True,
):
    """Plot one story-summary metric against story number."""
    results = _native_results_from_input(results_or_out_dir)
    story = results["story_summary"]
    if story.empty:
        raise ValueError("story_summary is empty. Run postprocess_native_dynamic_recorders first.")
    if metric not in story.columns:
        raise KeyError(f"{metric!r} is not in story_summary columns")

    direction = str(direction).upper()
    subset = story[story["direction"].astype(str).str.upper().eq(direction)].copy()
    if subset.empty:
        raise ValueError(f"No story summary rows found for direction {direction}")
    subset = subset.sort_values("story")

    if save_plot_data:
        export_cols = [
            col for col in (
                "direction", "story", "elevation", "story_height", "control_node", metric
            )
            if col in subset.columns
        ]
        source_path = None
        if results.get("post_dir") is not None:
            source_path = os.path.join(os.fspath(results["post_dir"]), "story_response_summary.csv")
        _native_write_plot_data(
            results,
            f"story_{metric}_{direction}.csv",
            subset[export_cols].copy(),
            print_path=print_data_path,
            source_path=source_path,
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 6.0))
    else:
        fig = ax.figure

    ax.plot(subset[metric], subset["story"], marker=marker, lw=1.5)
    ax.set_xlabel(x_label or metric)
    ax.set_ylabel("Story")
    ax.set_title(title or f"{direction} {metric}")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_native_story_displacement(
    results_or_out_dir,
    direction="UY",
    show=True,
    ax=None,
    save_plot_data=True,
    print_data_path=True,
):
    return plot_native_story_metric(
        results_or_out_dir,
        "max_abs_disp_mm",
        direction=direction,
        x_label="Max floor displacement (mm)",
        title=f"{str(direction).upper()} floor displacement envelope",
        show=show,
        ax=ax,
        save_plot_data=save_plot_data,
        print_data_path=print_data_path,
    )


def plot_native_story_drift(
    results_or_out_dir,
    direction="UY",
    method="both",
    show=True,
    ax=None,
    save_plot_data=True,
    print_data_path=True,
):
    """Plot master-node, torsion-inclusive, or both story-drift envelopes."""
    results = _native_results_from_input(results_or_out_dir)
    story = results["story_summary"]
    if story.empty:
        raise ValueError("story_summary is empty. Run postprocess_native_dynamic_recorders first.")
    direction = str(direction).upper()
    method = str(method).strip().lower()
    master_metric = (
        "max_drift_master_percent"
        if "max_drift_master_percent" in story.columns
        else "max_drift_percent"
    )
    torsion_metric = "max_drift_torsion_percent"

    if method in ("master", "center", "control"):
        return plot_native_story_metric(
            results,
            master_metric,
            direction=direction,
            x_label="Max interstory drift - diaphragm master (%)",
            title=f"{direction} master-node interstory drift envelope",
            show=show,
            ax=ax,
            save_plot_data=save_plot_data,
            print_data_path=print_data_path,
        )
    if method in ("torsion", "floor_max", "code"):
        if torsion_metric not in story.columns or not story[torsion_metric].notna().any():
            raise ValueError("Torsion-inclusive drift is unavailable. Re-run native postprocessing.")
        return plot_native_story_metric(
            results,
            torsion_metric,
            direction=direction,
            x_label="Max interstory drift - floor maximum (%)",
            title=f"{direction} torsion-inclusive interstory drift envelope",
            show=show,
            ax=ax,
            save_plot_data=save_plot_data,
            print_data_path=print_data_path,
        )
    if method != "both":
        raise ValueError("method must be 'master', 'torsion', or 'both'")

    subset = story[story["direction"].astype(str).str.upper().eq(direction)].copy()
    if subset.empty:
        raise ValueError(f"No story summary rows found for direction {direction}")
    subset = subset.sort_values("story")
    if torsion_metric not in subset.columns or not subset[torsion_metric].notna().any():
        raise ValueError("Torsion-inclusive drift is unavailable. Re-run native postprocessing.")

    export_cols = [
        column for column in (
            "direction", "story", "elevation", "story_height", "control_node",
            master_metric, torsion_metric, "torsion_drift_amplification",
            "torsion_dynamic_edge_ratio_at_peak", "torsion_peak_time_s",
            "torsion_governing_coordinate_mm",
        ) if column in subset.columns
    ]
    if save_plot_data:
        source_path = None
        if results.get("post_dir") is not None:
            source_path = os.path.join(os.fspath(results["post_dir"]), "story_response_summary.csv")
        _native_write_plot_data(
            results,
            f"story_drift_methods_{direction}.csv",
            subset[export_cols].copy(),
            print_path=print_data_path,
            source_path=source_path,
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.4, 6.2))
    else:
        fig = ax.figure
    ax.plot(
        subset[master_metric], subset["story"], marker="o", linestyle="--",
        lw=1.4, label="Diaphragm master node",
    )
    ax.plot(
        subset[torsion_metric], subset["story"], marker="o",
        lw=1.8, label="Floor maximum (torsion included)",
    )
    ax.set_xlabel("Max interstory drift (%)")
    ax.set_ylabel("Story")
    ax.set_title(f"{direction} interstory drift comparison")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_native_story_pfa(
    results_or_out_dir,
    direction="UY",
    show=True,
    ax=None,
    save_plot_data=True,
    print_data_path=True,
):
    return plot_native_story_metric(
        results_or_out_dir,
        "pfa_gal",
        direction=direction,
        x_label="Peak floor acceleration (gal)",
        title=f"{str(direction).upper()} PFA envelope",
        show=show,
        ax=ax,
        save_plot_data=save_plot_data,
        print_data_path=print_data_path,
    )


def plot_native_roof_displacement_time_history(
    out_dir,
    direction="UY",
    ax=None,
    show=True,
    save_plot_data=True,
    print_data_path=True,
):
    """Plot roof displacement time history from native node_floor_disp recorder."""
    out_dir, _post_dir = _native_postprocess_paths(out_dir)
    direction, _dof = _native_direction_dof(direction)
    source_path = out_dir / f"node_floor_disp_{direction}.out"
    time_values, values = _read_native_recorder(source_path)
    if time_values is None or values is None or values.size == 0:
        raise FileNotFoundError(f"Missing node_floor_disp_{direction}.out")
    roof_disp = values[:, -1]

    if save_plot_data:
        results = {"out_dir": str(out_dir), "post_dir": str(_post_dir)}
        data = pd.DataFrame({"time": time_values, "roof_disp_mm": roof_disp})
        _native_write_plot_data(
            results,
            f"roof_displacement_time_{direction}.csv",
            data,
            print_path=print_data_path,
            source_path=source_path,
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 4.0))
    else:
        fig = ax.figure
    ax.plot(time_values, roof_disp, lw=0.9, label=f"{direction} roof")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Roof displacement (mm)")
    ax.set_title(f"{direction} roof displacement time history")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def _native_story_time_history_columns(out_dir, value_count):
    story_path = os.path.join(os.fspath(out_dir), "meta_story.csv")
    if os.path.exists(story_path):
        story_df = pd.read_csv(story_path)
        if "story" in story_df:
            story_ids = story_df["story"].astype(int).tolist()
        else:
            story_ids = list(range(1, int(value_count) + 1))
    else:
        story_ids = list(range(1, int(value_count) + 1))

    n_values = int(value_count)
    if len(story_ids) != n_values:
        n_common = min(len(story_ids), n_values)
        print(
            "Story metadata count does not match recorder columns; "
            f"using first {n_common} columns."
        )
        story_ids = story_ids[:n_common]
    return story_ids


def extract_native_floor_displacement_time_history(
    out_dir,
    direction="UY",
    save_plot_data=True,
    print_data_path=True,
):
    """Export all-floor displacement histories from native node_floor_disp recorder."""
    out_dir, post_dir = _native_postprocess_paths(out_dir)
    direction, _dof = _native_direction_dof(direction)
    source_path = out_dir / f"node_floor_disp_{direction}.out"
    time_values, values = _read_native_recorder(source_path)
    if time_values is None or values is None or values.size == 0:
        raise FileNotFoundError(f"Missing node_floor_disp_{direction}.out")

    story_ids = _native_story_time_history_columns(out_dir, values.shape[1])
    values = values[:, :len(story_ids)]
    data = pd.DataFrame({"time": time_values})
    for i, story in enumerate(story_ids):
        data[f"story_{int(story):02d}_disp_mm"] = values[:, i]

    if save_plot_data:
        results = {"out_dir": str(out_dir), "post_dir": str(post_dir)}
        _native_write_plot_data(
            results,
            f"floor_displacement_time_{direction}.csv",
            data,
            print_path=print_data_path,
            source_path=source_path,
        )
    return data


def extract_native_floor_acceleration_time_history(
    out_dir,
    direction="UY",
    absolute=True,
    unit="gal",
    save_plot_data=True,
    print_data_path=True,
):
    """
    Export all-floor acceleration histories from native node_floor_accel recorder.

    With UniformExcitation, OpenSees node acceleration is relative to the moving
    base. absolute=True adds the recorded ground acceleration before export.
    """
    out_dir, post_dir = _native_postprocess_paths(out_dir)
    direction, _dof = _native_direction_dof(direction)
    source_path = out_dir / f"node_floor_accel_{direction}.out"
    time_values, values = _read_native_recorder(source_path)
    if time_values is None or values is None or values.size == 0:
        raise FileNotFoundError(f"Missing node_floor_accel_{direction}.out")

    accel_label = "absolute" if absolute else "relative"
    values = values.copy()
    if absolute:
        ground_accel = _native_ground_accel(out_dir, direction, time_values)
        values = values + ground_accel.reshape(-1, 1)

    unit_key = str(unit).strip().lower().replace("/", "_")
    if unit_key in ("gal", "cm_s2", "cm_s^2"):
        values = values / 10.0
        unit_suffix = "gal"
        col_suffix = "accel_gal"
    elif unit_key in ("mm_s2", "mm_s^2", "mm_sec2", "model"):
        unit_suffix = "mm_s2"
        col_suffix = "accel_mm_s2"
    else:
        raise ValueError("unit must be 'gal' or 'mm_s2'")

    story_ids = _native_story_time_history_columns(out_dir, values.shape[1])
    values = values[:, :len(story_ids)]
    data = pd.DataFrame({"time": time_values})
    for i, story in enumerate(story_ids):
        data[f"story_{int(story):02d}_{col_suffix}"] = values[:, i]

    if save_plot_data:
        results = {"out_dir": str(out_dir), "post_dir": str(post_dir)}
        _native_write_plot_data(
            results,
            f"floor_acceleration_time_{direction}_{accel_label}_{unit_suffix}.csv",
            data,
            print_path=print_data_path,
            source_path=source_path,
        )
    return data

def _native_filter_story_time_history_columns(data, value_suffix, stories=None):
    if stories is None:
        return [col for col in data.columns if col.startswith("story_") and col.endswith(value_suffix)]
    selected = []
    for story in stories:
        col = f"story_{int(story):02d}_{value_suffix}"
        if col not in data.columns:
            raise KeyError(f"Story column not found: {col}")
        selected.append(col)
    return selected


def _native_plot_story_time_history(data, value_columns, ylabel, title, ax=None, show=True):
    if not value_columns:
        raise ValueError("No story time-history columns were selected")
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
    else:
        fig = ax.figure

    cmap = plt.get_cmap("viridis")
    n_cols = len(value_columns)
    for i, col in enumerate(value_columns):
        color = cmap(i / max(n_cols - 1, 1))
        label = col.split("_")[1].lstrip("0") or "0"
        ax.plot(data["time"], data[col], lw=0.75, color=color, label=f"Story {label}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_native_floor_displacement_time_history(
    out_dir,
    direction="UY",
    stories=None,
    ax=None,
    show=True,
    save_plot_data=True,
    print_data_path=True,
):
    """Plot and export all-floor displacement time histories from native recorders."""
    direction, _dof = _native_direction_dof(direction)
    data = extract_native_floor_displacement_time_history(
        out_dir,
        direction=direction,
        save_plot_data=save_plot_data,
        print_data_path=print_data_path,
    )
    value_columns = _native_filter_story_time_history_columns(data, "disp_mm", stories=stories)
    return _native_plot_story_time_history(
        data,
        value_columns,
        ylabel="Floor displacement (mm)",
        title=f"{direction} floor displacement time histories",
        ax=ax,
        show=show,
    )


def plot_native_floor_acceleration_time_history(
    out_dir,
    direction="UY",
    absolute=True,
    unit="gal",
    stories=None,
    ax=None,
    show=True,
    save_plot_data=True,
    print_data_path=True,
):
    """Plot and export all-floor acceleration time histories from native recorders."""
    direction, _dof = _native_direction_dof(direction)
    data = extract_native_floor_acceleration_time_history(
        out_dir,
        direction=direction,
        absolute=absolute,
        unit=unit,
        save_plot_data=save_plot_data,
        print_data_path=print_data_path,
    )
    unit_key = str(unit).strip().lower().replace("/", "_")
    if unit_key in ("gal", "cm_s2", "cm_s^2"):
        value_suffix = "accel_gal"
        ylabel = "Floor acceleration (gal)"
    elif unit_key in ("mm_s2", "mm_s^2", "mm_sec2", "model"):
        value_suffix = "accel_mm_s2"
        ylabel = "Floor acceleration (mm/s^2)"
    else:
        raise ValueError("unit must be 'gal' or 'mm_s2'")
    value_columns = _native_filter_story_time_history_columns(data, value_suffix, stories=stories)
    accel_label = "absolute" if absolute else "relative"
    return _native_plot_story_time_history(
        data,
        value_columns,
        ylabel=ylabel,
        title=f"{direction} {accel_label} floor acceleration time histories",
        ax=ax,
        show=show,
    )

def plot_native_base_shear_time_history(
    out_dir,
    direction="UY",
    ax=None,
    show=True,
    save_plot_data=True,
    print_data_path=True,
):
    """Plot base shear time history from native base-reaction recorder."""
    out_dir, _post_dir = _native_postprocess_paths(out_dir)
    direction, _dof = _native_direction_dof(direction)
    source_path = out_dir / f"node_base_reaction_{direction}.out"
    time_values, values = _read_native_recorder(source_path)
    if time_values is None or values is None or values.size == 0:
        raise FileNotFoundError(f"Missing node_base_reaction_{direction}.out")
    base_shear = values.sum(axis=1)

    if save_plot_data:
        results = {"out_dir": str(out_dir), "post_dir": str(_post_dir)}
        data = pd.DataFrame({"time": time_values, "base_shear_N": base_shear})
        _native_write_plot_data(
            results,
            f"base_shear_time_{direction}.csv",
            data,
            print_path=print_data_path,
            source_path=source_path,
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.0, 4.0))
    else:
        fig = ax.figure
    ax.plot(time_values, base_shear, lw=0.9, label=f"{direction} base shear")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Base shear (N)")
    ax.set_title(f"{direction} base shear time history")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_native_base_shear_roof_displacement(
    out_dir,
    direction="UY",
    ax=None,
    show=True,
    save_plot_data=True,
    print_data_path=True,
):
    """Plot base shear versus roof displacement."""
    out_dir, _post_dir = _native_postprocess_paths(out_dir)
    direction, _dof = _native_direction_dof(direction)
    disp_source = out_dir / f"node_floor_disp_{direction}.out"
    reaction_source = out_dir / f"node_base_reaction_{direction}.out"
    time_u, disp = _read_native_recorder(disp_source)
    time_v, reaction = _read_native_recorder(reaction_source)
    if time_u is None or disp is None or time_v is None or reaction is None:
        raise FileNotFoundError(f"Missing displacement or base-reaction recorder for {direction}")
    roof_disp = disp[:, -1]
    base_shear = reaction.sum(axis=1)
    if len(time_u) != len(time_v) or not np.allclose(time_u, time_v):
        base_shear = np.interp(time_u, time_v, base_shear)

    if save_plot_data:
        results = {"out_dir": str(out_dir), "post_dir": str(_post_dir)}
        data = pd.DataFrame({
            "time": time_u,
            "roof_disp_mm": roof_disp,
            "base_shear_N": base_shear,
        })
        _native_write_plot_data(
            results,
            f"base_shear_roof_displacement_{direction}.csv",
            data,
            print_path=print_data_path,
            source_path=f"{disp_source}; {reaction_source}",
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.6))
    else:
        fig = ax.figure
    ax.plot(roof_disp, base_shear, lw=0.8)
    ax.set_xlabel("Roof displacement (mm)")
    ax.set_ylabel("Base shear (N)")
    ax.set_title(f"{direction} base shear - roof displacement")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_native_frame_story_fy(
    results_or_out_dir,
    kind=None,
    statistic="max_Fy",
    ax=None,
    show=True,
    save_plot_data=True,
    print_data_path=True,
):
    """Plot story-level frame local FY envelope from frame_localForce postprocess."""
    results = _native_results_from_input(results_or_out_dir)
    frame_story = results["frame_story_Fy"]
    if frame_story.empty:
        raise ValueError("frame_story_Fy is empty. Check frame_localForce recorder output.")
    if statistic not in frame_story.columns:
        raise KeyError(f"{statistic!r} is not in frame_story_Fy columns")
    data = frame_story.copy()
    if kind is not None:
        data = data[data["kind"].astype(str).str.lower().eq(str(kind).lower())]
    data = data.sort_values("story")

    if save_plot_data:
        source_path = None
        if results.get("post_dir") is not None:
            source_path = os.path.join(os.fspath(results["post_dir"]), "frame_story_Fy_envelope.csv")
        suffix = "all" if kind is None else str(kind)
        _native_write_plot_data(
            results,
            f"frame_story_Fy_{statistic}_{suffix}.csv",
            data.copy(),
            print_path=print_data_path,
            source_path=source_path,
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 6.0))
    else:
        fig = ax.figure
    for label, group in data.groupby("kind", dropna=False):
        ax.plot(group[statistic], group["story"], marker="o", lw=1.4, label=str(label))
    ax.set_xlabel(f"Frame {statistic} (N)")
    ax.set_ylabel("Story")
    ax.set_title("Frame local FY story envelope")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_native_wall_story_envelope(
    results_or_out_dir,
    response="forces",
    statistic="max_abs",
    ax=None,
    show=True,
    save_plot_data=True,
    print_data_path=True,
):
    """Plot wall element response envelope grouped by story."""
    results = _native_results_from_input(results_or_out_dir)
    key = "wall_stresses" if str(response).lower().startswith("stress") else "wall_forces"
    wall = results[key]
    if wall.empty:
        raise ValueError(f"{key} is empty. Check wall recorder support/output.")
    if statistic not in wall.columns:
        raise KeyError(f"{statistic!r} is not in {key} columns")
    data = (
        wall.groupby("story", dropna=False)[statistic]
        .agg(["max", "mean", lambda s: s.quantile(0.95)])
        .reset_index()
        .rename(columns={"<lambda_0>": "p95"})
        .sort_values("story")
    )

    if save_plot_data:
        source_name = "wall_stresses_envelope.csv" if key == "wall_stresses" else "wall_forces_envelope.csv"
        source_path = None
        if results.get("post_dir") is not None:
            source_path = os.path.join(os.fspath(results["post_dir"]), source_name)
        _native_write_plot_data(
            results,
            f"wall_story_{response}_{statistic}.csv",
            data.copy(),
            print_path=print_data_path,
            source_path=source_path,
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 6.0))
    else:
        fig = ax.figure
    ax.plot(data["max"], data["story"], marker="o", lw=1.4, label="max")
    ax.plot(data["p95"], data["story"], marker="s", lw=1.2, label="p95")
    ax.set_xlabel(f"Wall {response} {statistic}")
    ax.set_ylabel("Story")
    ax.set_title(f"Wall {response} story envelope")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def _native_column_bottom_elements(frame_df):
    if frame_df.empty:
        return frame_df.copy()
    data = frame_df[frame_df["kind"].astype(str).str.lower().eq("column")].copy()
    if data.empty:
        return data
    data["story"] = data["story"].astype(int)
    data["_z_sort"] = pd.to_numeric(data.get("z_mid"), errors="coerce")
    if "component" not in data:
        data["component"] = data["eleTag"].astype(str)
    idx = data.sort_values(["story", "component", "_z_sort", "eleTag"]).groupby(
        ["story", "component"], dropna=False
    ).head(1).index
    selected = data.loc[idx].sort_values(["story", "component"]).drop(columns=["_z_sort"])
    return selected


def postprocess_frame_wall_shear_share(
    out_dir,
    direction="UY",
    column_end="i",
    column_local_axis="auto",
    frame_sign=1.0,
    g=9800.0,
    print_summary=True,
):
    """
    Estimate frame-wall story shear sharing from native recorder files.

    No extra recorder is required. Total story shear is obtained from floor
    absolute accelerations and story masses. Frame shear is obtained by summing
    the bottom-end horizontal shear of one bottom column element for each column
    component in a story. Wall shear is taken as V_total - V_frame.

    For this 3D model, columns use geomTransf('PDelta', ..., 1, 0, 0). With
    vertical columns this gives local z approximately global X and local y
    approximately -global Y, so auto mapping uses Fz for UX and -Fy for UY.
    """
    from pathlib import Path as _Path

    out_dir, post_dir = _native_postprocess_paths(out_dir)
    post_dir.mkdir(parents=True, exist_ok=True)
    direction, _dof = _native_direction_dof(direction)
    frame_sign = float(frame_sign)

    story_df = pd.read_csv(out_dir / "meta_story.csv")
    frame_df = pd.read_csv(out_dir / "meta_frame_elements.csv")
    story_ids = story_df["story"].astype(int).to_numpy()
    story_mass = story_df["mass"].to_numpy(float)

    time_acc, floor_accel = _read_native_recorder(out_dir / f"node_floor_accel_{direction}.out")
    if time_acc is None or floor_accel is None:
        raise FileNotFoundError(f"Missing node_floor_accel_{direction}.out")
    ground_accel = _native_ground_accel(out_dir, direction, time_acc)
    abs_accel = floor_accel + ground_accel.reshape(-1, 1)
    story_inertia = abs_accel * story_mass.reshape(1, -1)
    total_story_shear = np.flip(np.cumsum(np.flip(story_inertia, axis=1), axis=1), axis=1)

    time_frame, frame_values = _read_native_recorder(out_dir / "frame_localForce.out")
    if time_frame is None or frame_values is None:
        raise FileNotFoundError("Missing frame_localForce.out")

    frame_ele_tags = frame_df["eleTag"].astype(int).tolist()
    n_ele = len(frame_ele_tags)
    if n_ele == 0:
        raise ValueError("meta_frame_elements.csv contains no frame elements")
    n_comp = frame_values.shape[1] // n_ele
    if n_comp < 12:
        raise ValueError(f"frame_localForce component count per element is {n_comp}, expected at least 12")
    frame_values = frame_values[:, :n_ele * n_comp].reshape(frame_values.shape[0], n_ele, n_comp)

    selected_cols = _native_column_bottom_elements(frame_df)
    if selected_cols.empty:
        raise ValueError("No column elements found in meta_frame_elements.csv")
    ele_to_index = {int(ele): i for i, ele in enumerate(frame_ele_tags)}

    end = str(column_end).strip().lower()
    if end not in ("i", "j"):
        raise ValueError("column_end must be 'i' or 'j'")
    offset = 0 if end == "i" else 6

    axis = str(column_local_axis).strip().lower()
    if axis == "auto":
        if direction == "UX":
            component_index = offset + 2
            local_component = "Fz"
            auto_sign = 1.0
        else:
            component_index = offset + 1
            local_component = "Fy"
            auto_sign = -1.0
    elif axis in ("fy", "local_y", "y"):
        component_index = offset + 1
        local_component = "Fy"
        auto_sign = 1.0
    elif axis in ("fz", "local_z", "z"):
        component_index = offset + 2
        local_component = "Fz"
        auto_sign = 1.0
    else:
        raise ValueError("column_local_axis must be auto, Fy, or Fz")

    frame_story_shear = np.zeros((len(time_frame), len(story_ids)), dtype=float)
    story_to_col_count = {int(story): 0 for story in story_ids}
    for row in selected_cols.itertuples(index=False):
        story = int(getattr(row, "story"))
        ele_tag = int(getattr(row, "eleTag"))
        if story not in story_to_col_count or ele_tag not in ele_to_index:
            continue
        story_index = int(np.where(story_ids == story)[0][0])
        ele_index = ele_to_index[ele_tag]
        frame_story_shear[:, story_index] += frame_values[:, ele_index, component_index] * auto_sign * frame_sign
        story_to_col_count[story] += 1

    if len(time_frame) != len(time_acc) or not np.allclose(time_frame, time_acc):
        interp = np.empty((len(time_acc), frame_story_shear.shape[1]), dtype=float)
        for j in range(frame_story_shear.shape[1]):
            interp[:, j] = np.interp(time_acc, time_frame, frame_story_shear[:, j])
        frame_story_shear = interp

    wall_story_shear = total_story_shear - frame_story_shear
    rows = []
    for j, story in enumerate(story_ids):
        total = total_story_shear[:, j]
        frame = frame_story_shear[:, j]
        wall = wall_story_shear[:, j]
        idx_total = int(np.nanargmax(np.abs(total)))
        total_peak = float(np.nanmax(np.abs(total)))
        frame_peak = float(np.nanmax(np.abs(frame)))
        wall_peak = float(np.nanmax(np.abs(wall)))
        denom = total_peak if total_peak > 0.0 else np.nan
        rows.append({
            "story": int(story),
            "direction": direction,
            "time_at_peak_total_s": float(time_acc[idx_total]),
            "peak_total_shear_N": total_peak,
            "peak_frame_shear_N": frame_peak,
            "peak_wall_shear_N": wall_peak,
            "frame_share": float(frame_peak / denom) if np.isfinite(denom) else np.nan,
            "wall_share": float(wall_peak / denom) if np.isfinite(denom) else np.nan,
            "signed_total_at_peak_N": float(total[idx_total]),
            "signed_frame_at_peak_N": float(frame[idx_total]),
            "signed_wall_at_peak_N": float(wall[idx_total]),
            "frame_share_at_peak_total": float(abs(frame[idx_total]) / denom) if np.isfinite(denom) else np.nan,
            "wall_share_at_peak_total": float(abs(wall[idx_total]) / denom) if np.isfinite(denom) else np.nan,
            "num_column_components": int(story_to_col_count.get(int(story), 0)),
            "column_end": end,
            "local_component": local_component,
            "local_axis_mapping": "auto: UX=Fz, UY=-Fy for vertical columns with vecxz=(1,0,0)" if axis == "auto" else axis,
        })

    summary = pd.DataFrame(rows)
    summary_path = post_dir / f"frame_wall_shear_share_{direction}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    time_path = post_dir / f"frame_wall_shear_time_{direction}.csv"
    time_data = {"time": time_acc}
    for j, story in enumerate(story_ids):
        time_data[f"total_s{int(story)}"] = total_story_shear[:, j]
        time_data[f"frame_s{int(story)}"] = frame_story_shear[:, j]
        time_data[f"wall_s{int(story)}"] = wall_story_shear[:, j]
    pd.DataFrame(time_data).to_csv(time_path, index=False, encoding="utf-8-sig")

    if print_summary:
        print(f"Frame-wall shear share summary = {summary_path}")
        print(f"Frame-wall shear time history = {time_path}")
        print("Assumption:", rows[0]["local_axis_mapping"] if rows else "")

    return summary


def plot_native_shear_share(
    results_or_out_dir,
    direction="UY",
    share_at_peak_total=False,
    ax=None,
    show=True,
    save_plot_data=True,
    print_data_path=True,
):
    """Plot frame and wall story shear sharing ratios."""
    results = _native_results_from_input(results_or_out_dir)
    direction, _dof = _native_direction_dof(direction)
    shear_share = results.get("shear_share", {})
    if direction in shear_share:
        data = shear_share[direction].copy()
    else:
        out_dir = results.get("out_dir")
        if out_dir is None:
            raise ValueError("No out_dir is available to load shear share results")
        data = postprocess_frame_wall_shear_share(out_dir, direction=direction, print_summary=False)
    if data.empty:
        raise ValueError(f"No shear share data found for {direction}")
    data = data.sort_values("story")

    frame_col = "frame_share_at_peak_total" if share_at_peak_total else "frame_share"
    wall_col = "wall_share_at_peak_total" if share_at_peak_total else "wall_share"

    if save_plot_data:
        source_path = None
        if results.get("post_dir") is not None:
            source_path = os.path.join(os.fspath(results["post_dir"]), f"frame_wall_shear_share_{direction}.csv")
        _native_write_plot_data(
            results,
            f"frame_wall_shear_share_{direction}.csv",
            data[["story", "direction", frame_col, wall_col, "peak_total_shear_N", "peak_frame_shear_N", "peak_wall_shear_N"]].copy(),
            print_path=print_data_path,
            source_path=source_path,
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.4, 6.0))
    else:
        fig = ax.figure
    ax.plot(data[frame_col] * 100.0, data["story"], marker="o", lw=1.4, label="Frame")
    ax.plot(data[wall_col] * 100.0, data["story"], marker="s", lw=1.4, label="Wall")
    ax.set_xlabel("Shear share (%)")
    ax.set_ylabel("Story")
    ax.set_title(f"{direction} frame-wall shear sharing")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def run_transient_analysis(
    dt,
    n_step,
    tol=1.0e-5,
    max_iter=40,
    print_flag=0,
    test_type="NormDispIncr",
    algorithm_type="Newton",
    min_substep_factor=8.0,
    max_trials_per_step=20,
    progress_file=None,
    response_recorder=None,
    record_interval=1,
    record_last_step=True,
    desc="Native transient",
    ncols=90,
    mininterval=5.0,
):
    """
    Run OpenSees transient analysis with conservative algorithm fallbacks.

    The model, excitation pattern, integrator and analysis object should already be
    defined before calling this function. The function only controls test(),
    algorithm(), analyze(), step cutting and the progress bar.
    """
    import os as _os
    from pathlib import Path as _Path
    from contextlib import redirect_stdout as _redirect_stdout
    from contextlib import redirect_stderr as _redirect_stderr

    try:
        from tqdm import tqdm as _tqdm
    except Exception:
        _tqdm = None

    dt = float(dt)
    n_step = int(n_step)
    record_interval = max(1, int(record_interval))
    min_step_dt = dt / float(min_substep_factor)

    progress_path = None
    if progress_file is not None:
        progress_path = _Path(progress_file)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text("event,step,time,ok,algorithm,dt\n", encoding="utf-8")

    def write_progress(event, step_num, ok=0, algo="", step_dt=None):
        if progress_path is None:
            return
        if step_dt is None:
            step_dt = dt
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(f"{event},{step_num},{ops.getTime():.8f},{ok},{algo},{step_dt:.8e}\n")

    def silent_analyze(n=1, step_dt=None):
        if step_dt is None:
            step_dt = dt
        with open(_os.devnull, "w") as devnull:
            with _redirect_stdout(devnull), _redirect_stderr(devnull):
                return ops.analyze(n, step_dt)

    def reset_solution(algo=algorithm_type, iters=max_iter):
        ops.test(test_type, tol, int(iters), int(print_flag))
        if algo == "Newton":
            ops.algorithm("Newton")
        elif algo == "KrylovNewton":
            ops.algorithm("KrylovNewton")
        elif algo == "NewtonLineSearch_Bisection":
            ops.algorithm("NewtonLineSearch", "-type", "Bisection")
        elif algo == "NewtonLineSearch_Secant":
            ops.algorithm("NewtonLineSearch", "-type", "Secant")
        elif algo == "NewtonLineSearch":
            ops.algorithm("NewtonLineSearch")
        else:
            raise ValueError(f"Unknown algorithm setting: {algo}")

    def try_algorithms_for_step(step_num, step_dt):
        fallback_plan = [
            ("Newton", "Newton", max_iter),
            ("Newton_more_iter", "Newton", max(max_iter, 100)),
            ("KrylovNewton", "KrylovNewton", max(max_iter, 80)),
            ("NewtonLineSearch_Bisection", "NewtonLineSearch_Bisection", max(max_iter, 80)),
            ("NewtonLineSearch_Secant", "NewtonLineSearch_Secant", max(max_iter, 80)),
        ]

        last_ok = -1
        for label, algo, iters in fallback_plan:
            reset_solution(algo, iters)
            write_progress("try_start", step_num, last_ok, label, step_dt)
            ok = silent_analyze(1, step_dt)
            write_progress("try_end", step_num, ok, label, step_dt)
            if ok == 0:
                reset_solution()
                return 0, label
            last_ok = ok

        reset_solution()
        return last_ok, "failed"

    def try_one_step(step_num):
        step_dt = dt
        trials = 0
        while True:
            ok, label = try_algorithms_for_step(step_num, step_dt)
            trials += 1
            if ok == 0:
                return 0, label
            if step_dt <= min_step_dt or trials >= int(max_trials_per_step):
                write_progress("stop_min_dt", step_num, ok, label, step_dt)
                return ok, label
            step_dt *= 0.5
            write_progress("cut_dt", step_num, ok, label, step_dt)

    reset_solution()

    ok = 0
    last_algo = "Newton"
    analysis_ok = True
    failed_step = None
    recorded_steps = 0

    if _tqdm is not None:
        pbar = _tqdm(total=n_step, desc=desc, ncols=ncols, mininterval=mininterval)
    else:
        pbar = None

    try:
        for step_num in range(1, n_step + 1):
            ok, last_algo = try_one_step(step_num)
            if ok != 0:
                analysis_ok = False
                failed_step = step_num
                msg = f"Step {step_num} failed, time = {ops.getTime():.6f} s, last algorithm = {last_algo}"
                if pbar is not None:
                    pbar.write(msg)
                else:
                    print(msg)
                break

            if response_recorder is not None:
                should_record = (step_num % record_interval == 0) or (record_last_step and step_num == n_step)
                if should_record:
                    response_recorder()
                    recorded_steps += 1

            if pbar is not None:
                if step_num % 10 == 0:
                    pbar.set_postfix_str(f"time={ops.getTime():.2f}s, algo={last_algo}", refresh=False)
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    final_time = float(ops.getTime())
    if analysis_ok:
        print(f"Native transient finished successfully: {n_step} steps, final time = {final_time:.6f} s")
    else:
        print(f"Native transient stopped at step {failed_step}, current time = {final_time:.6f} s")

    if progress_path is not None:
        print(f"Progress log saved in: {progress_path}")

    return {
        "ok": int(ok),
        "analysis_ok": bool(analysis_ok),
        "failed_step": failed_step,
        "final_time": final_time,
        "last_algorithm": last_algo,
        "n_step_requested": n_step,
        "dt": dt,
        "progress_file": str(progress_path) if progress_path is not None else None,
        "record_interval": record_interval,
        "recorded_steps": recorded_steps,
    }


