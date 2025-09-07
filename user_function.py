import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from openseespy.opensees import section
import openseespy.opensees as ops
import os
import matplotlib.pyplot as plt
import opstool as opst  # 一种新型的纤维模型建模方式
import math
import opsvis as opsv
from tqdm import tqdm
import sys
from contextlib import redirect_stderr
import numpy as np 
import pandas as pd
import time
from openseespy.opensees import *
#==================================
# ========== 打印函数 ============
#==================================

def print_all_material_tags():
    # 打印所有材料编号和描述
    print("=== 所有材料编号 ===")
    for key, (tag, desc) in MaterialTags.items():
        print(f"{key:12} = {tag:<2}  # {desc}")

def print_all_section_tags():
    # 打印所有截面编号和描述
    print("=== 所有截面编号 ===")
    for key, (tag, desc) in SectionTags.items():
        print(f"{key:20} = {tag:<2}  # {desc}")

def search_tags(keyword):
    # 搜索材料和截面编号
    if not isinstance(keyword, str):
        keyword = str(keyword)

    print(f"=== 查询关键字：'{keyword}' 的结果 ===")
    found = False

    for key, (tag, desc) in MaterialTags.items():
        if keyword in key or keyword in desc:
            print(f"材料编号: {key:12} = {tag:<2}  # {desc}")
            found = True

    for key, (tag, desc) in SectionTags.items():
        if keyword in key or keyword in desc:
            print(f"截面编号: {key:20} = {tag:<2}  # {desc}")
            found = True

    if not found:
        print("❌ 未找到匹配项。")

def print_unit_table():
    # This function prints a table comparing IKS (inch-kip-second) units with SI (International System of Units) units.
    print("=== IKS（inch-kip-second）单位制 与 SI 单位制详细对照 ===")
    print("┌────────────────────┬─────────────┬────────────┬────────────────────────────┐")
    print("│ 物理量             │ IKS单位      │ SI单位     │ 换算关系                    │")
    print("├────────────────────┼──────────── ┼────────────┼────────────────────────────┤")
    print("│ 长度               │ inch        │ meter (m)  │ 1 in = 0.0254 m            │")
    print("│ 力                 │ kip         │ newton (N) │ 1 kip = 4448.22 N          │")
    print("│ 应力               │ ksi         │ MPa        │ 1 ksi ≈ 6.895 MPa          │")
    print("│ 时间               │ sec         │ second (s) │ 相同                       │")
    print("│ 质量               │ kip·sec²/in │ kg         │ ≈ 14.594 kg                │")
    print("│ 密度               │ kip·sec²/in⁴│ kg/m³      │ ≈ 3.6e7 kg/m³              │")
    print("│ 单位重             │ kip/in³     │ N/m³       │ 1 kip/in³ ≈ 2.527e7 N/m³   │")
    print("│ 加速度             │ in/sec²     │ m/s²       │ 1 in/s² = 0.0254 m/s²      │")
    print("│ 刚度               │ kip/in      │ N/m        │ ≈ 175126 N/m               │")
    print("│ 力矩               │ kip·inch    │ N·m        │ 1 kip·in ≈ 113 N·m         │")
    print("│ 能量               │ kip·inch    │ Joule (J)  │ 1 kip·in ≈ 113 J           │")
    print("│ 应变               │ —           │ —          │ 无量纲，相同                │")
    print("│ 应变率             │ 1/sec       │ 1/s        │ 相同                        │")
    print("└────────────────────┴────────────┴────────────┴─────────────────────────────┘\n")


# ==================================
# ========== 工具函数 ============
# ==================================

def GeneratePeaks(Dmax, DincrStatic=0.01, CycleType='Full', Fact=1.0):
    """
    根据峰值位移、增量和循环类型生成位移步列表
    并在当前目录下生成 IDstep.txt 文件
    """
    iDstep = [0.0]
    Disp = 0.0
    Dmax_scaled = Dmax * Fact

    if Dmax_scaled == 0:
        iDstep = [0.0]
    else:
        dx = DincrStatic if Dmax_scaled > 0 else -DincrStatic
        NstepsPeak = int(abs(Dmax_scaled) / DincrStatic)
        if NstepsPeak == 0: NstepsPeak = 1  # 至少走一步

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



def GetPeakStep(iDmax1, all_steps):
    """
    从已经生成的 all_steps 中获取每个峰值位移对应的全局加载步索引
    iDmax1: 峰值位移列表
    all_steps: [(目标位移, 增量)] 的加载序列
    """
    peak_step_indices = []

    # 只取 all_steps 中的目标位移部分
    disp_targets = [step[0] for step in all_steps]

    for Dmax in iDmax1:
        # 找到 Dmax 在目标位移中的第一次出现
        try:
            peak_index = disp_targets.index(Dmax)
        except ValueError:
            # 如果 Dmax 不精确匹配，考虑用误差范围查找
            peak_index = min(
                range(len(disp_targets)),
                key=lambda i: abs(disp_targets[i] - Dmax)
            )

        peak_step_indices.append((Dmax, peak_index))

    print(f"总共分析步数为: {len(all_steps)}")
    print("\n各峰值位移在总路径中出现的位置（单位: 第几步）:")
    for i, (dmax, step_id) in enumerate(peak_step_indices, 1):
        dmax_mm = dmax * 25.4
        print(f"第 {i:2d} 个峰值 {dmax_mm:.1f} mm 出现在第 {step_id:4d} 步")


# 封装静默分析函数
def silent_analyze(n=1):
    devnull = open(os.devnull, 'w')
    with redirect_stderr(devnull):
        return ops.analyze(n)
    

def wall_mesh(
    wall_width, wall_height,
    x_coords=None, y_coords=None,
    num_ele_x=None, num_ele_y=None,
    edge_zone_width=0.0,
    sec_tag_edge=1, sec_tag_wall=2,
    shell_type="ShellNLDKGQ"
):
    """
    剪力墙网格生成函数
    支持：
    - 传入 x_coords, y_coords 生成不规则网格
    - 传入 num_ele_x, num_ele_y 自动生成均匀网格
    """
    start_time = time.time()

    # 自动计算坐标
    if x_coords is None:
        if num_ele_x is None:
            raise ValueError("必须提供 x_coords 或 num_ele_x")
        x_coords = np.linspace(0, wall_width, num_ele_x + 1).tolist()

    if y_coords is None:
        if num_ele_y is None:
            raise ValueError("必须提供 y_coords 或 num_ele_y")
        y_coords = np.linspace(0, wall_height, num_ele_y + 1).tolist()

    num_node_x = len(x_coords)
    num_node_y = len(y_coords)
    num_ele_x = num_node_x - 1
    num_ele_y = num_node_y - 1

    # 自动获取当前最大节点号和单元号
    existing_nodes = getNodeTags()
    existing_eles = getEleTags()
    start_node_tag = max(existing_nodes) if existing_nodes else 0
    start_ele_tag = max(existing_eles) if existing_eles else 0

    # ----------------------------
    # 节点生成
    # ----------------------------
    node_tag = start_node_tag + 1
    node_grid = []
    for y_coord in y_coords:
        row_nodes = []
        for x_coord in x_coords:
            node(node_tag, x_coord, y_coord, 0.0)
            row_nodes.append(node_tag)
            node_tag += 1
        node_grid.append(row_nodes)

    # ----------------------------
    # 单元生成
    # ----------------------------
    ele_tag = start_ele_tag + 1
    count_edge_ele = 0
    count_wall_ele = 0

    for j in range(num_ele_y):
        for i in range(num_ele_x):
            ele_start_x = x_coords[i]

            # 判断边缘区
            if ele_start_x < edge_zone_width or x_coords[i+1] > (wall_width - edge_zone_width):
                section_id = sec_tag_edge
                count_edge_ele += 1
            else:
                section_id = sec_tag_wall
                count_wall_ele += 1

            # 节点编号 (ShellNLDKGQ: BL, BR, TR, TL)
            node_bl = node_grid[j][i]
            node_br = node_grid[j][i+1]
            node_tl = node_grid[j+1][i]
            node_tr = node_grid[j+1][i+1]

            element(shell_type, ele_tag, node_bl, node_br, node_tr, node_tl, section_id)
            ele_tag += 1

    # ----------------------------
    # 打印模型统计信息
    # ----------------------------
    print("-" * 30)
    print("模型统计信息:")
    print(f"墙体尺寸 (WxH): {wall_width} m x {wall_height} m")
    print(f"网格划分 (XxY): {num_ele_x} x {num_ele_y}")
    print(f"边缘区域宽度: {edge_zone_width} m")
    print(f"分配给边缘截面 (ID={sec_tag_edge}) 的单元数: {count_edge_ele}")
    print(f"分配给墙身截面 (ID={sec_tag_wall}) 的单元数: {count_wall_ele}")
    print(f"总单元数: {ele_tag - start_ele_tag - 1}")
    print(f"节点编号范围: {start_node_tag + 1} ~ {node_tag - 1}")
    print(f"单元编号范围: {start_ele_tag + 1} ~ {ele_tag - 1}")
    print(f"模型创建耗时: {time.time() - start_time:.2f} 秒")
    print("-" * 30)

    # ----------------------------
    # 返回信息
    # ----------------------------
    return {
    "added_nodes": (start_node_tag + 1, node_tag - 1),
    "added_elements": (start_ele_tag + 1, ele_tag - 1),
    "node_grid": node_grid,
    "bottom_nodes": node_grid[0],
    "top_nodes": node_grid[-1],
    "left_edge_nodes": [row[0] for row in node_grid],
    "right_edge_nodes": [row[-1] for row in node_grid],
    "control_node": node_grid[-1][len(x_coords)//2],  # 顶部中点
    "edge_elements": count_edge_ele,
    "wall_elements": count_wall_ele,
    "elapsed_time": time.time() - start_time
    }

def add_vertical_rebars(node_grid, cols, area, mat_tag, start_ele_tag=None):
    """
    根据 wall_mesh 返回的 node_grid 自动生成纵向 truss 钢筋
    cols: 纵筋所在的列索引（list[int]）
    area: 钢筋截面面积 (m²)
    mat_tag: 钢筋材料号
    start_ele_tag: 起始单元编号（默认接着已有的最大编号）
    """
    if start_ele_tag is None:
        existing_eles = getEleTags()
        start_ele_tag = max(existing_eles) if existing_eles else 0

    ele_tag = start_ele_tag + 1
    for col in cols:
        for row in range(len(node_grid) - 1):
            n1 = node_grid[row][col]
            n2 = node_grid[row + 1][col]
            element("truss", ele_tag, n1, n2, area, mat_tag)
            ele_tag += 1

    print(f"已生成纵向钢筋 truss 单元 {ele_tag - start_ele_tag - 1} 根")
    return ele_tag - 1

# ==================================
# ========== 可视化函数 ============
# ==================================
def plot_model(show_node=True, show_ele=True,
                           node_size=12, ele_size=12, show_local_axes=True):
    """
    用 opstool.vis.pyvista 绘制模型，可选显示节点号/单元号
    """
    from openseespy.opensees import getNodeTags, nodeCoord, getEleTags, eleNodes

    # 设置为 notebook=True，避免阻塞
    opst.vis.pyvista.set_plot_props(notebook=True)
    fig = opst.vis.pyvista.plot_model(show_local_axes=show_local_axes)

    # 显示节点编号
    if show_node:
        node_tags = getNodeTags()
        coords = np.array([nodeCoord(tag) for tag in node_tags])
        if coords.shape[1] == 2:
            coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
        fig.add_point_labels(coords,
                             [str(tag) for tag in node_tags],
                             font_size=node_size,
                             point_color="black")

    # 显示单元编号
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



def plt_layershell_section(section_data):
    """
    可视化分层壳截面。
    所有绘图所需信息（图例、颜色、钢筋标识）都从 section_data 中自动读取。
    """
    layers = section_data["layers"]
    wall_width = section_data["width"]
    # 从 section_data 中获取材料属性，如果不存在则使用空字典
    mat_props = section_data.get("material_properties", {})

    fig, ax = plt.subplots(figsize=(10, 4))
    current_y = 0
    used_tags = set()
    offset_toggle = True

    for layer in layers:
        matTag = layer["matTag"]
        thickness = layer["thickness"]
        
        # 自动识别钢筋：检查 'rebar' 键，默认为 False
        rebar = layer.get("rebar", False)

        # 从 mat_props 获取颜色和名称
        mat_info = mat_props.get(matTag, {}) # 获取该材料的信息字典
        facecolor = mat_info.get("color", 'gray') # 获取颜色，默认为灰色
        mat_name = mat_info.get("name", f"Material {matTag}") # 获取名称

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

        # 标签位置偏移
        y_center = current_y + thickness / 2
        if rebar:
            offset = 0.002 # 调整偏移量以适应截面总厚度
            y_text = y_center + offset if offset_toggle else y_center - offset
            offset_toggle = not offset_toggle
        else:
            y_text = y_center

        label = f'{mat_name} ({thickness*1000:.2f} mm)'
        ax.text(
            wall_width / 2, y_text,
            label,
            ha='center', va='center',
            fontsize=8, color='black'
        )

        used_tags.add(matTag)
        current_y += thickness

    # 坐标设置
    ax.set_xlim(0, wall_width)
    ax.set_ylim(0, current_y)
    ax.set_aspect('auto') # 自动调整纵横比，避免图像被过度拉伸
    ax.invert_yaxis()
    ax.set_xlabel("Wall Length (m)")
    ax.set_ylabel("Wall Thickness (m)")
    ax.set_title(f"LayeredShell Section tag={section_data['tag']}")
    ax.grid(True, linestyle='--', alpha=0.3)

    # 创建图例
    legend_handles = []
    for tag in sorted(list(used_tags)): # 排序以保证图例顺序一致
        mat_info = mat_props.get(tag, {})
        color = mat_info.get("color", "gray")
        name = mat_info.get("name", f"Material {tag}")
        legend_handles.append(Patch(facecolor=color, edgecolor='black', label=name))
        
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    plt.show()


# ============================================================
# 4. 自动创建 LayeredShell 截面 (简化调用)
# ============================================================
def create_layershell(section_data):
    """
    输入一个包含所有信息的section_data字典，自动生成OpenSees命令并可视化截面。
    新结构:
    {
        "tag": int,
        "width": float,
        "layers": [
            {"matTag": int, "thickness": float, "rebar": bool (可选)},
            ...
        ],
        "material_properties": {
            matTag1: {"name": str, "color": str},
            ...
        }
    }
    """
    tag = section_data["tag"]
    layers = section_data["layers"]
    n_layers = len(layers)

    # 拼接OpenSees命令
    # 使用 ops.section() 而不是 eval(cmd) 更安全、更符合 openseespy 的用法
    args = [tag, n_layers]
    for layer in layers:
        args.extend([layer['matTag'], layer['thickness']])
    section('LayeredShell', *args)
    
    # 绘制截面 (调用简化后的函数)
    plt_layershell_section(section_data)