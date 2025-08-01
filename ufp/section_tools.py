import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from openseespy.opensees import section

def visualize_section_with_legend(section_data, mat_colors, mat_names, rebar_tags=(10, 11)):
    layers = section_data["layers"]
    wall_length = section_data["length"]

    fig, ax = plt.subplots(figsize=(10, 4))
    current_y = 0
    used_tags = set()
    offset_toggle = True

    for layer in layers:
        matTag = layer["matTag"]
        thickness = layer["thickness"]
        facecolor = mat_colors.get(matTag, 'gray')
        is_rebar = matTag in rebar_tags
        edgecolor = mat_colors.get(matTag, 'black') if is_rebar else 'black'

        rect = plt.Rectangle(
            (0, current_y), wall_length, thickness,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linestyle='--' if is_rebar else '-',
            linewidth=1.5 if is_rebar else 0.5
        )
        ax.add_patch(rect)

        y_center = current_y + thickness / 2
        if is_rebar:
            offset = 0.002
            y_text = y_center + offset if offset_toggle else y_center - offset
            offset_toggle = not offset_toggle
        else:
            y_text = y_center

        label = f'{mat_names.get(matTag, "Material")} ({thickness*1000:.2f} mm)'
        ax.text(
            wall_length / 2, y_text,
            label,
            ha='center', va='center',
            fontsize=8, color='black'
        )

        used_tags.add(matTag)
        current_y += thickness

    ax.set_xlim(0, wall_length)
    ax.set_ylim(0, current_y)
    ax.set_aspect(wall_length / current_y)
    ax.invert_yaxis()
    ax.set_xlabel("Wall Length (m)")
    ax.set_ylabel("Wall Thickness (m)")
    ax.set_title(f"LayeredShell Section tag={section_data['tag']}")
    ax.grid(True, linestyle='--', alpha=0.3)

    legend_handles = [
        Patch(facecolor=mat_colors[tag], edgecolor='black', label=mat_names[tag])
        for tag in used_tags if tag in mat_names
    ]
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    plt.show()


def create_layered_shell_section(section_data, mat_colors, mat_names):
    tag = section_data["tag"]
    layers = section_data["layers"]
    n_layers = len(layers)

    args = []
    for layer in layers:
        args.extend([layer["matTag"], layer["thickness"]])

    section('LayeredShell', tag, n_layers, *args)

    visualize_section_with_legend(section_data, mat_colors, mat_names)
