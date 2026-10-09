"""Public compatibility API for the UFP OpenSeesPy utilities."""

from .user_function import (
    # Tags and general analysis helpers
    GeneratePeaks,
    GetPeakStep,
    print_all_material_tags,
    print_all_section_tags,
    print_unit_table,
    search_tags,
    silent_analyze,

    # Section registries and section creation
    FRAME_INTEGRATION_REGISTRY,
    FRAME_SECTION_REGISTRY,
    RC_UNIT_WEIGHT_N_PER_MM3,
    SHELL_SECTION_REGISTRY,
    build_rect_retrofit_section,
    build_rect_section,
    create_layershell,
    define_frame_integration,
    make_rebar_row_points,
    make_rebar_side_points,
    rebar_area_from_dia,
    rebar_equivalent_layer_thickness,
    register_frame_integration,
    register_frame_section,
    register_frame_section_from_props,
    register_shell_section,
    wall_explicit_vertical_rebar_area,

    # Nodes, meshes, reinforcement, and model inspection
    add_vertical_rebars,
    add_vertical_rebars_by_x,
    beam_mesh,
    build_node_bucket_map,
    check_shared_nodes,
    check_wall_corner_pair,
    column_mesh,
    component_direction_summary,
    get_component_by_name,
    get_nodes_at_z,
    get_or_create_node,
    highlight_component_in_model,
    make_wall_zones,
    node_distance,
    plot_component_by_name,
    plot_model,
    get_shell_layer_strains,
    principal_strains_2d,
    plot_wall_smooth_cloud,
    print_model_summary,
    report_component_node_issues,
    report_near_miss_component_issues,
    reset_vertical_rebar_registry,
    retrofit_column_mesh,
    scan_model_near_miss_nodes,
    scan_model_node_connectivity,
    show_component_info,
    wall_grid_column_indices_by_x,
    wall_mesh,
    wall_mesh_by_base_nodes,
    wall_mesh_zoned,

    # Gravity, mass, diaphragms, and table-driven modeling
    apply_floor_gravity_and_mass,
    apply_floor_gravity_load,
    apply_model_table,
    apply_story_self_weight_gravity_and_mass,
    apply_story_self_weight_gravity_loads,
    assign_floor_mass_to_master,
    assign_floor_mass_to_nodes,
    assign_story_mass_to_diaphragm_masters,
    create_rigid_diaphragm_at_floor,
    create_story_rigid_diaphragms,
    summarize_story_self_weight,

    # Modal analysis, damping, and ground motions
    analyze_modal_participation,
    apply_rayleigh_by_modes,
    apply_uniform_damping,
    plot_bidirectional_ground_motion,
    plot_bidirectional_response_spectrum,
    print_modal_participation_summary,
    read_ground_motion,
    run_transient_analysis,
    save_current_eigen_data_for_opstool,
    select_uniform_damping_frequency_band,
    set_rayleigh_damping,
    setup_bidirectional_modal_rayleigh,

    # Native recorders and post-processing
    collect_native_component_metadata,
    extract_native_floor_acceleration_time_history,
    extract_native_floor_displacement_time_history,
    load_native_postprocess_results,
    plot_native_base_shear_roof_displacement,
    plot_native_base_shear_time_history,
    plot_native_floor_acceleration_time_history,
    plot_native_floor_displacement_time_history,
    plot_native_frame_story_fy,
    plot_native_roof_displacement_time_history,
    plot_native_shear_share,
    plot_native_story_displacement,
    plot_native_story_drift,
    plot_native_story_metric,
    plot_native_story_pfa,
    plot_native_wall_story_envelope,
    postprocess_frame_wall_shear_share,
    postprocess_native_dynamic_recorders,
    select_floor_corner_nodes,
    setup_full_native_dynamic_recorders,

    # Layered-shell visualization
    plt_layershell_section,
)

# Export only the compatibility names imported above, not the submodule object.
__all__ = tuple(
    name
    for name in globals()
    if not name.startswith("_") and name != "user_function"
)
