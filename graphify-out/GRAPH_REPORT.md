# Graph Report - visionview  (2026-11-24)

## Corpus Check
- 45 files · ~44,215 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: (none) 1, .build 1, .spec 1)

## Summary
- 601 nodes · 1192 edges · 33 communities (11 shown, 22 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 87 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `bc094d0c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_audit.py
- rfdetr_bbox.py
- WindowModule
- cv2
- NodeBlock
- VideoLabel
- NodeCanvas
- process_canva.py
- steps
- ParameterDialog
- ProcessWindow
- Basler
- VisionView
- DirectoryViewer
- template_matching.py
- build_target
- build.sh script
- CLAUDE.md
- .claude/CLAUDE.md
- NodeSocket
- StartupSplash
- CalibrationDialog
- ModuleList
- CameraInitDialog
- ToggleSwitch
- DraggableListWidget
- NodeConnection

## God Nodes (most connected - your core abstractions)
1. `WindowModule` - 52 edges
2. `VisionApp` - 50 edges
3. `VideoLabel` - 27 edges
4. `NodeCanvas` - 24 edges
5. `Basler` - 22 edges
6. `NodeBlock` - 19 edges
7. `ParameterDialog` - 19 edges
8. `ProcessWindow` - 15 edges
9. `check_deep_learning_support()` - 15 edges
10. `compose_split_view()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `test_cv_pipeline_toggle_switch()` --uses--> `ToggleSwitch`  [INFERRED]
  test_audit.py → utils/process_canva.py
- `NodeBlock` --uses--> `ParameterDialog`  [INFERRED]
  utils/process_canva.py → computer_vision/parameter_dialog.py
- `test_device_compatibility_and_ai_fallback()` --calls--> `ProcessWindow`  [EXTRACTED]
  test_audit.py → utils/process_canva.py
- `test_video_streaming_optimizations()` --calls--> `WindowModule`  [EXTRACTED]
  test_audit.py → utils/window_module.py
- `test_cv_pipeline_toggle_switch()` --calls--> `ImageWorker`  [EXTRACTED]
  test_audit.py → camera/image_worker.py

## Import Cycles
- None detected.

## Communities (33 total, 22 thin omitted)

### Community 0 - "test_audit.py"
Cohesion: 0.07
Nodes (20): get_cv_module(), VisionApp, test_ai_modules_integration(), test_area_square_polygon_tools(), test_capture_image_shortcuts(), test_crop_and_template_drag_dropdowns(), test_cv_pipeline_toggle_switch(), test_filters() (+12 more)

### Community 1 - "rfdetr_bbox.py"
Cohesion: 0.05
Nodes (23): load_metadata_config(), MODEL, render_detection_overlay(), resolve_default_weights(), resolve_model_class(), run_detection_inference(), load_metadata_config(), MODEL (+15 more)

### Community 3 - "cv2"
Cohesion: 0.05
Nodes (20): CameraWorker, get_cv_module(), ImageWorker, process(), process(), hex_to_bgr(), process(), process() (+12 more)

### Community 4 - "NodeBlock"
Cohesion: 0.07
Nodes (12): test_device_compatibility_and_ai_fallback(), test_startup_ai_environment_scan_and_install_flow(), check_deep_learning_support(), check_startup_ai_environment(), detect_gpu_hardware(), get_device_setting(), set_device_setting(), show_unsupported_feature_popup() (+4 more)

### Community 7 - "process_canva.py"
Cohesion: 0.11
Nodes (8): get_ai_scripts(), process(), resolve_model_files(), GuiSignal, compute_angle_deg(), fit_circle_3p(), run_1d_edge_caliper(), get_cached_cv_modules()

### Community 8 - "steps"
Cohesion: 0.06
Nodes (36): name, runs-on, steps, defaults, name, runs-on, steps, strategy (+28 more)

### Community 9 - "ParameterDialog"
Cohesion: 0.12
Nodes (19): ParameterDialog, pick_color(), reset_crop_btn(), reset_grab_btn(), reset_pick_btn(), sync_tolerances(), trigger_crop_drag(), on_cancelled() (+11 more)

### Community 11 - "Basler"
Cohesion: 0.17
Nodes (6): Basler, check_item(), build_subtree(), test_camera_parameters_tabs(), test_checkbox_blue_fill_style(), test_spinbox_and_input_styling()

### Community 12 - "VisionView"
Cohesion: 0.09
Nodes (21): 1. Prerequisites, 2. Installation, 3. Running in Development Mode, 4. Running the Test Suite, Adding Custom Filter Nodes & AI Models, Automated Self-Provisioning Compilation, Development Setup, Distributing to End Users (+13 more)

### Community 14 - "template_matching.py"
Cohesion: 0.32
Nodes (4): get_live_template(), get_template(), process(), set_live_template()

### Community 22 - "NodeSocket"
Cohesion: 0.16
Nodes (3): InputNode, NodeSocket, OutputNode

## Knowledge Gaps
- **22 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `WindowModule` connect `WindowModule` to `test_audit.py`, `VideoLabel`, `process_canva.py`, `ProcessWindow`, `.get_clean_title`, `.map_to_real`, `.contextMenuEvent`, `CalibrationDialog`?**
  _High betweenness centrality (0.164) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `WindowModule` (e.g. with `CalibrationDialog` and `ProcessWindow`) actually correct?**
  _`WindowModule` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_audit.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07365967365967366 - nodes in this community are weakly interconnected._
- **Why does `ParameterDialog` connect `ParameterDialog` to `test_audit.py`, `Basler`, `NodeBlock`, `process_canva.py`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Should `rfdetr_bbox.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0519774011299435 - nodes in this community are weakly interconnected._
- **Why does `VisionApp` connect `test_audit.py` to `process_canva.py`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Should `WindowModule` be split into smaller, more focused modules?**
  _Cohesion score 0.11692307692307692 - nodes in this community are weakly interconnected._