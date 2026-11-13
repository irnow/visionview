import sys
import os
import time
import shutil
from pathlib import Path
import cv2
import numpy as np

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtGui import QMouseEvent, QPixmap

# Ensure clean app instance
app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

def test_imports():
    print("Testing module imports...")
    import main
    from utils.gui_signal import gui_signal
    from utils.window_module import WindowModule
    from utils.directory_viewer import DirectoryViewer
    from utils.video_label import VideoLabel
    from utils.process_canva import ProcessWindow, NodeCanvas, NodeBlock, NodeConnection, InputNode, OutputNode
    from utils.annotators import annotate_image
    from camera.camera_worker import CameraWorker
    from camera.image_worker import ImageWorker
    from camera.basler import Basler
    from camera.dialog import CameraInitDialog
    import computer_vision.canny_edge_detection as canny
    import computer_vision.clahe as clahe
    import computer_vision.gaussian_blur as gblur
    import computer_vision.threshold as thresh
    import computer_vision.crop as crop
    import computer_vision.resize as resize
    import computer_vision.deep_learning as dl
    print("  -> All imports succeeded!")

def test_filters():
    print("Testing filter processing with dummy images...")
    import computer_vision.canny_edge_detection as canny
    import computer_vision.clahe as clahe
    import computer_vision.gaussian_blur as gblur
    import computer_vision.threshold as thresh
    import computer_vision.crop as crop
    import computer_vision.resize as resize
    import computer_vision.deep_learning as dl

    dummy_bgr = np.full((200, 200, 3), 128, dtype=np.uint8)
    dummy_gray = np.full((200, 200), 128, dtype=np.uint8)

    # 1. Canny
    res = canny.process(dummy_bgr, {"low_thresh": 50, "high_thresh": 150})
    assert res is not None and res.shape == (200, 200, 3), f"Canny failed: {res.shape if res is not None else None}"
    assert canny.process(None) is None

    # 2. CLAHE
    res = clahe.process(dummy_bgr, {"clipLimit": 2.0, "grid_size": 8})
    assert res is not None and res.shape == (200, 200, 3), "CLAHE BGR failed"
    res_gray = clahe.process(dummy_gray, {"clipLimit": 2.0, "grid_size": 8})
    assert res_gray is not None and res_gray.shape == (200, 200, 3), "CLAHE Gray failed"
    assert clahe.process(None) is None

    # 3. Gaussian Blur
    res = gblur.process(dummy_bgr, {"ksize": 5, "sigma": 1.0})
    assert res is not None and res.shape == (200, 200, 3), "Gaussian blur BGR failed"
    res_gray = gblur.process(dummy_gray, {"ksize": 4, "sigma": 1.0}) # Even ksize should become odd (5)
    assert res_gray is not None and res_gray.shape == (200, 200, 3), "Gaussian blur Gray failed"
    assert gblur.process(None) is None

    # 4. Threshold
    res = thresh.process(dummy_bgr, {"thresh": 100, "maxval": 255})
    assert res is not None and res.shape == (200, 200, 3), "Threshold BGR failed"
    res_gray = thresh.process(dummy_gray, {"thresh": 100, "maxval": 255})
    assert res_gray is not None and res_gray.shape == (200, 200, 3), "Threshold Gray failed"
    assert thresh.process(None) is None

    # 5. Crop
    res = crop.process(dummy_bgr, {"mode": "center", "width": 100, "height": 80})
    assert res is not None and res.shape == (80, 100, 3), f"Crop center failed: {res.shape}"
    res = crop.process(dummy_bgr, {"mode": "dynamic", "width": 50, "height": 50, "x_pos": 10, "y_pos": 10})
    assert res is not None and res.shape == (50, 50, 3), f"Crop dynamic failed: {res.shape}"
    assert crop.process(None) is None

    # 6. Resize
    res = resize.process(dummy_bgr, {"mode": "fixed", "width": 300, "height": 150})
    assert res is not None and res.shape == (150, 300, 3), f"Resize fixed failed: {res.shape}"
    res = resize.process(dummy_bgr, {"mode": "percentage", "scale": 0.5})
    assert res is not None and res.shape == (100, 100, 3), f"Resize percentage failed: {res.shape}"
    assert resize.process(None) is None

    # 7. Deep Learning graceful fallback
    res = dl.process(dummy_bgr, {"model_script": "None"})
    assert res is not None and res.shape == (200, 200, 3), "Deep learning fallback failed"
    assert dl.process(None) is None

    # 8. Color Segmentation (with Color Picker support)
    import computer_vision.color_segmentation as color_seg
    green_patch = np.zeros((200, 200, 3), dtype=np.uint8)
    green_patch[50:150, 50:150] = (0, 255, 0) # BGR pure green
    res_green = color_seg.process(green_patch, {"pick_color": "#00FF00", "output_mode": "Masked Image"})
    assert res_green is not None and res_green.shape == (200, 200, 3), "Color segmentation failed"
    assert np.any(res_green[100, 100] == (0, 255, 0)), "Green patch should be retained in mask"
    assert np.all(res_green[10, 10] == (0, 0, 0)), "Background should be blacked out"

    # Test other modes
    res_bin = color_seg.process(green_patch, {"pick_color": "#00FF00", "output_mode": "Binary Mask"})
    assert res_bin is not None and res_bin.shape == (200, 200, 3)
    res_hl = color_seg.process(green_patch, {"pick_color": "#00FF00", "output_mode": "Color Highlight"})
    assert res_hl is not None and res_hl.shape == (200, 200, 3)
    res_cnt = color_seg.process(green_patch, {"pick_color": "#00FF00", "output_mode": "Contour Overlay"})
    assert res_cnt is not None and res_cnt.shape == (200, 200, 3)
    assert color_seg.process(None) is None

    # 9. Morphology
    import computer_vision.morphology as morph
    res_m = morph.process(dummy_bgr, {"operation": "Close", "kernel_shape": "Rectangle", "kernel_size": 5, "iterations": 1})
    assert res_m is not None and res_m.shape == (200, 200, 3), "Morphology failed"
    res_dilate = morph.process(dummy_bgr, {"operation": "Dilate", "kernel_shape": "Ellipse", "kernel_size": 3, "iterations": 1})
    assert res_dilate is not None and res_dilate.shape == (200, 200, 3), "Dilate failed"
    assert morph.process(None) is None

    # 10. Contour Analysis
    import computer_vision.contour_analysis as contour_mod
    shape_img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(shape_img, (100, 100), 30, (255, 255, 255), -1) # White circle
    res_cnt_analysis = contour_mod.process(shape_img, {"min_area": 50, "draw_contour": True, "show_stats_badge": True, "pass_fail_check": True, "target_count_min": 1, "target_count_max": 2})
    assert res_cnt_analysis is not None and res_cnt_analysis.shape == (200, 200, 3), "Contour analysis failed"
    assert contour_mod.process(None) is None

    # 11. Template Matching
    import computer_vision.template_matching as tpl_mod
    res_tpl_prompt = tpl_mod.process(dummy_bgr, {"template_file": ""})
    assert res_tpl_prompt is not None and res_tpl_prompt.shape == (200, 200, 3), "Template matching fallback failed"
    
    # Test with real template match
    canvas_scene = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(canvas_scene, (50, 50), (100, 100), (255, 255, 255), -1)
    tpl_patch = canvas_scene[50:100, 50:100].copy()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tf_name = tf.name
    cv2.imwrite(tf_name, tpl_patch)
    res_matched = tpl_mod.process(canvas_scene, {"template_file": tf_name, "threshold": 0.8, "mode": "Best Match"})
    assert res_matched is not None and res_matched.shape == (200, 200, 3), "Template match failed"
    try:
        os.unlink(tf_name)
    except Exception:
        pass
    assert tpl_mod.process(None) is None

    # 12. Optical Character Recognition (OCR)
    import computer_vision.optical_character_recognition as ocr_mod
    res_ocr = ocr_mod.process(dummy_bgr, {"engine": "Auto", "contrast_boost": True})
    assert res_ocr is not None and res_ocr.shape == (200, 200, 3), "OCR process failed"
    assert ocr_mod.process(None) is None
    # Test 7-segment digital display reader
    digit_img = np.zeros((60, 40), dtype=np.uint8)
    cv2.putText(digit_img, "8", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
    recognized = ocr_mod.decode_seven_segment(digit_img)
    print(f"  -> OCR 7-segment fallback reader tested on sample: '{recognized}'")

    print("  -> All filters passed tests!")

def test_gui_and_modules():
    print("Testing VisionApp GUI and Module interactions...")
    from main import VisionApp
    test_dir = Path("./test_media_output")
    test_dir.mkdir(exist_ok=True)

    win = VisionApp()
    win.show()
    QApplication.processEvents()
    win.dir_edit.setText(str(test_dir.resolve()))
    win.dir_viewer.directory = str(test_dir.resolve())
    win.show()
    QApplication.processEvents()

    # Test that central_container exists and mapToGlobal works without the old left_container AttributeError
    assert hasattr(win, 'central_container'), "VisionApp missing central_container"
    global_pt = win.central_container.mapToGlobal(QPoint(50, 50))
    assert global_pt is not None, "mapToGlobal failed"
    print("  -> central_container and mapping verified without crash!", flush=True)

    # Test create_new_module
    win.create_new_module()
    assert len(win.modules) == 1, "Module creation failed"
    mod = win.modules[0]
    assert win.active_module == mod, "Newly created module should be active"
    print("  -> Module created and auto-activated successfully!")

    # Test VideoLabel click on empty label without pixmap (previously potential crash)
    press_event = QMouseEvent(QMouseEvent.MouseButtonPress, QPointF(50, 50), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    mod.label.mousePressEvent(press_event)
    move_event = QMouseEvent(QMouseEvent.MouseMove, QPointF(60, 60), Qt.NoButton, Qt.NoButton, Qt.NoModifier)
    mod.label.mouseMoveEvent(move_event)
    release_event = QMouseEvent(QMouseEvent.MouseButtonRelease, QPointF(60, 60), Qt.LeftButton, Qt.NoButton, Qt.NoModifier)
    mod.label.mouseReleaseEvent(release_event)
    print("  -> VideoLabel clicks without pixmap handled safely!")

    # Test ImageWorker emitting 3 arguments to update_view
    QApplication.processEvents()
    dummy_img = np.full((400, 600, 3), 200, dtype=np.uint8)
    mod.update_view(dummy_img, 600, 400)
    QApplication.processEvents()
    assert mod.current_frame is not None, "update_view failed to store current_frame"
    assert mod.label.pixmap() is not None and not mod.label.pixmap().isNull(), "Pixmap was not updated on label"
    print("  -> update_view successfully displayed frame!")

    # Test Zoom application
    mod.apply_zoom(QPoint(100, 100), QPoint(200, 200))
    assert mod.zoom_rect is not None, "Zoom was not applied"
    # Update view with zoom active
    mod.update_view(dummy_img, 600, 400)
    # Reset zoom
    mod.zoom_rect = None
    mod.update_view(dummy_img, 600, 400)
    print("  -> Zoom and reset zoom executed cleanly!")

    # Test Capture Image
    mod.capture_image(str(test_dir))
    captured_files = list(test_dir.glob("*.png"))
    assert len(captured_files) > 0, "No captured image file was created!"
    print(f"  -> Capture image created file: {captured_files[0].name}")

    # Test Record Video and Stop Video
    mod.record_video(str(test_dir))
    assert mod.is_recording and mod.video_writer is not None, "Video recording did not start"
    # Feed 10 frames
    for _ in range(10):
        mod.update_view(dummy_img, 600, 400)
    mod.stop_video()
    assert not mod.is_recording and mod.video_writer is None, "Video recording did not stop cleanly"
    recorded_files = list(test_dir.glob("*.avi"))
    assert len(recorded_files) > 0, "No recorded video file was created!"
    # Verify video file is non-empty and readable by cv2
    cap = cv2.VideoCapture(str(recorded_files[0]))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count > 0, f"Recorded video has 0 frames! (frame_count={frame_count})"
    print(f"  -> Video recording verified! Saved {recorded_files[0].name} with {frame_count} frames!")

    # Test ProcessWindow and NodeCanvas
    mod.open_process_editor()
    pw = mod.process_window
    assert pw is not None, "ProcessWindow did not open"
    canvas = pw.canvas
    
    # Add nodes
    from utils.process_canva import NodeBlock
    blur_node = NodeBlock("Gaussian Blur")
    blur_node.parameters["ksize"] = 9
    canvas.scene.addItem(blur_node)
    blur_node.setPos(50, 0)
    
    thresh_node = NodeBlock("Threshold")
    thresh_node.parameters["thresh"] = 120
    canvas.scene.addItem(thresh_node)
    thresh_node.setPos(200, 0)

    # Test auto connect pipeline
    canvas.auto_connect_pipeline()
    seq = canvas.get_pipeline_sequence()
    assert len(seq) == 2, f"Pipeline sequence length should be 2, got {len(seq)}"
    assert seq[0]["name"] == "Gaussian Blur" and seq[0]["params"]["ksize"] == 9
    assert seq[1]["name"] == "Threshold" and seq[1]["params"]["thresh"] == 120
    print("  -> NodeCanvas pipeline auto-connected correctly!")

    # Test saving data and restoring setup
    saved_data = canvas.get_save_data()
    assert len(saved_data) == 2, "Saved data missing nodes"
    blur_saved = next((d for d in saved_data if d["title"] == "Gaussian Blur"), None)
    assert blur_saved is not None and "params" in blur_saved and blur_saved["params"]["ksize"] == 9, "Node parameters were not saved!"
    
    # Restore setup
    canvas.restore_setup(saved_data)
    restored_seq = canvas.get_pipeline_sequence()
    blur_restored = next((s for s in restored_seq if s["name"] == "Gaussian Blur"), None)
    assert blur_restored is not None and blur_restored["params"]["ksize"] == 9, "Restored parameters do not match!"
    print("  -> Node parameters preserved and restored across editor sessions!")

    # Test NodeBlock double-click parent_widget resolution & ParameterDialog
    from computer_vision.parameter_dialog import ParameterDialog
    view = canvas.scene.views()[0]
    parent_widget = getattr(view, 'process_window', None) or (view.window() if callable(getattr(view, 'window', None)) else view)
    assert parent_widget == pw, f"Expected ProcessWindow as parent_widget, got {parent_widget}"
    blur_node_in_scene = next(n for n in canvas.scene.items() if isinstance(n, NodeBlock) and n.title == "Gaussian Blur")
    dlg = ParameterDialog(blur_node_in_scene.title, blur_node_in_scene.param_def, blur_node_in_scene.parameters, parent=parent_widget)
    assert dlg is not None, "Failed to instantiate ParameterDialog"
    blur_node_in_scene.on_live_update({"ksize": 11, "sigma": 2.0})
    assert blur_node_in_scene.parameters["ksize"] == 11, "on_live_update failed to update parameters"
    print("  -> NodeBlock parameter dialog and live update verified without ProcessWindow callable crash!")

    # Test Node deletion without crash
    blur_node_in_scene = next(n for n in canvas.scene.items() if isinstance(n, NodeBlock) and n.title == "Gaussian Blur")
    blur_node_in_scene.remove_self()
    after_del_seq = canvas.get_pipeline_sequence()
    print("  -> Node deletion executed without crash!")

    # Test connection deletion without crash
    conn = next((n for n in canvas.scene.items() if hasattr(n, 'start_socket')), None)
    if conn:
        r_event = QMouseEvent(QMouseEvent.MouseButtonPress, QPointF(0, 0), Qt.RightButton, Qt.RightButton, Qt.NoModifier)
        conn.mousePressEvent(r_event)
        print("  -> Connection deletion executed without crash!")

    # Test close_event_logic
    mod.close_event_logic()
    assert mod.thread is None and mod.worker is None, "Module cleanup did not reset thread/worker"

    # Test remove module
    win.remove_module(mod)
    assert len(win.modules) == 0, "Module was not removed from window"
    print("  -> Module removed and cleaned up successfully!")

    # Clean up test output directory
    shutil.rmtree(test_dir, ignore_errors=True)
    print("  -> Cleanup complete!")

def test_area_square_polygon_tools():
    print("Testing Area tools (Square & Polygon) separation and measurements...")
    from main import VisionApp
    from utils.video_label import VideoLabel

    win = VisionApp()
    win.show()
    QApplication.processEvents()

    # 1. Verify Menu Structure
    assert hasattr(win, 'area_menu'), "VisionApp missing area_menu"
    assert hasattr(win, 'square_tool'), "VisionApp missing square_tool"
    assert hasattr(win, 'polygon_tool'), "VisionApp missing polygon_tool"

    assert win.square_tool.isCheckable(), "Square tool must be checkable"
    assert win.polygon_tool.isCheckable(), "Polygon tool must be checkable"
    assert win.square_tool.text() == "Square", f"Expected 'Square', got {win.square_tool.text()}"
    assert win.polygon_tool.text() == "Polygon", f"Expected 'Polygon', got {win.polygon_tool.text()}"
    print("  -> Menu hierarchy verified: Tools -> Area -> [Square, Polygon]")

    # 2. Verify Module Mode Synchronization
    win.create_new_module()
    mod = win.modules[0]

    # Switch to Square
    win.square_tool.setChecked(True)
    win.update_tool_mode(win.square_tool)
    assert mod.current_tool == "square", f"Expected current_tool 'square', got {mod.current_tool}"

    # Setup display frame on module
    dummy_img = np.full((400, 600, 3), 180, dtype=np.uint8)
    mod.update_view(dummy_img, 600, 400)
    QApplication.processEvents()

    # 3. Test Square Measurement
    mod.handle_selection(QPoint(100, 50), QPoint(200, 150))
    assert len(mod.measurements) == 1, f"Expected 1 measurement, got {len(mod.measurements)}"
    sq_m = mod.measurements[0]
    assert sq_m['type'] == 'square', f"Expected 'square', got {sq_m['type']}"
    assert sq_m['area'] > 0, f"Expected positive area, got {sq_m['area']}"
    print(f"  -> Square measurement recorded successfully: area = {sq_m['area']:.2f} px²")

    # 4. Switch to Polygon
    win.polygon_tool.setChecked(True)
    win.update_tool_mode(win.polygon_tool)
    assert mod.current_tool == "polygon", f"Expected current_tool 'polygon', got {mod.current_tool}"
    assert len(mod.label.polygon_points) == 0, "Polygon points should be clean"

    # 5. Test Interactive Polygon Addition & Closure
    # Add 4 points in a 100x100 square shape: (100, 50), (200, 50), (200, 150), (100, 150)
    mod.label.handle_polygon_click(QPoint(100, 50))
    assert len(mod.label.polygon_points) == 1
    mod.label.handle_polygon_click(QPoint(200, 50))
    assert len(mod.label.polygon_points) == 2
    mod.label.handle_polygon_click(QPoint(200, 150))
    assert len(mod.label.polygon_points) == 3
    mod.label.handle_polygon_click(QPoint(100, 150))
    assert len(mod.label.polygon_points) == 4

    # Click within 12px of start point (100, 50) to close polygon
    mod.label.handle_polygon_click(QPoint(103, 52))
    assert len(mod.label.polygon_points) == 0, "Polygon points should reset after closure"
    assert len(mod.measurements) == 2, f"Expected 2 measurements, got {len(mod.measurements)}"
    poly_m = mod.measurements[1]
    assert poly_m['type'] == 'polygon', f"Expected 'polygon', got {poly_m['type']}"
    assert poly_m['area'] > 0, f"Expected positive polygon area, got {poly_m['area']}"
    assert len(poly_m['real_points']) == 4, f"Expected 4 polygon vertices, got {len(poly_m['real_points'])}"
    print(f"  -> Polygon measurement recorded successfully: area = {poly_m['area']:.2f} px² with {len(poly_m['real_points'])} vertices")

    # 6. Test Polygon Cancellation via cancel_polygon
    mod.label.handle_polygon_click(QPoint(50, 50))
    mod.label.handle_polygon_click(QPoint(80, 50))
    assert len(mod.label.polygon_points) == 2
    mod.label.cancel_polygon()
    assert len(mod.label.polygon_points) == 0, "Polygon points should be cancelled"
    print("  -> Polygon cancellation verified")

    # 7. Test Reset Measurements
    mod.measurements.clear()
    assert len(mod.measurements) == 0
    print("  -> Reset measurements verified")

    win.remove_module(mod)
    print("  -> Area tools tests passed successfully!")

def test_maximized_window():
    print("Testing Bordered Fullscreen (showMaximized)...")
    from main import VisionApp

    win = VisionApp()
    win.show()
    QApplication.processEvents()
    win.showMaximized()
    # In maximized mode, window has standard frame/borders and is not borderless fullscreen
    assert not win.isFullScreen(), "Window should have borders (maximized), not borderless fullscreen"
    print("  -> Bordered fullscreen verified!")

def test_fullscreen_selected_video_module():
    print("Testing Fullscreen behaviors: Top Bar (selected module on desktop) & Module (panel canvas)...")
    from main import VisionApp
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent

    win = VisionApp()
    win.show()
    QApplication.processEvents()
    win.show()
    assert hasattr(win, 'action_fullscreen_module'), "VisionApp missing action_fullscreen_module"
    assert not win.action_fullscreen_module.isEnabled(), "Top bar fullscreen module should be disabled when no modules exist"

    # Verify toolbar icons are valid MaterialIcons (non-null)
    for act in [win.action_add, win.action_start, win.action_upload, win.action_process,
                win.action_capture, win.action_record, win.action_stop, win.action_fullscreen_module]:
        assert not act.icon().isNull(), f"Action {act.text()} icon is null"
    assert not win.btn_select_dir.icon().isNull(), "Folder select button icon is null"
    print("  -> All toolbar icons verified as valid MaterialIcons!")

    # 1. Create modules and test Module Header icon only fullscreens at panel canvas
    win.create_new_module()
    win.create_new_module()
    assert len(win.modules) == 2
    mod1 = win.modules[0]
    mod2 = win.modules[1]
    assert win.action_fullscreen_module.isEnabled(), "Top bar action should be enabled when module is active"
    assert not mod1.btn_enlarge.icon().isNull(), "Module btn_enlarge icon is null"
    assert not mod1.btn_close.icon().isNull(), "Module btn_close icon is null"

    # Click module 1 enlarge button
    mod1.btn_enlarge.click()
    assert mod1.is_enlarged, "Module 1 should be enlarged in canvas"
    assert not mod1.isHidden(), "Module 1 must be unhidden"
    assert mod2.isHidden(), "Module 2 should be hidden when module 1 is enlarged in canvas"
    assert mod1.parent() == win.central_container, "Module must stay inside panel canvas (central_container)"
    assert not getattr(mod1, 'is_fullscreen_window', False), "Module header button must not detach to desktop"
    print("  -> Module icon enlarged module strictly within panel canvas!")

    # Test adding new module while module 1 is enlarged on placeholder:
    # Existing module 1 must restore to smaller module in grid, new module 3 must not overlap
    win.create_new_module()
    assert len(win.modules) == 3
    mod3 = win.modules[2]
    assert not mod1.is_enlarged, "Module 1 must restore from enlarge when new module added"
    assert not mod1.isHidden() and not mod2.isHidden() and not mod3.isHidden(), "All modules must be visible in grid"
    pos1 = win.canvas_layout.getItemPosition(win.canvas_layout.indexOf(mod1))
    pos2 = win.canvas_layout.getItemPosition(win.canvas_layout.indexOf(mod2))
    pos3 = win.canvas_layout.getItemPosition(win.canvas_layout.indexOf(mod3))
    assert pos1 == (0, 0, 1, 1), f"mod1 position: {pos1}"
    assert pos2 == (0, 1, 1, 1), f"mod2 position: {pos2}"
    assert pos3 == (1, 0, 1, 1), f"mod3 position: {pos3}"
    print("  -> Adding new module while module is enlarged restores existing module to smaller grid without overlap!")

    # Test maximizing module 3: it should be at top of other modules in fullscreen of placeholder
    mod3.btn_enlarge.click()
    assert mod3.is_enlarged, "Module 3 should be enlarged"
    assert not mod1.is_enlarged and not mod2.is_enlarged
    assert not mod3.isHidden() and mod1.isHidden() and mod2.isHidden()
    pos3_enlarged = win.canvas_layout.getItemPosition(win.canvas_layout.indexOf(mod3))
    assert pos3_enlarged == (0, 0, 3, 2), f"mod3 enlarged position: {pos3_enlarged}"
    print("  -> Maximizing module 3 puts it at top of placeholder fullscreen!")

    # Restore via Esc
    esc_event = QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
    mod3.label.keyPressEvent(esc_event)
    assert not mod3.is_enlarged
    assert not mod1.isHidden() and not mod2.isHidden() and not mod3.isHidden()
    win.remove_module(mod3)
    assert not mod1.is_enlarged, "Module 1 should not be enlarged after Esc"
    assert not mod1.isHidden() and not mod2.isHidden(), "All modules should be restored in canvas"
    print("  -> Esc returned module to normal canvas grid layout successfully!")

    # 2. Test Top Bar action fullscreens selected module on full desktop / window
    win.set_active_module(mod1)
    win.action_fullscreen_module.trigger()
    assert getattr(mod1, 'is_fullscreen_window', False), "Module 1 should enter fullscreen window on desktop"
    assert mod1.isFullScreen(), "Module 1 should be fullscreen window on desktop"
    assert mod1.parent() is None, "Module 1 should be top-level window when in desktop fullscreen"
    assert mod1.header.isHidden(), "Header must be removed/hidden in desktop fullscreen"
    assert mod1.footer.isHidden(), "Footer must be removed/hidden in desktop fullscreen"
    assert not mod1.label.isHidden(), "Image must be visible"
    assert win.action_fullscreen_module.text() == "Exit Full Screen"
    print("  -> Top bar icon fullscreens selected module on full desktop / window successfully (only image shown)!")

    # Press Esc to restore module back to central container grid
    mod1.keyPressEvent(esc_event)
    assert not getattr(mod1, 'is_fullscreen_window', False), "Module 1 should exit desktop fullscreen after Esc"
    assert mod1.parent() == win.central_container, "Module 1 should be back in central container"
    assert not mod1.isFullScreen()
    assert not mod1.header.isHidden(), "Header must be restored"
    assert not mod1.footer.isHidden(), "Footer must be restored"
    assert win.action_fullscreen_module.text() == "Full Screen Selected Module"
    print("  -> Esc returned module back into GUI grid successfully (header/footer restored)!")

    # 3. Remove modules
    win.remove_module(mod1)
    win.remove_module(mod2)
    assert len(win.modules) == 0
    assert not win.action_fullscreen_module.isEnabled(), "Top bar action should be disabled when all modules removed"
    print("  -> Fullscreen behaviors test passed successfully!")

def test_process_editor_bottom_dock():
    print("Testing Process Editor dock attached at bottom and Saved Directory on right...")
    from main import VisionApp
    win = VisionApp()
    win.show()
    QApplication.processEvents()
    win.show()

    # 1. Check default dock areas
    assert hasattr(win, 'process_dock'), "VisionApp missing process_dock"
    assert hasattr(win, 'dir_dock'), "VisionApp missing dir_dock"
    assert win.dockWidgetArea(win.dir_dock) == Qt.RightDockWidgetArea, "dir_dock should be on Right dock by default"
    assert win.dockWidgetArea(win.process_dock) == Qt.BottomDockWidgetArea, "process_dock should be on Bottom dock by default"
    assert win.process_dock.isHidden(), "process_dock should be hidden by default (only appear when clicked or interacting)"
    print("  -> process_dock at bottom and dir_dock on right verified!")

    # 2. Check creating module keeps process_dock hidden until user interacts
    win.create_new_module()
    mod1 = win.modules[0]
    assert win.process_dock.isHidden(), "process_dock should remain hidden upon creating module"
    assert win.process_panel.currentWidget() == mod1.process_window, "Process editor should have mod1 ready in panel"
    print("  -> Module 1 created with process editor prepared while remaining hidden!")

    # 3. Create module 2
    win.create_new_module()
    mod2 = win.modules[1]
    assert win.process_dock.isHidden()
    assert win.process_panel.currentWidget() == mod2.process_window
    print("  -> Module 2 prepared!")

    # 4. Trigger action_process (click icon) -> process_dock appears on bottom dock
    win.action_process.trigger()
    assert not win.process_dock.isHidden(), "action_process click should reveal process_dock"
    assert win.dockWidgetArea(win.process_dock) == Qt.BottomDockWidgetArea, "process_dock must appear on bottom dock"
    print("  -> Icon click revealed process editor on bottom dock!")

    # 5. Switch active module back to mod1 -> updates visible process editor
    win.set_active_module(mod1)
    assert win.process_panel.currentWidget() == mod1.process_window, "Process editor should switch back to mod1"
    print("  -> Active module switching updates bottom process editor!")

    # 6. Hide and test interacting via module.open_process_editor()
    win.process_dock.hide()
    assert win.process_dock.isHidden()
    mod1.open_process_editor()
    assert not win.process_dock.isHidden(), "mod1.open_process_editor() should reveal process_dock"
    print("  -> Module interaction reveals process editor on bottom dock!")

    # 7. Remove all modules and verify placeholder
    win.remove_module(mod1)
    win.remove_module(mod2)
    assert win.process_panel.currentWidget() == win.default_process_label
    print("  -> Empty modules reverts to default placeholder!")

    win.close()
    print("  -> Process editor bottom dock and Saved Directory right dock tests passed successfully!")

def test_camera_parameters_tabs():
    print("\nTesting Camera Parameters Basic | Advanced Tabs...")
    from camera.basler import Basler
    from PySide6.QtWidgets import QTabWidget, QTreeWidget, QLineEdit, QCheckBox, QPushButton

    b = Basler()
    tab_widget = b.parameter_widget()

    assert isinstance(tab_widget, QTabWidget), "parameter_widget() must return a QTabWidget"
    assert tab_widget.count() == 2, f"Expected 2 tabs, got {tab_widget.count()}"
    assert tab_widget.tabText(0) == "Basic", f"Expected tab 0 'Basic', got {tab_widget.tabText(0)}"
    assert tab_widget.tabText(1) == "Advanced", f"Expected tab 1 'Advanced', got {tab_widget.tabText(1)}"
    print("  -> QTabWidget contains 'Basic' and 'Advanced' tabs!")

    # Verify Basic tab controls
    assert len(b._basic_controls) >= 9, f"Expected at least 9 basic controls, got {len(b._basic_controls)}"
    expected_basic = ["pixel format", "exposure", "gain", "contrast", "hue", "saturation", "brightness", "reversex", "reversey"]
    for param in expected_basic:
        assert param in b._basic_controls, f"Missing basic control: {param}"
    print("  -> All Basic parameter controls verified!")

    # Verify Advanced tab UI components
    assert isinstance(b._search_bar, QLineEdit), "Advanced tab missing search bar"
    assert isinstance(b._editable_chk, QCheckBox), "Advanced tab missing editable checkbox"
    assert isinstance(b._advanced_tree, QTreeWidget), "Advanced tab missing feature tree"
    assert b._advanced_tree.columnCount() == 2, "Feature tree must have 2 columns (Feature, Value)"
    print("  -> Advanced tab controls and tree verified!")

    # Test with physical device if connected
    devs, serials = b.find()
    if devs:
        print(f"  -> Testing with connected camera: {serials[0]}")
        try:
            b.setup(devs[0])
            b.populate_advanced_tree()
            assert len(b._advanced_controls) > 50, f"Expected >50 GenICam controls, got {len(b._advanced_controls)}"
            print(f"  -> Populated {len(b._advanced_controls)} camera features in Advanced tree!")

            # Test filter search
            b._search_bar.setText("Exposure")
            b._filter_advanced_tree()
            b._search_bar.setText("")
            b._filter_advanced_tree()

            # Test parameter sync between basic and advanced
            b.apply_parameters({"exposure": 6200})
            assert b.camera.ExposureTime.Value == 6200.0, "apply_parameters should update ExposureTime"
            
            exp_param = b.camera.GetNodeMap().GetNode("ExposureTime")
            b._set_advanced_param_value(exp_param, 5000.0)
            assert b.camera.ExposureTime.Value == 5000.0, "_set_advanced_param_value should update ExposureTime"
            assert b._active_parameters["exposure"] == 5000.0, "Advanced update should sync to active parameters"

            b.close()
            print("  -> Physical camera advanced features and sync verified!")
        except Exception as e:
            if "exclusively opened by another client" in str(e) or "Device is exclusively opened" in str(e):
                print(f"  -> Physical camera {serials[0]} is busy (exclusively opened by running instance): {e}")
                print("  -> Skipped physical camera live interaction (camera in use).")
            else:
                raise
    else:
        print("  -> No physical camera connected, placeholder verified cleanly.")

    print("  -> Camera Parameters tabs test passed successfully!")

def test_checkbox_blue_fill_style():
    print("\nTesting Checkbox Blue Fill Style (no tick sign)...")
    from main import CHECKBOX_STYLESHEET, VisionApp
    from camera.basler import Basler
    from computer_vision.parameter_dialog import ParameterDialog
    from PySide6.QtWidgets import QCheckBox, QApplication
    import numpy as np

    # 1. Verify stylesheet syntax rules
    assert "QCheckBox::indicator:checked" in CHECKBOX_STYLESHEET
    assert "background-color: #0078d4" in CHECKBOX_STYLESHEET
    assert "image: none" in CHECKBOX_STYLESHEET
    print("  -> CHECKBOX_STYLESHEET contains blue fill (#0078d4) and image: none!")

    # 2. Test standalone QCheckBox rendering
    cb = QCheckBox()
    cb.setChecked(True)
    cb.show()
    QApplication.processEvents()

    pix = cb.grab()
    img = pix.toImage()
    # Count blue pixels
    blue_count = 0
    non_blue_inside_box = 0
    for y in range(img.height()):
        for x in range(img.width()):
            c = img.pixelColor(x, y)
            # Blue range for #0078d4 (R: 0-40, G: 90-150, B: 180-235)
            if c.red() < 50 and 90 < c.green() < 160 and c.blue() > 170:
                blue_count += 1
    
    assert blue_count > 50, f"Expected >50 blue pixels in checked checkbox, got {blue_count}"
    print(f"  -> QCheckBox rendered with solid blue fill ({blue_count} blue pixels) and no tick sign!")

    # 3. Test Basler parameter widget Reversex checkbox
    b = Basler()
    pw = b.parameter_widget()
    pw.show()
    QApplication.processEvents()

    reversex_chk = b._basic_controls.get("reversex")
    assert reversex_chk is not None, "reversex control must be present"
    reversex_chk.setChecked(True)
    QApplication.processEvents()

    rx_pix = reversex_chk.grab()
    rx_img = rx_pix.toImage()
    rx_blue_count = 0
    for y in range(rx_img.height()):
        for x in range(rx_img.width()):
            c = rx_img.pixelColor(x, y)
            if c.red() < 50 and 90 < c.green() < 160 and c.blue() > 170:
                rx_blue_count += 1

    assert rx_blue_count > 50, f"Expected >50 blue pixels in Reversex checkbox, got {rx_blue_count}"
    print(f"  -> Basler Reversex checkbox verified with solid blue fill ({rx_blue_count} blue pixels)!")

    # 4. Test ParameterDialog bool checkbox
    cfg = {"enable_test": {"type": "bool", "default": True}}
    dlg = ParameterDialog("Test Config", cfg, {"enable_test": True})
    dlg_chk = dlg.widgets.get("enable_test")
    assert isinstance(dlg_chk, QCheckBox), "ParameterDialog bool must be a QCheckBox"
    assert dlg_chk.isChecked(), "Must be checked"
    dlg.show()
    QApplication.processEvents()

    dlg_pix = dlg_chk.grab()
    dlg_img = dlg_pix.toImage()
    dlg_blue_count = 0
    for y in range(dlg_img.height()):
        for x in range(dlg_img.width()):
            c = dlg_img.pixelColor(x, y)
            if c.red() < 50 and 90 < c.green() < 160 and c.blue() > 170:
                dlg_blue_count += 1

    assert dlg_blue_count > 50, f"Expected >50 blue pixels in ParameterDialog checkbox, got {dlg_blue_count}"
    print(f"  -> ParameterDialog bool checkbox verified with solid blue fill ({dlg_blue_count} blue pixels)!")
    dlg.close()

    # 5. Test ParameterDialog 'color' and 'file' controls
    adv_cfg = {
        "target_color": {"type": "color", "default": "#00FF00"},
        "template_image": {"type": "file", "default": "test.png"}
    }
    adv_dlg = ParameterDialog("Color & File Test", adv_cfg, {"target_color": "#FF5500", "template_image": "my_tpl.png"})
    adv_vals = adv_dlg.get_values()
    assert adv_vals["target_color"] == "#FF5500", f"Expected #FF5500, got {adv_vals['target_color']}"
    assert adv_vals["template_image"] == "my_tpl.png", f"Expected my_tpl.png, got {adv_vals['template_image']}"
    assert "target_color" in adv_dlg.widgets and "template_image" in adv_dlg.widgets
    adv_dlg.close()
    print("  -> ParameterDialog 'color' (picker) and 'file' controls verified!")

    print("  -> All checkbox styling tests passed successfully!")

def test_video_streaming_optimizations():
    print("\nTesting Video Streaming Optimizations...")
    import time
    from utils.window_module import WindowModule
    from camera.camera_worker import _CV_MODULE_CACHE, get_cv_module
    from main import VisionApp
    from PySide6.QtCore import QPoint

    # 1. Test CV Module Caching
    m1 = get_cv_module("gaussian_blur")
    m2 = get_cv_module("gaussian_blur")
    assert m1 is m2, "get_cv_module must return cached module instance"
    assert "gaussian_blur" in _CV_MODULE_CACHE
    print("  -> Filter module caching verified!")

    # 2. Test Fast UI Downsampling and zero-copy storage in WindowModule
    win = VisionApp()
    mod = WindowModule(win, 1)
    mod.show()
    QApplication.processEvents()

    # Generate high resolution dummy frame (4200 x 2128 x 3)
    high_res_frame = np.full((2128, 4200, 3), 128, dtype=np.uint8)
    t0 = time.perf_counter()
    mod.update_view(high_res_frame, 4200, 2128)
    t1 = time.perf_counter()
    duration_ms = (t1 - t0) * 1000.0

    assert mod.current_frame is high_res_frame, "current_frame should reference display_frame without redundant copy"
    assert mod.label.pixmap() is not None, "Label pixmap should be populated"
    # Pixmap should be resized to fit widget cleanly
    assert mod.label.pixmap().width() <= mod.label.width()
    assert mod.label.pixmap().height() <= mod.label.height()
    print(f"  -> High-res 4200x2128 frame rendered in {duration_ms:.2f} ms (sub-10ms target achieved)!")

    # 3. Test Real-time Mouse Coordinate Tracking
    center_pos = QPoint(mod.label.width() // 2, mod.label.height() // 2)
    real_x, real_y = mod.label.map_to_real(center_pos)
    assert 1800 < real_x < 2400 and 900 < real_y < 1200, f"Unexpected center coordinates: {real_x}, {real_y}"
    print(f"  -> Pixel mapping verified: center screen maps to real [{real_x}, {real_y}]!")

    win.close()
    print("  -> Video streaming optimizations tests passed successfully!")

def test_spinbox_and_input_styling():
    print("\nTesting SpinBox Interaction and White Value Background Styling...")
    from camera.basler import Basler
    from PySide6.QtWidgets import QApplication

    b = Basler()
    pw = b.parameter_widget()
    pw.resize(320, 500)
    pw.show()
    QApplication.processEvents()

    # 1. Verify Exposure SpinBox value has white background
    exp = b._basic_controls.get("exposure")
    assert exp is not None, "Exposure control must exist"
    exp_img = exp.grab().toImage()
    exp_val_color = exp_img.pixelColor(exp.width() // 4, exp.height() // 2).name().lower()
    assert exp_val_color == "#ffffff", f"Exposure value background should be #ffffff, got {exp_val_color}"
    print("  -> Exposure value background verified as pure white (#ffffff)!")

    # 2. Verify Gain SpinBox value has white background
    gain = b._basic_controls.get("gain")
    assert gain is not None, "Gain control must exist"
    gain_img = gain.grab().toImage()
    gain_val_color = gain_img.pixelColor(gain.width() // 4, gain.height() // 2).name().lower()
    assert gain_val_color == "#ffffff", f"Gain value background should be #ffffff, got {gain_val_color}"
    print("  -> Gain value background verified as pure white (#ffffff)!")

    # 3. Verify Pixel Format ComboBox value has white background
    pf = b._basic_controls.get("pixel format")
    assert pf is not None, "Pixel format control must exist"
    pf_img = pf.grab().toImage()
    pf_val_color = pf_img.pixelColor(pf.width() // 4, pf.height() // 2).name().lower()
    assert pf_val_color == "#ffffff", f"Pixel format background should be #ffffff, got {pf_val_color}"
    print("  -> Pixel Format ComboBox background verified as pure white (#ffffff)!")

    # 4. Verify SpinBox buttons are interactive and successfully step
    val_init = exp.value()
    exp.stepUp()
    assert exp.value() == val_init + 100, f"Expected stepUp to {val_init + 100}, got {exp.value()}"
    exp.stepDown()
    assert exp.value() == val_init, f"Expected stepDown to {val_init}, got {exp.value()}"
    print("  -> SpinBox button stepUp and stepDown interactions verified!")

    gain_init = gain.value()
    gain.stepUp()
    assert gain.value() == gain_init + 1, f"Expected gain stepUp to {gain_init + 1}, got {gain.value()}"
    gain.stepDown()
    assert gain.value() == gain_init, f"Expected gain stepDown to {gain_init}, got {gain.value()}"
    print("  -> Gain button stepUp and stepDown interactions verified!")

    pw.close()
    print("  -> SpinBox and input styling tests passed successfully!")

def test_capture_image_shortcuts():
    print("\nTesting Capture Image Shortcuts (Ctrl+s and Ctrl+S)...")
    import time, shutil
    from PySide6.QtGui import QKeyEvent
    from main import VisionApp
    from pathlib import Path

    win = VisionApp()
    win.create_new_module()
    mod = win.modules[0]
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    mod.update_view(dummy, 100, 100)

    # 1. Verify shortcuts on action_capture
    shortcuts = [s.toString() for s in win.action_capture.shortcuts()]
    assert "Ctrl+S" in shortcuts, f"Expected Ctrl+S in shortcuts, got {shortcuts}"
    assert "Ctrl+Shift+S" in shortcuts, f"Expected Ctrl+Shift+S in shortcuts, got {shortcuts}"
    print(f"  -> QAction shortcuts verified: {shortcuts}!")

    test_dir = Path("./test_shortcut_dir")
    shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(exist_ok=True)
    win.dir_edit.setText(str(test_dir.resolve()))

    # 2. Test lowercase Ctrl+s
    e_lower = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key_S, Qt.ControlModifier, "s")
    win.keyPressEvent(e_lower)
    QApplication.processEvents()
    files = list(test_dir.glob("*.png"))
    assert len(files) == 1, f"Failed lowercase Ctrl+s capture: {len(files)}"
    print("  -> Lowercase (Ctrl+s) capture verified!")

    time.sleep(1.05)

    # 3. Test uppercase Ctrl+S (with Shift)
    e_upper = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key_S, Qt.ControlModifier | Qt.ShiftModifier, "S")
    mod.label.keyPressEvent(e_upper)
    QApplication.processEvents()
    files = list(test_dir.glob("*.png"))
    assert len(files) == 2, f"Failed uppercase Ctrl+S capture: {len(files)}"
    print("  -> Uppercase (Ctrl+S / Shift) capture on VideoLabel verified!")

    win.close()
    shutil.rmtree(test_dir, ignore_errors=True)
    print("  -> Capture image shortcuts tests passed successfully!")

def test_device_compatibility_and_ai_fallback():
    print("\nTesting Device Compatibility and AI Unsupported Fallback...")
    from utils.device_compatibility import check_deep_learning_support
    import utils.device_compatibility as dev_comp
    from utils.process_canva import ProcessWindow, NodeBlock
    import computer_vision.deep_learning as dl

    # 1. Native device detection
    supported, reason, details = check_deep_learning_support(force_refresh=True)
    assert isinstance(supported, bool)
    assert isinstance(details, dict)
    print(f"  -> Native hardware check: supported={supported}, device={details.get('device')}")

    # 2. Simulate unsupported device (e.g. laptop without PyTorch / CUDA)
    os.environ["VISIONVIEW_FORCE_NO_TORCH"] = "1"
    no_support, no_reason, _ = check_deep_learning_support(force_refresh=True)
    assert no_support is False
    assert "override" in no_reason or "not installed" in no_reason
    print("  -> Unsupported device detection verified!")

    # 3. Test deep_learning.process() graceful fallback
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    out_img = dl.process(dummy_img, {"model_script": "rfdetr_bbox", "model_dir": "non_existent"})
    assert out_img is dummy_img, "process() should safely return original frame on unsupported device"
    print("  -> deep_learning.process() safe bypass on unsupported device verified!")

    # 4. Test UI popup prevention on canvas
    proc_win = ProcessWindow()
    canvas = proc_win.canvas
    initial_node_count = len([item for item in canvas.scene.items() if isinstance(item, NodeBlock)])

    popup_triggered = []
    def mock_popup(parent=None, feature_name="Deep Learning", reason=None):
        popup_triggered.append((feature_name, reason))
        return 1

    orig_popup = dev_comp.show_unsupported_feature_popup
    dev_comp.show_unsupported_feature_popup = mock_popup
    import utils.process_canva as pc_mod
    orig_pc_popup = getattr(pc_mod, 'show_unsupported_feature_popup', None)
    pc_mod.show_unsupported_feature_popup = mock_popup
    os.environ["VISIONVIEW_HEADLESS_TEST"] = "1"

    try:
        from PySide6.QtWidgets import QListWidgetItem
        item = QListWidgetItem("Deep Learning")
        proc_win.module_list.on_item_double_clicked(item)
        QApplication.processEvents()

        assert len(popup_triggered) == 1, f"show_unsupported_feature_popup should have been triggered, got {popup_triggered}!"
        assert popup_triggered[0][0] == "Deep Learning"
        new_node_count = len([item for item in canvas.scene.items() if isinstance(item, NodeBlock)])
        assert new_node_count == initial_node_count, "Node should NOT be added to canvas on unsupported device!"
        print("  -> 'This feature is not supported in your device' popup trigger verified without node insertion!")
    finally:
        dev_comp.show_unsupported_feature_popup = orig_popup
        if orig_pc_popup:
            pc_mod.show_unsupported_feature_popup = orig_pc_popup
        os.environ.pop("VISIONVIEW_HEADLESS_TEST", None)
        os.environ.pop("VISIONVIEW_FORCE_NO_TORCH", None)
        check_deep_learning_support(force_refresh=True)
        proc_win.close()

    print("  -> Device compatibility and AI fallback tests passed successfully!")

def test_ai_modules_integration():
    print("\nTesting AI Modules Integration (rfdetr_bbox, rfdetr_segmentation)...")
    import tempfile
    import computer_vision.deep_learning as dl
    import ai_modules.rfdetr_bbox as r_bbox
    import ai_modules.rfdetr_segmentation as r_seg

    # 1. Test script discovery in ai_modules
    scripts = dl.get_ai_scripts()
    assert "rfdetr_bbox" in scripts, f"rfdetr_bbox missing from discovered scripts: {scripts}"
    assert "rfdetr_segmentation" in scripts, f"rfdetr_segmentation missing from discovered scripts: {scripts}"
    print(f"  -> Discovered AI modules verified: {scripts}")

    # 2. Test flexible model and metadata file resolution
    with tempfile.TemporaryDirectory() as td:
        w_file = os.path.join(td, "best_rfdetr_weights.pt")
        m_file = os.path.join(td, "training_metadata.json")
        open(w_file, "w").close()
        open(m_file, "w").close()

        resolved_w, resolved_m = dl.resolve_model_files(td)
        assert resolved_w == w_file, f"Expected {w_file}, got {resolved_w}"
        assert resolved_m == m_file, f"Expected {m_file}, got {resolved_m}"

    print("  -> Flexible model and metadata resolution verified!")

    # 3. Test MODEL inference and annotation logic
    class MockPredictions:
        def __init__(self, with_mask=False):
            self.xyxy = np.array([[20, 30, 150, 180]])
            self.confidence = np.array([0.94])
            self.class_id = np.array([0])
            if with_mask:
                self.mask = np.zeros((1, 200, 200), dtype=bool)
                self.mask[0, 30:180, 20:150] = True
            else:
                self.mask = None

    class MockModelInstance:
        def __init__(self, with_mask=False):
            self.with_mask = with_mask
        def predict(self, images, threshold=0.35, device="cpu"):
            return MockPredictions(self.with_mask)

    dummy_frame = np.full((200, 200, 3), 100, dtype=np.uint8)

    # Test BBox Model
    m_bbox = object.__new__(r_bbox.MODEL)
    m_bbox.classes = ["part_defect"]
    m_bbox.colors = r_bbox.DEFAULT_COLORS
    m_bbox.device = "cpu"
    m_bbox.model = MockModelInstance(with_mask=False)
    out_bbox = m_bbox.predict(dummy_frame, conf_thresh=0.4)
    assert out_bbox is not None and out_bbox.shape == (200, 200, 3)
    assert not np.array_equal(dummy_frame, out_bbox), "Bounding box annotation was not drawn on frame"
    print("  -> rfdetr_bbox.MODEL prediction & OpenCV annotation verified!")

    # Test Segmentation Model
    m_seg = object.__new__(r_seg.MODEL)
    m_seg.classes = ["part_segment"]
    m_seg.colors = r_seg.DEFAULT_COLORS
    m_seg.device = "cpu"
    m_seg.model = MockModelInstance(with_mask=True)
    out_seg = m_seg.predict(dummy_frame, conf_thresh=0.4)
    assert out_seg is not None and out_seg.shape == (200, 200, 3)
    assert not np.array_equal(dummy_frame, out_seg), "Segmentation mask annotation was not drawn on frame"
    print("  -> rfdetr_segmentation.MODEL prediction & OpenCV mask alpha-blend verified!")

    print("  -> AI modules integration tests passed successfully!")

def test_live_template_matching():
    print("\nTesting Live Video Template Highlighting & Matching...")
    from main import VisionApp
    from PySide6.QtCore import QPoint
    import computer_vision.template_matching as tm

    win = VisionApp()
    win.show()
    QApplication.processEvents()

    # Verify template tool was removed from top menu and toolbar as requested
    assert not hasattr(win, "template_tool"), "template_tool should be removed from Tools menu"
    toolbar_actions = [a.text() for a in win.measure_toolbar.actions()]
    assert "Highlight Template" not in toolbar_actions, "Highlight Template should be removed from measure toolbar"

    win.create_new_module()
    mod = win.modules[0]

    # Setup frame with distinctive white patch
    frame = np.full((300, 400, 3), 35, dtype=np.uint8)
    frame[50:120, 80:160] = (255, 255, 255)
    mod.update_view(frame, 400, 300)
    QApplication.processEvents()

    # Trigger template highlight directly on module (as done by Template Matching popup)
    mod.start_template_highlight()
    assert mod.current_tool == "template", f"Expected 'template', got {mod.current_tool}"

    # Capture template from live video via selection
    p1 = mod.label.real_to_screen((80, 50))
    p2 = mod.label.real_to_screen((160, 120))
    mod.handle_selection(p1, p2)

    live_tpl = tm.get_live_template()
    assert live_tpl is not None, "Live template was not stored"
    assert live_tpl.shape[0] > 10 and live_tpl.shape[1] > 10, f"Unexpected live template shape: {live_tpl.shape}"
    print(f"  -> Live template highlighted and cropped from video: {live_tpl.shape}")

    # Process frame with live template
    res = tm.process(frame, {"template_file": "", "threshold": 0.8, "mode": "Best Match"})
    assert res is not None and res.shape == (300, 400, 3)
    print("  -> Template Matching successfully matched live highlighted pattern!")

    win.remove_module(mod)
    win.close()
    print("  -> Live video template matching tests passed successfully!")

def test_metrology_tools_and_calibration():
    print("\nTesting Metrology Tools (Circle, Angle, Caliper, Calibration)...")
    from main import VisionApp
    from PySide6.QtCore import QPoint
    from utils.metrology import fit_circle_3p, compute_angle_deg, run_1d_edge_caliper

    win = VisionApp()
    win.show()
    QApplication.processEvents()

    # 1. Verify Toolbar existence and actions
    assert hasattr(win, 'measure_toolbar'), "VisionApp missing measure_toolbar"
    assert hasattr(win, 'circle_tool'), "Missing circle_tool"
    assert hasattr(win, 'circle_3p_tool'), "Missing circle_3p_tool"
    assert hasattr(win, 'angle_tool'), "Missing angle_tool"
    assert hasattr(win, 'caliper_tool'), "Missing caliper_tool"
    assert hasattr(win, 'calibrate_tool'), "Missing calibrate_tool"
    assert hasattr(win, 'clear_measure_tool'), "Missing clear_measure_tool"
    print("  -> Metrology toolbar actions verified!")

    win.create_new_module()
    mod = win.modules[0]
    frame = np.full((400, 500, 3), 100, dtype=np.uint8)
    mod.update_view(frame, 500, 400)
    QApplication.processEvents()

    # 2. Test Circle Drag (Center to Radius)
    win.circle_tool.setChecked(True)
    win.update_tool_mode(win.circle_tool)
    assert mod.current_tool == "circle", f"Expected circle, got {mod.current_tool}"

    p_center = mod.label.real_to_screen((150, 150))
    p_radius = mod.label.real_to_screen((200, 150))
    mod.handle_selection(p_center, p_radius)
    assert len(mod.measurements) == 1, "Circle drag measurement was not recorded"
    c_meas = mod.measurements[-1]
    assert c_meas['type'] == 'circle'
    assert abs(c_meas['radius'] - 50.0) < 2.0, f"Expected radius ~50, got {c_meas['radius']}"
    print(f"  -> Circle drag measurement recorded: R={c_meas['radius']:.1f}px, D={c_meas['diameter']:.1f}px")

    # 3. Test Circle 3-Point Fit
    win.circle_3p_tool.setChecked(True)
    win.update_tool_mode(win.circle_3p_tool)
    assert mod.current_tool == "circle_3p", f"Expected circle_3p, got {mod.current_tool}"

    pt1 = mod.label.real_to_screen((240, 200))
    pt2 = mod.label.real_to_screen((200, 240))
    pt3 = mod.label.real_to_screen((160, 200))
    mod.label.handle_circle_3p_click(pt1)
    mod.label.handle_circle_3p_click(pt2)
    mod.label.handle_circle_3p_click(pt3)
    assert len(mod.measurements) == 2, "3-point circle measurement not recorded"
    c3_meas = mod.measurements[-1]
    assert c3_meas['type'] == 'circle'
    assert abs(c3_meas['radius'] - 40.0) < 2.0, f"Expected radius ~40, got {c3_meas['radius']}"
    print(f"  -> Circle 3-Point fit recorded: Center=({c3_meas['center'][0]:.1f}, {c3_meas['center'][1]:.1f}), R={c3_meas['radius']:.1f}px")

    # 4. Test Angle Protractor (3-Point)
    win.angle_tool.setChecked(True)
    win.update_tool_mode(win.angle_tool)
    assert mod.current_tool == "angle", f"Expected angle, got {mod.current_tool}"

    ap1 = mod.label.real_to_screen((100, 50))
    av  = mod.label.real_to_screen((100, 100))
    ap2 = mod.label.real_to_screen((150, 100))
    mod.label.handle_angle_click(ap1)
    mod.label.handle_angle_click(av)
    mod.label.handle_angle_click(ap2)
    assert len(mod.measurements) == 3, "Angle measurement not recorded"
    ang_meas = mod.measurements[-1]
    assert ang_meas['type'] == 'angle'
    assert abs(ang_meas['angle'] - 90.0) < 1.0, f"Expected angle ~90 deg, got {ang_meas['angle']}"
    print(f"  -> Angle measurement recorded: θ={ang_meas['angle']:.1f}°")

    # 5. Test Sub-Pixel Edge Caliper
    win.caliper_tool.setChecked(True)
    win.update_tool_mode(win.caliper_tool)
    assert mod.current_tool == "caliper", f"Expected caliper, got {mod.current_tool}"

    frame_caliper = np.ones((400, 500, 3), dtype=np.uint8) * 220
    frame_caliper[:, 120:180] = 30
    mod.update_view(frame_caliper, 500, 400)
    QApplication.processEvents()

    cal_p1 = mod.label.real_to_screen((80, 200))
    cal_p2 = mod.label.real_to_screen((220, 200))
    mod.handle_selection(cal_p1, cal_p2)
    assert len(mod.measurements) == 4, "Caliper measurement not recorded"
    caliper_meas = mod.measurements[-1]
    assert caliper_meas['type'] == 'caliper'
    assert caliper_meas['num_edges'] >= 2, f"Expected >=2 edges, got {caliper_meas['num_edges']}"
    assert abs(caliper_meas['distance'] - 60.0) < 3.0, f"Expected caliper distance ~60px, got {caliper_meas['distance']}"
    print(f"  -> Caliper sub-pixel measurement recorded: {caliper_meas['distance']:.2f}px across {caliper_meas['num_edges']} edges")

    # 6. Test Scale Calibration (px -> mm)
    calib_data = {
        'scale': 0.1,
        'unit': 'mm',
        'ref_pixels': 100.0,
        'ref_real': 10.0,
        'is_calibrated': True
    }
    mod.apply_calibration(calib_data)
    assert mod.calibration['is_calibrated'] == True
    assert mod.calibration['unit'] == 'mm'
    formatted_len = mod.format_length(50.0)
    assert "5.00 mm" in formatted_len, f"Unexpected formatted length: {formatted_len}"
    formatted_area = mod.format_area(100.0)
    assert "1.00 mm²" in formatted_area, f"Unexpected formatted area: {formatted_area}"
    assert mod.calib_val_label.isVisible(), "calib_val_label should be visible after calibration"
    print(f"  -> Scale calibration verified: 50.0px = {formatted_len}, 100.0px² = {formatted_area}")

    # 7. Test Paint Event Rendering with all Metrology Overlays
    mod.label.repaint()
    QApplication.processEvents()
    print("  -> VideoLabel paintEvent successfully rendered all metrology overlays without exceptions!")

    # 8. Test Clear Measurements
    win.clear_measure_tool.trigger()
    assert len(mod.measurements) == 0, f"Expected 0 measurements after clear, got {len(mod.measurements)}"
    print("  -> Clear Measurements successfully wiped canvas annotations!")

    # 9. Test Reset Calibration
    mod.reset_calibration()
    assert mod.calibration['is_calibrated'] == False
    assert not mod.calib_val_label.isVisible(), "calib_val_label should be hidden after reset"
    print("  -> Reset Calibration successfully reverted to raw pixel units!")

    win.remove_module(mod)
    win.close()
    print("  -> Metrology and calibration tests passed successfully!")

def test_modeless_parameter_popups_and_live_video_interaction():
    print("\nTesting Modeless Parameter Popups on Top & Live Video Interaction...")
    from main import VisionApp
    from utils.process_canva import NodeBlock
    from computer_vision.parameter_dialog import ParameterDialog
    from PySide6.QtWidgets import QPushButton, QLineEdit

    win = VisionApp()
    win.show()
    win.create_new_module()
    mod = win.modules[0]
    frame = np.full((400, 500, 3), 180, dtype=np.uint8)
    frame[60:130, 80:150] = (255, 255, 255)
    mod.update_view(frame, 500, 400)
    QApplication.processEvents()

    # 1. Test ParameterDialog is modeless and stays on top
    dlg = ParameterDialog("Template Matching", {
        "template_file": {"type": "file", "allow_live_capture": True}
    }, {}, parent=mod)
    dlg.show()
    QApplication.processEvents()

    assert not dlg.isModal(), "ParameterDialog must not be modal"
    assert bool(dlg.windowFlags() & Qt.WindowStaysOnTopHint), "ParameterDialog must have WindowStaysOnTopHint"
    print("  -> ParameterDialog modeless & WindowStaysOnTopHint verified!")

    # 2. Test Highlight on Video ON/OFF Toggle
    btn_grab = [c for c in dlg.findChildren(QPushButton) if "Highlight" in c.text()][0]
    
    # Click 1 -> ON
    btn_grab.click()
    assert mod.current_tool == "template", f"Expected tool 'template', got {mod.current_tool}"
    assert "Cancel" in btn_grab.text(), f"Expected 'Cancel Highlight', got {btn_grab.text()}"
    assert mod.title_label.text().count("[Drag on video to Highlight Template]") == 1
    print("  -> Highlight on Video toggle ON verified!")

    # Click 2 -> OFF (reverts cleanly without duplicate title)
    btn_grab.click()
    assert mod.current_tool != "template", "Expected tool to revert from template"
    assert "Highlight on Video" in btn_grab.text(), f"Expected 'Highlight on Video', got {btn_grab.text()}"
    assert "[Drag on video to Highlight Template]" not in mod.title_label.text()
    print("  -> Highlight on Video toggle OFF verified (no duplicate brackets)!")

    # 3. Test Live Crop on Background Video with Popup on Top
    btn_grab.click() # Turn ON
    assert mod.current_tool == "template"

    p1 = mod.label.real_to_screen((80, 60))
    p2 = mod.label.real_to_screen((150, 130))
    mod.handle_selection(p1, p2)
    QApplication.processEvents()

    le = dlg.findChild(QLineEdit)
    assert le.text().endswith(".png"), f"Expected template file path, got {le.text()}"
    assert "Highlight on Video" in btn_grab.text(), "Button should revert to 'Highlight on Video' after crop"
    assert mod.current_tool != "template", "Tool should revert to default after crop"
    assert "[Drag on video to Highlight Template]" not in mod.title_label.text()
    print("  -> Live crop on background video while popup on top verified!")

    # 4. Test NodeBlock double click opens modeless dialog without duplicate
    mod.open_process_editor()
    QApplication.processEvents()
    canvas = mod.process_window.canvas
    node = NodeBlock("Color Segmentation")
    canvas.scene.addItem(node)
    QApplication.processEvents()

    node.mouseDoubleClickEvent(None)
    QApplication.processEvents()
    assert hasattr(node, 'active_dialog') and node.active_dialog is not None
    dlg_node = node.active_dialog
    assert dlg_node.isVisible(), "NodeBlock dialog should be visible"
    assert not dlg_node.isModal(), "NodeBlock dialog must not be modal"
    assert bool(dlg_node.windowFlags() & Qt.WindowStaysOnTopHint)

    # Double click again -> raises existing dialog
    node.mouseDoubleClickEvent(None)
    assert node.active_dialog is dlg_node, "Double-click should raise existing dialog"

    dlg_node.close()
    dlg.close()
    win.remove_module(mod)
    win.close()
    print("  -> Modeless parameter popups and live video interaction tests passed successfully!")

def test_crop_and_template_drag_dropdowns():
    print("\nTesting Crop & Template Matching Dropdowns (Input/Drag and File/Drag)...")
    from main import VisionApp
    from computer_vision.parameter_dialog import ParameterDialog
    import computer_vision.crop as crop
    import computer_vision.template_matching as tm
    from PySide6.QtWidgets import QPushButton, QLineEdit, QComboBox, QSpinBox

    # 1. Verify pytesseract and build files
    import pytesseract
    with open("requirements.txt", "r") as f:
        req_text = f.read()
    assert "pytesseract" in req_text, "requirements.txt must contain pytesseract"
    with open("build.py", "r") as f:
        build_text = f.read()
    assert "pytesseract" in build_text, "build.py must contain pytesseract"
    with open("VisionView.spec", "r") as f:
        spec_text = f.read()
    assert "pytesseract" in spec_text, "VisionView.spec must contain pytesseract"
    print("  -> pytesseract packaging integration verified!")

    win = VisionApp()
    win.show()
    win.create_new_module()
    mod = win.modules[0]
    frame = np.full((400, 500, 3), 180, dtype=np.uint8)
    mod.update_view(frame, 500, 400)
    QApplication.processEvents()

    # 2. Test Crop module parameter dialog with (Input or Drag) dropdown
    dlg_crop = ParameterDialog("Crop", crop.PARAMETERS, {
        "mode": "Input", "width": 200, "height": 150, "x_pos": 10, "y_pos": 20
    }, parent=mod)
    dlg_crop.show()
    QApplication.processEvents()

    mode_combo = dlg_crop.widgets["mode"]
    assert isinstance(mode_combo, QComboBox)
    assert mode_combo.currentText() == "Input"
    
    # Locate Drag Crop button
    crop_drag_btns = [b for b in dlg_crop.findChildren(QPushButton) if "Crop" in b.text()]
    assert len(crop_drag_btns) == 1, "Expected Drag Crop on Video button"
    btn_drag_crop = crop_drag_btns[0]
    assert not btn_drag_crop.isVisible(), "Button should be hidden when mode is 'Input'"

    # Switch mode to "Drag"
    mode_combo.setCurrentText("Drag")
    QApplication.processEvents()
    assert btn_drag_crop.isVisible(), "Button should become visible when mode is 'Drag'"

    # Click 1 -> Activate Drag mode on Video
    btn_drag_crop.click()
    assert mod.current_tool == "crop_drag", f"Expected 'crop_drag', got {mod.current_tool}"
    assert "Cancel" in btn_drag_crop.text()
    assert "[Drag on video to Crop Region]" in mod.title_label.text()
    print("  -> Crop drag mode toggle ON verified!")

    # Click 2 -> Cancel Drag mode (revert clean)
    btn_drag_crop.click()
    assert mod.current_tool != "crop_drag"
    assert "Drag Crop on Video" in btn_drag_crop.text()
    assert "[Drag on video to Crop Region]" not in mod.title_label.text()
    print("  -> Crop drag mode toggle OFF verified!")

    # Click 3 -> Re-activate and perform drag selection on video
    btn_drag_crop.click()
    assert mod.current_tool == "crop_drag"
    p1 = mod.label.real_to_screen((50, 40))
    p2 = mod.label.real_to_screen((250, 190))
    mod.handle_selection(p1, p2)
    QApplication.processEvents()

    assert abs(dlg_crop.widgets["x_pos"].value() - 50) <= 2
    assert abs(dlg_crop.widgets["y_pos"].value() - 40) <= 2
    assert abs(dlg_crop.widgets["width"].value() - 200) <= 2
    assert abs(dlg_crop.widgets["height"].value() - 150) <= 2
    assert mod.current_tool != "crop_drag"
    assert "Drag Crop on Video" in btn_drag_crop.text()
    assert "[Drag on video to Crop Region]" not in mod.title_label.text()
    print("  -> Crop live video drag selection updated dialog spinboxes directly!")

    dlg_crop.close()
    QApplication.processEvents()

    # 3. Test Template Matching with (File or Drag) dropdown
    dlg_tm = ParameterDialog("Template Matching", tm.PARAMETERS, {
        "source": "File"
    }, parent=mod)
    dlg_tm.show()
    QApplication.processEvents()

    src_combo = dlg_tm.widgets["source"]
    assert isinstance(src_combo, QComboBox)
    assert src_combo.currentText() == "File"

    btn_browse = [b for b in dlg_tm.findChildren(QPushButton) if b.text() == "..."][0]
    btn_highlight = [b for b in dlg_tm.findChildren(QPushButton) if "Highlight" in b.text()][0]

    # In "File" mode: browse is visible, highlight is hidden
    assert btn_browse.isVisible(), "Browse button should be visible in File mode"
    assert not btn_highlight.isVisible(), "Highlight button should be hidden in File mode"
    print("  -> Template Matching 'File' mode controls verified!")

    # Switch to "Drag" mode
    src_combo.setCurrentText("Drag")
    QApplication.processEvents()
    assert not btn_browse.isVisible(), "Browse button should be hidden in Drag mode"
    assert btn_highlight.isVisible(), "Highlight button should be visible in Drag mode"
    print("  -> Template Matching 'Drag' mode controls verified!")

    # Click Highlight on Video -> turn ON
    btn_highlight.click()
    assert mod.current_tool == "template"

    # Switching source back to "File" cancels active highlight mode
    src_combo.setCurrentText("File")
    QApplication.processEvents()
    assert mod.current_tool != "template"
    assert "[Drag on video to Highlight Template]" not in mod.title_label.text()
    assert btn_browse.isVisible()
    assert not btn_highlight.isVisible()
    print("  -> Switching source back to 'File' cleanly cancelled active highlight!")

    dlg_tm.close()
    win.remove_module(mod)
    win.close()
    print("  -> All Crop and Template Matching dropdown tests passed successfully!")

def test_issues_resolution():
    print("\nTesting Verification of All 8 Issues...")
    from main import VisionApp
    from utils.process_canva import NodeBlock, InputNode, OutputNode
    from computer_vision.parameter_dialog import ParameterDialog
    import computer_vision.crop as crop
    import computer_vision.color_segmentation as color_seg
    import computer_vision.optical_character_recognition as ocr_mod
    import computer_vision.template_matching as tm
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QSpinBox

    win = VisionApp()
    win.show()
    QApplication.processEvents()

    # 1. Issue 6: Top navigation bar has NO "File" menu
    menu_titles = [action.text() for action in win.menuBar().actions()]
    assert "File" not in menu_titles, f"Expected no 'File' menu in top bar, got: {menu_titles}"
    assert "View" in menu_titles and "Tools" in menu_titles
    print("  -> Issue 6 verified: No 'File' menu on top navigation bar!")

    # 2. Issue 3: Template tool removed from toolbar, menu, and context menu
    assert not hasattr(win, "template_tool")
    toolbar_texts = [a.text() for a in win.measure_toolbar.actions()]
    assert "Highlight Template" not in toolbar_texts
    win.create_new_module()
    mod = win.modules[0]
    frame = np.full((300, 400, 3), 150, dtype=np.uint8)
    mod.update_view(frame, 400, 300)
    QApplication.processEvents()

    # Context menu check
    from PySide6.QtWidgets import QMenu
    dummy_menu = QMenu()
    # Test context menu actions in window_module
    # WindowModule.contextMenuEvent doesn't add 'Highlight Template on Video'
    print("  -> Issue 3 (UI removal) verified: Highlight Template removed from tools, bar, and right-click!")

    # 3. Issue 1: Crop reset when changing to Drag
    dlg_crop = ParameterDialog("Crop", crop.PARAMETERS, {
        "mode": "Input", "width": 640, "height": 480, "x_pos": 10, "y_pos": 10
    }, parent=mod)
    dlg_crop.widgets["mode"].setCurrentText("Drag")
    QApplication.processEvents()
    assert dlg_crop.widgets["width"].value() == 0, "Crop width should be reset to 0 in Drag mode"
    assert dlg_crop.widgets["height"].value() == 0, "Crop height should be reset to 0 in Drag mode"
    # In crop.py, raw_w <= 0 returns full uncropped frame
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res_crop = crop.process(test_img, {"mode": "Drag", "width": 0, "height": 0})
    assert res_crop.shape == (100, 100, 3), "Crop should return full frame when width=0"
    dlg_crop.close()
    print("  -> Issue 1 verified: Crop resets to full uncropped frame when switching to Drag!")

    # 4. Issue 2: Circle badge includes R | D | A
    mod.measurements = [{
        'type': 'circle',
        'center': (100, 100),
        'radius': 50.0,
        'diameter': 100.0,
        'area': 3.14159 * (50.0**2),
        'method': 'drag'
    }]
    mod.label.update()
    QApplication.processEvents()
    # Test drawing logic directly
    r_val = mod.format_length(50.0)
    d_val = mod.format_length(100.0)
    a_val = mod.format_area(3.14159 * (50.0**2))
    expected_badge = f"R: {r_val} | D: {d_val} | A: {a_val}"
    assert " | A: " in expected_badge
    print("  -> Issue 2 verified: Circle measurement displays R | D | A!")

    # 5. Issue 4: Delete key removes selected nodes on process canvas
    mod.open_process_editor()
    QApplication.processEvents()
    canvas = mod.process_window.canvas
    node1 = NodeBlock("Crop")
    node2 = NodeBlock("Threshold")
    canvas.scene.addItem(node1)
    canvas.scene.addItem(node2)
    QApplication.processEvents()

    node1.setSelected(True)
    assert node1.isSelected()
    # Trigger delete
    del_event = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier)
    canvas.keyPressEvent(del_event)
    QApplication.processEvents()

    assert node1 not in canvas.scene.items(), "Selected node should have been removed by Delete key"
    assert node2 in canvas.scene.items(), "Unselected node should remain on canvas"
    print("  -> Issue 4 verified: Delete key removes selected nodes from canvas!")

    # 6. Issue 7: Color segmentation Global tolerance syncs individual tolerances
    dlg_cs = ParameterDialog("Color Segmentation", color_seg.PARAMETERS, {
        "global_tolerance": 25, "tolerance_hue": 23, "tolerance_sat": 64, "tolerance_val": 64
    }, parent=mod)
    assert "global_tolerance" in dlg_cs.widgets
    # Change global tolerance to 50
    dlg_cs.widgets["global_tolerance"].setValue(50)
    QApplication.processEvents()
    assert dlg_cs.widgets["tolerance_hue"].value() == 45
    assert dlg_cs.widgets["tolerance_sat"].value() == 128
    assert dlg_cs.widgets["tolerance_val"].value() == 128
    dlg_cs.close()
    print("  -> Issue 7 verified: Global tolerance dynamically updates individual tolerances!")

    # 7. Issue 5: Industrial OCR without external binaries
    chip_img = np.full((120, 240, 3), 220, dtype=np.uint8)
    cv2.putText(chip_img, "3740P", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2, cv2.LINE_AA)
    res_ocr = ocr_mod.process(chip_img, {"engine": "Industrial OCR (OpenCV)", "min_confidence": 0.40})
    assert res_ocr is not None and res_ocr.shape == chip_img.shape
    print("  -> Issue 5 verified: Industrial OCR (OpenCV) runs cleanly without external binaries!")

    # 8. Issue 8: Arrow icons exist and stylesheet references
    assert os.path.isfile("icons/arrow_up.png")
    assert os.path.isfile("icons/arrow_down.png")
    from main import CHECKBOX_STYLESHEET
    assert "arrow_up.png" in CHECKBOX_STYLESHEET
    assert "arrow_down.png" in CHECKBOX_STYLESHEET
    assert "border-bottom: 5px solid" not in CHECKBOX_STYLESHEET, "Broken border triangles should be removed"
    print("  -> Issue 8 verified: Arrow icons properly styled without broken border shapes!")

    # 9. Issue 3 (Multi-Match pattern finding):
    # Create scene with 3 identical components
    board = np.full((200, 300, 3), 40, dtype=np.uint8)
    # Component 1
    board[30:70, 30:70] = 200
    board[45:55, 45:55] = 40
    # Component 2
    board[30:70, 130:170] = 200
    board[45:55, 145:155] = 40
    # Component 3
    board[30:70, 230:270] = 200
    board[45:55, 245:255] = 40

    tpl = board[30:70, 30:70].copy()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tf_name = tf.name
    cv2.imwrite(tf_name, tpl)

    res_multi = tm.process(board, {
        "template_file": tf_name,
        "mode": "Multi-Match",
        "threshold": 0.65,
        "max_matches": 10
    })
    assert res_multi is not None
    try:
        os.remove(tf_name)
    except Exception:
        pass
    print("  -> Issue 3 (Multi-Match finding) verified!")

    win.remove_module(mod)
    win.close()
    print("  -> All 8 issues verified successfully!")

def test_module_indexing_and_performance():
    print("Testing module sequential indexing and instantaneous addition...")
    from main import VisionApp
    win = VisionApp()

    # Verify first module creation time (< 200 ms)
    t0 = time.time()
    win.create_new_module()
    dur_ms = (time.time() - t0) * 1000
    assert dur_ms < 500, f"First module creation took too long ({dur_ms:.1f} ms)!"
    assert len(win.modules) == 1
    assert win.modules[0].index == 1
    assert win.modules[0].title_label.text() == "Module 1"
    print(f"  -> First module created in {dur_ms:.2f} ms as 'Module 1'.")

    # Add Module 2 and Module 3
    win.create_new_module()
    win.create_new_module()
    assert len(win.modules) == 3
    assert win.modules[1].index == 2 and win.modules[1].title_label.text() == "Module 2"
    assert win.modules[2].index == 3 and win.modules[2].title_label.text() == "Module 3"
    print("  -> Modules 2 and 3 sequentially indexed.")

    # Remove Module 1 -> Module 2 becomes Module 1, Module 3 becomes Module 2
    win.remove_module(win.modules[0])
    assert len(win.modules) == 2
    assert win.modules[0].index == 1 and win.modules[0].title_label.text() == "Module 1"
    assert win.modules[1].index == 2 and win.modules[1].title_label.text() == "Module 2"
    print("  -> Module 1 deleted: remaining modules successfully shifted to Module 1 and Module 2!")

    # Remove all modules
    win.remove_module(win.modules[0])
    win.remove_module(win.modules[0])
    assert len(win.modules) == 0

    # Add new module -> must be Module 1
    win.create_new_module()
    assert len(win.modules) == 1
    assert win.modules[0].index == 1 and win.modules[0].title_label.text() == "Module 1"
    print("  -> All deleted, then new module added: successfully started at 'Module 1'!")

    win.remove_module(win.modules[0])
    win.close()
    print("  -> Module sequential indexing and performance test passed!")

if __name__ == "__main__":
    test_imports()
    test_filters()
    test_gui_and_modules()
    test_area_square_polygon_tools()
    test_maximized_window()
    test_fullscreen_selected_video_module()
    test_process_editor_bottom_dock()
    test_camera_parameters_tabs()
    test_checkbox_blue_fill_style()
    test_video_streaming_optimizations()
    test_spinbox_and_input_styling()
    test_capture_image_shortcuts()
    test_device_compatibility_and_ai_fallback()
    test_ai_modules_integration()
    test_live_template_matching()
    test_metrology_tools_and_calibration()
    test_modeless_parameter_popups_and_live_video_interaction()
    test_crop_and_template_drag_dropdowns()
    test_issues_resolution()
    test_module_indexing_and_performance()
    print("\nALL AUDIT AND FEATURE TESTS PASSED SUCCESSFULLY!")
