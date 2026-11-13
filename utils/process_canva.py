import os
import importlib
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QGraphicsView,
    QGraphicsScene, QVBoxLayout, QLabel,
    QGraphicsRectItem, QGraphicsTextItem,
    QGraphicsEllipseItem, QGraphicsPathItem,
    QGraphicsProxyWidget, QGraphicsItem, QPushButton,
    QMenu, QToolButton
)

from PySide6.QtGui import QColor, QPainter, QPen, QPainterPath, QDrag
from PySide6.QtCore import Qt, QPointF, QMimeData

from computer_vision.parameter_dialog import ParameterDialog
from utils.device_compatibility import check_deep_learning_support, show_unsupported_feature_popup

# -------------------------------
# PROCESS WINDOW
# -------------------------------

class ProcessWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.window_module = parent

        self.setWindowTitle("Vision Process Editor")

        # 1. Create vertical layout for the widget
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 2. Setup the splitter (Canvas + Module List)
        splitter = QSplitter(Qt.Horizontal)
        self.canvas = NodeCanvas(self)
        self.module_list = ModuleList(self.canvas)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.module_list)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([900, 300])

        layout.addWidget(splitter)

        self.send_to_worker()

    def setCentralWidget(self, widget):
        pass

    def send_to_worker(self):
        if self.window_module and self.window_module.worker:
            sequence = self.canvas.get_pipeline_sequence()
            # Directly update the worker's pipeline list
            self.window_module.worker.pipeline = sequence
            print(f"Worker pipeline updated: {sequence}")

    def closeEvent(self, event):
        # Tell the parent module to save the current canvas state
        if self.window_module and hasattr(self, 'canvas'):
            self.window_module.saved_nodes = self.canvas.get_save_data()
        event.accept()

    def hideEvent(self, event):
        if self.window_module and hasattr(self, 'canvas'):
            self.window_module.saved_nodes = self.canvas.get_save_data()
        super().hideEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if hasattr(self, 'canvas') and self.canvas:
                self.canvas.delete_selected_items()
                event.accept()
                return
        super().keyPressEvent(event)

# -------------------------------
# NODE CANVAS
# -------------------------------

class NodeCanvas(QGraphicsView):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process_window = parent

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.setRenderHint(QPainter.Antialiasing)

        self.setBackgroundBrush(QColor("#2b2b2b"))

        self.connection_start = None
        self.temp_line = None

        self.setAcceptDrops(True)

        self.temp_line = None

        self.create_default_nodes()

    def create_default_nodes(self):

        input_node = InputNode()
        output_node = OutputNode()

        self.scene.addItem(input_node)
        self.scene.addItem(output_node)

        input_node.setPos(-200, 0)
        output_node.setPos(400, 0)

    # -------------------
    # DRAG DROP SUPPORT
    # -------------------

    def dragEnterEvent(self, event):
        event.accept()

    def dragMoveEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        name = event.mimeData().text()
        clean_name = name.replace(" (Unsupported)", "").strip()

        if clean_name.lower() == "deep learning":
            supported, reason, _ = check_deep_learning_support()
            if not supported:
                parent_win = getattr(self, 'process_window', None) or self.window()
                show_unsupported_feature_popup(parent=parent_win, feature_name="Deep Learning", reason=reason)
                return

        node = NodeBlock(clean_name)
        pos = self.mapToScene(event.position().toPoint())
        self.scene.addItem(node)
        node.setPos(pos)

    # -------------------
    # ZOOM
    # -------------------

    def wheelEvent(self, event):

        zoom_factor = 1.25

        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1/zoom_factor, 1/zoom_factor)

    # -------------------
    # GRID
    # -------------------

    def drawBackground(self, painter, rect):

        super().drawBackground(painter, rect)

        grid = 30
        pen = QPen(QColor("#3a3a3a"))
        painter.setPen(pen)

        left = int(rect.left()) - (int(rect.left()) % grid)
        top = int(rect.top()) - (int(rect.top()) % grid)

        x = left
        while x < rect.right():
            painter.drawLine(x, rect.top(), x, rect.bottom())
            x += grid

        y = top
        while y < rect.bottom():
            painter.drawLine(rect.left(), y, rect.right(), y)
            y += grid

    def mouseMoveEvent(self, event):

        if self.connection_start:

            pos = self.mapToScene(event.position().toPoint())

            if not self.temp_line:
                self.temp_line = self.scene.addPath(QPainterPath())

            start = self.connection_start.scenePos()

            path = QPainterPath()
            path.moveTo(start)

            dx = (pos.x() - start.x()) * 0.5

            path.cubicTo(
                start.x()+dx, start.y(),
                pos.x()-dx, pos.y(),
                pos.x(), pos.y()
            )

            self.temp_line.setPath(path)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.connection_start:
            pos = self.mapToScene(event.position().toPoint())
            items = self.scene.items(pos)

            target_socket = None
            for item in items:
                if isinstance(item, NodeSocket) and item.socket_type == "input":
                    target_socket = item
                    break

            if target_socket and self.connection_start.node != target_socket.node:
                # Remove any existing connection to target_socket (single-input rule)
                for existing_conn in list(target_socket.connections):
                    if existing_conn.start_socket and existing_conn in existing_conn.start_socket.connections:
                        existing_conn.start_socket.connections.remove(existing_conn)
                    if existing_conn in target_socket.connections:
                        target_socket.connections.remove(existing_conn)
                    if existing_conn.scene():
                        self.scene.removeItem(existing_conn)

                connection = NodeConnection(self.connection_start, target_socket)
                self.scene.addItem(connection)

                self.connection_start.connections.append(connection)
                target_socket.connections.append(connection)
                self.update_pipeline_live()

            # remove temporary line
            if self.temp_line:
                if self.temp_line.scene():
                    self.scene.removeItem(self.temp_line)
                self.temp_line = None

            self.connection_start = None

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected_items()
            event.accept()
            return
        super().keyPressEvent(event)

    def delete_selected_items(self):
        """Removes all currently selected NodeBlocks and Connections, keeps Input/Output nodes"""
        selected = self.scene.selectedItems()
        if not selected:
            return

        nodes_to_remove = [item for item in selected if isinstance(item, NodeBlock) and not isinstance(item, (InputNode, OutputNode))]
        conns_to_remove = [item for item in selected if isinstance(item, NodeConnection)]

        for node in nodes_to_remove:
            node.remove_self()

        for conn in conns_to_remove:
            if conn.scene() is not None:
                if hasattr(conn, 'start_socket') and conn in getattr(conn.start_socket, 'connections', []):
                    conn.start_socket.connections.remove(conn)
                if hasattr(conn, 'end_socket') and conn in getattr(conn.end_socket, 'connections', []):
                    conn.end_socket.connections.remove(conn)
                self.scene.removeItem(conn)

        self.update_pipeline_live()

    def update_pipeline_live(self):
        """Helper to find the parent window and trigger send_to_worker"""
        proc_win = getattr(self, 'process_window', None)
        if proc_win is None and callable(getattr(self, 'window', None)):
            proc_win = self.window()
        if hasattr(proc_win, 'send_to_worker'):
            proc_win.send_to_worker()

# Inside NodeCanvas class
    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())
        
        # If we clicked a NodeBlock, let it handle its own menu (Remove Node)
        if isinstance(item, NodeBlock):
            super().contextMenuEvent(event)
            return

        # Canvas Right-Click Menu
        menu = QMenu()
        auto_connect_action = menu.addAction("Auto Connect Nodes")
        clear_action = menu.addAction("Clear All Blocks") # New Option
        
        action = menu.exec(event.globalPos())
        
        if action == auto_connect_action:
            self.auto_connect_pipeline()
        elif action == clear_action:
            self.clear_canvas()

    def clear_canvas(self):
        """Removes all NodeBlocks and Connections, keeps Input/Output"""
        # 1. Collect items to remove (to avoid modifying list while iterating)
        to_remove = []
        for item in self.scene.items():
            if isinstance(item, (NodeBlock, NodeConnection)):
                to_remove.append(item)
        
        # 2. Clean up connections from sockets first
        for item in to_remove:
            if isinstance(item, NodeConnection):
                if item in item.start_socket.connections:
                    item.start_socket.connections.remove(item)
                if item in item.end_socket.connections:
                    item.end_socket.connections.remove(item)
            
            # 3. Remove from scene
            self.scene.removeItem(item)

        self.update_pipeline_live()            
        print("Canvas cleared. Filter pipeline reset.")

    def auto_connect_pipeline(self):
        # 1. Clear all existing connections in the scene
        for item in list(self.scene.items()):
            if isinstance(item, NodeConnection):
                # Clean up references in sockets first
                if item.start_socket and item in item.start_socket.connections:
                    item.start_socket.connections.remove(item)
                if item.end_socket and item in item.end_socket.connections:
                    item.end_socket.connections.remove(item)
                if item.scene():
                    self.scene.removeItem(item)

        # 2. Categorize nodes
        input_node = next((n for n in self.scene.items() if isinstance(n, InputNode)), None)
        output_node = next((n for n in self.scene.items() if isinstance(n, OutputNode)), None)
        blocks = [n for n in self.scene.items() if isinstance(n, NodeBlock)]

        # 3. Sort blocks based on their X position
        blocks.sort(key=lambda n: n.x())

        # 4. Create the chain
        if not input_node or not output_node:
            return

        current_output_socket = input_node.output_socket

        # Connect Input -> Block 1 -> Block 2 ...
        for block in blocks:
            connection = NodeConnection(current_output_socket, block.input_socket)
            self.scene.addItem(connection)
            current_output_socket.connections.append(connection)
            block.input_socket.connections.append(connection)
            
            # Move to the next link in the chain
            current_output_socket = block.output_socket

        # Finally, connect the last block to the Output Node
        last_connection = NodeConnection(current_output_socket, output_node.input_socket)
        self.scene.addItem(last_connection)
        current_output_socket.connections.append(last_connection)
        output_node.input_socket.connections.append(last_connection)
        self.update_pipeline_live()

    def get_pipeline_sequence(self):
        sequence = []
        input_node = next((item for item in self.scene.items() if isinstance(item, InputNode)), None)
        
        if not input_node:
            return sequence

        visited_nodes = set()
        current_socket = input_node.output_socket
        while current_socket and current_socket.connections:
            conn = current_socket.connections[0]
            if not conn.end_socket:
                break
            next_node = conn.end_socket.node
            
            if next_node in visited_nodes:
                print("Cycle detected in pipeline graph. Stopping traversal.")
                break
            visited_nodes.add(next_node)

            if isinstance(next_node, NodeBlock):
                sequence.append({
                    "name": next_node.title,
                    "params": dict(next_node.parameters)
                })
                current_socket = next_node.output_socket
            elif isinstance(next_node, OutputNode):
                break
            else:
                break
                
        return sequence

    def push_live_pipeline(self):
        sequence = self.get_pipeline_sequence()
        proc_win = getattr(self, 'process_window', None)
        if proc_win is None and callable(getattr(self, 'window', None)):
            proc_win = self.window()
        if hasattr(proc_win, 'window_module') and proc_win.window_module and proc_win.window_module.worker:
            proc_win.window_module.worker.pipeline = sequence

    def get_save_data(self):
        """Returns a list of dictionaries containing node positions, titles and parameters."""
        data = []
        for item in self.scene.items():
            if isinstance(item, NodeBlock):
                data.append({
                    'title': item.title,
                    'x': item.x(),
                    'y': item.y(),
                    'params': dict(item.parameters)
                })
        return data

    def restore_setup(self, node_data):
        """Recreates NodeBlocks from saved data and auto-connects them."""
        # 1. Clear existing blocks first (if any)
        self.clear_canvas() 

        # 2. Re-add saved blocks with restored parameters
        for node_info in node_data:
            node = NodeBlock(node_info['title'])
            if 'params' in node_info and isinstance(node_info['params'], dict):
                node.parameters.update(node_info['params'])
            self.scene.addItem(node)
            node.setPos(node_info['x'], node_info['y'])
        
        # 3. Automatically restore the connections based on position
        self.auto_connect_pipeline()

class InputNode(QGraphicsRectItem):

    def __init__(self):

        super().__init__(0, 0, 200, 80)

        self.title = "Input Node"

        self.setBrush(QColor("#2c3e50"))
        self.setPen(QPen(QColor("#5a5a5a"), 2))

        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)

        # Label
        self.label = QGraphicsTextItem("Camera Input", self)
        self.label.setDefaultTextColor(Qt.white)

        # center the label
        rect = self.boundingRect()
        self.label.setPos((rect.width() - self.label.boundingRect().width()) / 2, 10)

        # output socket
        self.output_socket = NodeSocket(self, "output")
        self.output_socket.setPos(200, 40)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            for conn in self.output_socket.connections:
                conn.update_path()

        return super().itemChange(change, value)
    
class OutputNode(QGraphicsRectItem):

    def __init__(self):

        super().__init__(0, 0, 200, 80)

        self.title = "Output Node"

        self.setBrush(QColor("#8e44ad"))
        self.setPen(QPen(QColor("#5a5a5a"), 2))

        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)

        # Label
        self.label = QGraphicsTextItem("Output", self)
        self.label.setDefaultTextColor(Qt.white)

        rect = self.boundingRect()
        self.label.setPos((rect.width() - self.label.boundingRect().width()) / 2, 10)

        # input socket
        self.input_socket = NodeSocket(self, "input")
        self.input_socket.setPos(0, 40)

    def itemChange(self, change, value):

        if change == QGraphicsItem.ItemPositionChange:
            for conn in self.input_socket.connections:
                conn.update_path()

        return super().itemChange(change, value)
# -------------------------------
# MODULE LIST
# -------------------------------
_CV_MODULES_CACHE = None

def get_cached_cv_modules():
    global _CV_MODULES_CACHE
    if _CV_MODULES_CACHE is not None:
        return _CV_MODULES_CACHE

    from pathlib import Path
    import sys

    folder = Path(__file__).resolve().parent.parent / "computer_vision"
    if not folder.exists() and getattr(sys, 'frozen', False):
        base = Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
        if (base / "computer_vision").exists():
            folder = base / "computer_vision"

    modules_to_add = []
    if folder.exists():
        for file in folder.iterdir():
            if file.suffix in [".py", ".pyc"]:
                name = file.stem
                if name not in ["parameter_dialog", "__init__"]:
                    display_name = name.replace("_", " ").title()
                    if display_name not in modules_to_add:
                        modules_to_add.append(display_name)

    if not modules_to_add:
        modules_to_add = [
            "Canny Edge Detection", "Clahe", "Color Segmentation",
            "Contour Analysis", "Crop", "Deep Learning", "Gaussian Blur",
            "Morphology", "Optical Character Recognition", "Resize", "Template Matching", "Threshold"
        ]

    modules_to_add.sort()
    _CV_MODULES_CACHE = modules_to_add
    return _CV_MODULES_CACHE

# List of CV modules on right panel
class ModuleList(QWidget):
    def __init__(self, canvas):
        super().__init__()

        self.canvas = canvas

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # Header
        self.label = QLabel("Computer Vision Modules")
        self.label.setStyleSheet("font-weight:bold; font-size:14px;")
        layout.addWidget(self.label)

        # List widget
        self.list_widget = DraggableListWidget(self)
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)

        layout.addWidget(self.list_widget)

        self.load_modules()

    def load_modules(self):
        """Load modules from the folder sorted alphabetically (cached)"""
        modules_to_add = get_cached_cv_modules()

        ai_supported, ai_reason, _ = check_deep_learning_support()
        for display_name in modules_to_add:
            item = QListWidgetItem(display_name)
            if display_name.lower() == "deep learning" and not ai_supported:
                item.setText(f"{display_name} (Unsupported)")
                item.setToolTip(f"Deep Learning is not supported on this device:\n{ai_reason}")
                item.setForeground(QColor("#888888"))
            self.list_widget.addItem(item)

    def on_item_double_clicked(self, item):
        """When an item is double-clicked, add it to the center of the canvas"""
        name = item.text()
        clean_name = name.replace(" (Unsupported)", "").strip()

        if clean_name.lower() == "deep learning":
            supported, reason, _ = check_deep_learning_support()
            if not supported:
                parent_win = getattr(self, 'process_window', None) or self.window()
                show_unsupported_feature_popup(parent=parent_win, feature_name="Deep Learning", reason=reason)
                return

        node = NodeBlock(clean_name)
        
        view_center = self.canvas.viewport().rect().center()
        scene_pos = self.canvas.mapToScene(view_center)
        
        self.canvas.scene.addItem(node)
        node.setPos(scene_pos)
        
        self.canvas.auto_connect_pipeline()

# -------------------------------
# DRAGGABLE LIST WIDGET
# -------------------------------

class DraggableListWidget(QListWidget):
    """A QListWidget that can drag its items with module names"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)

    def startDrag(self, supportedActions):
        """Start drag with module name as text"""
        item = self.currentItem()
        if not item:
            return

        mime_data = QMimeData()
        title_text = item.text()
        mime_data.setText(title_text)  # the node title

        drag = QDrag(self)
        drag.setMimeData(mime_data)

        drag.exec(Qt.MoveAction)

# -------------------------------
# NODE SOCKET
# -------------------------------

class NodeSocket(QGraphicsEllipseItem):

    def __init__(self, node, socket_type):

        super().__init__(-6,-6,12,12,node)

        self.node = node
        self.socket_type = socket_type
        self.connections = []

        color = "#3498db" if socket_type == "input" else "#e67e22"
        self.setBrush(QColor(color))

        self.setAcceptedMouseButtons(Qt.LeftButton)

    def mousePressEvent(self, event):
        views = self.scene().views() if self.scene() else []
        if views and self.socket_type == "output":
            views[0].connection_start = self
        event.accept()

    def mouseReleaseEvent(self, event):
        event.ignore()

# -------------------------------
# CONNECTION
# -------------------------------

class NodeConnection(QGraphicsPathItem):

    def __init__(self, start_socket, end_socket):

        super().__init__()

        self.start_socket = start_socket
        self.end_socket = end_socket

        self.setPen(QPen(QColor("#aaaaaa"),2))

        self.setAcceptHoverEvents(True)

        self.update_path()

    def update_path(self):

        p1 = self.start_socket.scenePos()
        p2 = self.end_socket.scenePos()

        path = QPainterPath()
        path.moveTo(p1)

        dx = (p2.x() - p1.x()) * 0.5

        path.cubicTo(
            p1.x()+dx, p1.y(),
            p2.x()-dx, p2.y(),
            p2.x(), p2.y()
        )

        self.setPath(path)

    def mousePressEvent(self, event):
        # Right click removes the connection
        if event.button() == Qt.RightButton:
            scene_ref = self.scene()
            if self.start_socket and self in self.start_socket.connections:
                self.start_socket.connections.remove(self)

            if self.end_socket and self in self.end_socket.connections:
                self.end_socket.connections.remove(self)

            if scene_ref:
                scene_ref.removeItem(self)
                if scene_ref.views():
                    canvas = scene_ref.views()[0]
                    canvas.update_pipeline_live()
            event.accept()
        else:
            super().mousePressEvent(event)


# -------------------------------
# NODE BLOCK
# -------------------------------

class NodeBlock(QGraphicsRectItem):

    def __init__(self, title):

        super().__init__(0, 0, 200, 90)

        self.title = title
        self.parameters = {} # Holds current values
        self.param_def = {}  # Holds the dictionary from the script

        # Node styling
        self.setBrush(QColor("#3c3f41"))
        self.setPen(QPen(QColor("#5a5a5a"), 2))

        self.setFlag(QGraphicsRectItem.ItemIsMovable)
        self.setFlag(QGraphicsRectItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)

        # Add the title label
        self.title_text = QGraphicsTextItem(title, self)
        self.title_text.setDefaultTextColor(Qt.white)
        self.title_text.setPos(10, 5)  # top-left position

        # optional: make the label centered
        rect = self.boundingRect()
        self.title_text.setPos((rect.width() - self.title_text.boundingRect().width()) / 2, 5)

        # sockets
        self.input_socket = NodeSocket(self, "input")
        self.input_socket.setPos(0, 45)

        self.output_socket = NodeSocket(self, "output")
        self.output_socket.setPos(200, 45)

        self.load_parameter_definition()

    def load_parameter_definition(self):
        """Dynamically imports the script to get its PARAMETERS dict"""
        try:
            # Convert title back to filename (e.g., "Gaussian Blur" -> "gaussian_blur")
            module_name = self.title.lower().replace(" ", "_")
            module = importlib.import_module(f"computer_vision.{module_name}")
            
            # Look for a variable named PARAMETERS in the script
            if hasattr(module, "PARAMETERS"):
                self.param_def = module.PARAMETERS
                # Initialize with defaults
                self.parameters = {k: v["default"] for k, v in self.param_def.items()}
        except Exception as e:
            print(f"No parameter definition found for {self.title}: {e}")

    def mouseDoubleClickEvent(self, event):
        if not self.param_def: return

        parent_widget = None
        if self.scene() and self.scene().views():
            view = self.scene().views()[0]
            if hasattr(view, 'process_window') and view.process_window is not None:
                parent_widget = view.process_window
            elif callable(getattr(view, 'window', None)):
                parent_widget = view.window()
            else:
                parent_widget = getattr(view, 'window', view)

        if self.title.strip().lower() == "deep learning":
            supported, reason, _ = check_deep_learning_support()
            if not supported:
                show_unsupported_feature_popup(parent=parent_widget, feature_name="Deep Learning", reason=reason)
                return

        # Check if an existing dialog for this block is already open
        if hasattr(self, 'active_dialog') and self.active_dialog is not None:
            try:
                if self.active_dialog.isVisible():
                    self.active_dialog.raise_()
                    self.active_dialog.activateWindow()
                    return
            except RuntimeError:
                self.active_dialog = None

        dialog = ParameterDialog(self.title, self.param_def, self.parameters, parent=parent_widget)
        self.active_dialog = dialog
        
        # Connect the LIVE signal
        dialog.parametersChanged.connect(self.on_live_update)

        def on_dialog_finished(result):
            self.parameters = dialog.get_values()
            self.on_live_update(self.parameters)
            self.active_dialog = None

        dialog.finished.connect(on_dialog_finished)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def on_live_update(self, new_params):
        self.parameters = new_params
        # Find the canvas and tell it to push the update to the camera
        if self.scene() and self.scene().views():
            canvas = self.scene().views()[0]
            if hasattr(canvas, 'push_live_pipeline'):
                canvas.push_live_pipeline()

    def itemChange(self, change, value):

        if change == QGraphicsItem.ItemPositionChange:
            for socket in [self.input_socket, self.output_socket]:
                for conn in socket.connections:
                    conn.update_path()

        return super().itemChange(change, value)
    
    def contextMenuEvent(self, event):
        menu = QMenu()
        remove_action = menu.addAction("Remove Node")
        
        # Execute menu
        action = menu.exec(event.screenPos())
        
        if action == remove_action:
            self.remove_self()

    def remove_self(self):
        """Safely removes the node and all its connected lines"""
        if hasattr(self, 'active_dialog') and self.active_dialog is not None:
            try:
                self.active_dialog.close()
            except RuntimeError:
                pass
            self.active_dialog = None

        # 1. Gather all connections from both input and output sockets
        all_connections = list(self.input_socket.connections + self.output_socket.connections)
        scene_ref = self.scene()
        
        for conn in all_connections:
            # 2. Tell the other sockets to forget about this connection
            if conn.start_socket and conn in conn.start_socket.connections:
                conn.start_socket.connections.remove(conn)
            
            if conn.end_socket and conn in conn.end_socket.connections:
                conn.end_socket.connections.remove(conn)
            
            # 3. Remove connection from scene
            if scene_ref and conn.scene() == scene_ref:
                scene_ref.removeItem(conn)

        # 4. Remove the node itself from the scene
        if scene_ref and self.scene() == scene_ref:
            scene_ref.removeItem(self)

        # 5. Trigger live update
        if scene_ref and scene_ref.views():
            canvas = scene_ref.views()[0]
            canvas.update_pipeline_live()
