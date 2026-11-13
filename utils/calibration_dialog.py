from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
                             QLabel, QDoubleSpinBox, QComboBox, QCheckBox, 
                             QDialogButtonBox, QFrame)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

try:
    from qt_material_icons import MaterialIcon
except ImportError:
    MaterialIcon = lambda name: QIcon()


class CalibrationDialog(QDialog):
    """
    Dialog to calibrate camera pixel resolution to physical real-world units (mm, µm, inch, etc.)
    using a measured reference line.
    """
    def __init__(self, measured_px, current_calib=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scale Calibration (Pixel -> Physical)")
        self.setMinimumWidth(380)
        self.measured_px = max(0.1, float(measured_px))
        self.current_calib = current_calib or {}

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header Info Frame
        header_frame = QFrame(self)
        header_frame.setStyleSheet("background-color: #2b2b2b; border-radius: 6px; padding: 8px;")
        hdr_layout = QVBoxLayout(header_frame)
        hdr_layout.setContentsMargins(4, 4, 4, 4)
        
        hdr_title = QLabel("Physical Dimension Calibration", header_frame)
        hdr_title.setStyleSheet("font-weight: bold; font-size: 13px; color: #ffffff;")
        hdr_desc = QLabel(
            "Enter the known real-world dimension of the measured feature.\n"
            "All measurements on the canvas will automatically use calibrated units.", 
            header_frame
        )
        hdr_desc.setStyleSheet("color: #bbbbbb; font-size: 11px;")
        hdr_layout.addWidget(hdr_title)
        hdr_layout.addWidget(hdr_desc)
        layout.addWidget(header_frame)

        # Form
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignRight)
        form_layout.setSpacing(10)

        # Measured px
        self.lbl_px = QLabel(f"<b>{self.measured_px:.2f} px</b>")
        self.lbl_px.setStyleSheet("font-size: 13px; color: #00e5ff;")
        form_layout.addRow("Reference Line Length:", self.lbl_px)

        # Real distance input
        self.spin_real = QDoubleSpinBox(self)
        self.spin_real.setRange(0.0001, 1000000.0)
        self.spin_real.setDecimals(4)
        self.spin_real.setSingleStep(1.0)
        default_real = self.current_calib.get('ref_real', 10.0) if self.current_calib.get('is_calibrated') else 10.0
        self.spin_real.setValue(float(default_real))
        self.spin_real.setStyleSheet("font-size: 12px; padding: 4px;")
        form_layout.addRow("Known Real Dimension:", self.spin_real)

        # Units combo
        self.combo_unit = QComboBox(self)
        units = [
            ("mm (Millimeters)", "mm"),
            ("µm (Micrometers)", "µm"),
            ("cm (Centimeters)", "cm"),
            ("m (Meters)", "m"),
            ("in (Inches)", "in"),
            ("mil (Thousandths of inch)", "mil")
        ]
        for label, ucode in units:
            self.combo_unit.addItem(label, ucode)

        cur_unit = self.current_calib.get('unit', 'mm')
        for i in range(self.combo_unit.count()):
            if self.combo_unit.itemData(i) == cur_unit:
                self.combo_unit.setCurrentIndex(i)
                break

        form_layout.addRow("Measurement Unit:", self.combo_unit)
        layout.addLayout(form_layout)

        # Result badge
        self.lbl_result = QLabel(self)
        self.lbl_result.setStyleSheet(
            "background-color: #1e3a5f; color: #64b5f6; border: 1px solid #1976d2; "
            "border-radius: 4px; padding: 8px; font-weight: bold; font-size: 12px; text-align: center;"
        )
        self.lbl_result.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_result)

        # Apply to all modules
        self.chk_apply_all = QCheckBox("Apply scale calibration to all active modules", self)
        self.chk_apply_all.setChecked(True)
        layout.addWidget(self.chk_apply_all)

        # Dialog Buttons
        self.btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.btn_box.button(QDialogButtonBox.Ok).setText("Apply Calibration")
        self.btn_box.accepted.connect(self.accept)
        self.btn_box.rejected.connect(self.reject)
        layout.addWidget(self.btn_box)

        self.spin_real.valueChanged.connect(self.update_result_preview)
        self.combo_unit.currentIndexChanged.connect(self.update_result_preview)
        self.update_result_preview()

    def update_result_preview(self):
        real_val = self.spin_real.value()
        unit = self.combo_unit.currentData() or "mm"
        if self.measured_px > 0 and real_val > 0:
            scale = real_val / self.measured_px
            px_per_unit = self.measured_px / real_val
            self.lbl_result.setText(
                f"Scale: 1 px = {scale:.5g} {unit}  ({px_per_unit:.2f} px / {unit})"
            )
        else:
            self.lbl_result.setText("Invalid calibration input")

    def get_calibration_data(self):
        real_val = float(self.spin_real.value())
        unit = str(self.combo_unit.currentData() or "mm")
        scale = real_val / self.measured_px
        return {
            'scale': scale,
            'unit': unit,
            'ref_pixels': self.measured_px,
            'ref_real': real_val,
            'is_calibrated': True,
            'apply_all': self.chk_apply_all.isChecked()
        }
