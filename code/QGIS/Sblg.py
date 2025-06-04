from qgis.utils import iface
from qgis.PyQt.QtGui import QColor
from qgis.core import (QgsRasterBandStats,
                       QgsColorRampShader,
                       QgsRasterShader,
                       QgsSingleBandPseudoColorRenderer,
                       QgsStyle,
                       QgsProject)
import math

# ─────────────────────────────────────────────────────────────────────────────
# (A) Defaults: if the user doesn’t supply anything, these values are used.

DEFAULT_CLASS_COUNTS = list(range(8, 16))
DEFAULT_PALETTE      = "Spectral_r"
DEFAULT_STEPS        = [1000.0, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.1]

# ─────────────────────────────────────────────────────────────────────────────
# (B) Helper functions using the parameters where needed.

def choose_classification(raw_min, raw_max, steps, class_counts):
    raw_range = raw_max - raw_min
    best = None  # will hold (padding, –step, N, new_min, new_max)
    for step in steps:
        for N in class_counts:
            total_span = N * step
            if total_span < raw_range:
                continue

            # outward rounding:
            new_min = math.floor(raw_min / step) * step
            new_max = math.ceil (raw_max / step) * step

            # force exact N * step span
            actual = new_max - new_min
            if actual > total_span:
                new_max = new_min + total_span
            elif actual < total_span:
                new_max = new_min + total_span

            padding = (new_max - new_min) - raw_range
            candidate = (padding, -step, N, new_min, new_max)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        # fallback (very large raw_range)
        step = 1.0
        N = max(class_counts)
        new_min = math.floor(raw_min)
        new_max = new_min + N * step
        return step, N, new_min, new_max

    _, neg_step, N, new_min, new_max = best
    return -neg_step, N, new_min, new_max

def decimal_digits(step):
    s = f"{step:.3f}"
    if "." not in s:
        return 0
    s = s.rstrip("0").rstrip(".")
    if "." in s:
        return len(s.split(".")[1])
    return 0

def build_items_no_endcaps(color_ramp, step, N, new_min):
    items = []
    digits = decimal_digits(step)
    for i in range(N):
        lower = new_min + i * step
        upper = lower + step
        t = float(i) / float(max(N - 1, 1))
        clr = color_ramp.color(t)
        label = f"{lower:.{digits}f} – {upper:.{digits}f}"
        items.append(QgsColorRampShader.ColorRampItem(upper, clr, label))
    return items

# ─────────────────────────────────────────────────────────────────────────────
# (C) The main reclassification function. It takes optional arguments; if the caller omits them, we use the DEFAULT_*..
def nice_steps(palette_name = None, class_counts = None, step_sizes   = None):
    """
    Reclassify all currently selected single-band rasters in the Layers panel,
    using:
      • class_counts:   a list of integer class counts (e.g. [8,9,…,15]) → pick best.
      • palette_name:   name of a color ramp in the default style (e.g. "Spectral").
      • step_sizes:     list of allowed step sizes (e.g. [2,1,0.5,…]).
    Any argument left as None will fall back to its DEFAULT_* counterpart.
    """
    # Use defaults if the user didn’t supply anything:
    if class_counts is None:
        class_counts = DEFAULT_CLASS_COUNTS
    if palette_name is None:
        palette_name = DEFAULT_PALETTE
    if step_sizes is None:
        step_sizes = DEFAULT_STEPS

    # Grab the selected layers (must be run from the Python Console):
    selected_layers = iface.layerTreeView().selectedLayers()
    if not selected_layers:
        iface.messageBar().pushWarning("Reclassification", "No raster layers selected.")
        return

    # Ensure the chosen palette actually exists; otherwise pick the first one:
    style = QgsStyle().defaultStyle()

    # If the user asked for “Spectral_r”, strip the “_r” and mark it inverted:
    if palette_name.endswith("_r"):
        base_name = palette_name[:-2]  # “Spectral”
        if style.colorRamp(base_name) is None:
            # fallback if even “Spectral” isn’t found
            base_name = style.colorRampNames()[0]
        color_ramp = style.colorRamp(base_name).clone()  # make a copy before inverting
        color_ramp.invert()
    else:
        # Otherwise just grab the named ramp normally
        if style.colorRamp(palette_name) is None:
            palette_name = style.colorRampNames()[0]
        color_ramp = style.colorRamp(palette_name)

    for layer in selected_layers:
        if layer.type() != layer.RasterLayer:
            continue

        provider = layer.dataProvider()
        stats = provider.bandStatistics(1, QgsRasterBandStats.Min | QgsRasterBandStats.Max)
        raw_min = stats.minimumValue
        raw_max = stats.maximumValue
        if raw_min is None or raw_max is None:
            continue

        # 1) Choose the best (step, N, new_min, new_max) from our lists:
        step, num_classes, new_min, new_max = choose_classification(raw_min, raw_max, step_sizes, class_counts)

        # 2) Build a discrete pseudocolor shader with exactly N intervals:
        shader = QgsColorRampShader()
        shader.setColorRampType(QgsColorRampShader.Discrete)
        items = build_items_no_endcaps(color_ramp, step, num_classes, new_min)
        shader.setColorRampItemList(items)

        raster_shader = QgsRasterShader()
        raster_shader.setRasterShaderFunction(shader)

        new_renderer = QgsSingleBandPseudoColorRenderer(provider, 1, raster_shader)
        new_renderer.setClassificationMin(new_min)
        new_renderer.setClassificationMax(new_min + num_classes * step)

        layer.setRenderer(new_renderer)
        layer.triggerRepaint()

        # 3) Expand just this layer’s node (so it doesn’t collapse):
        root       = QgsProject.instance().layerTreeRoot()
        node       = root.findLayer(layer.id())
        model      = iface.layerTreeView().layerTreeModel()
        proxy      = iface.layerTreeView().model()
        source_idx = model.node2index(node)
        proxy_idx  = proxy.mapFromSource(source_idx)
        iface.layerTreeView().setExpanded(proxy_idx, True)

        print(
            f"Layer '{layer.name()}': raw [{raw_min:.4f} … {raw_max:.4f}] → "
            f"rounded [{new_min:.4f} … {new_max:.4f}], step={step}, classes={num_classes}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# (D) Example calls:
#
#  • Default behavior (8–15 classes, “Spectral” palette):
#      reclassify_selected_layers()
#
#  • Force exactly 7 classes (ignore the 8–15 range):
#      reclassify_selected_layers(class_counts=[7])
#
#  • Use a “Blues” ramp instead of “Spectral”:
#      reclassify_selected_layers(palette_name="Blues")
#
#  • Use only 5, 10, or 15 classes (so it picks whichever of those best fits):
#      reclassify_selected_layers(class_counts=[5, 10, 15])
#
#  • Try only coarse steps 10, 20, 50 (and force 8–12 classes):
#      reclassify_selected_layers(
#          class_counts=list(range(8, 13)),
#          step_sizes=[50.0, 20.0, 10.0]
#      )
# ─────────────────────────────────────────────────────────────────────────────

# By default, run with no arguments (i.e. the defaults).
# If you want to override, call reclassify_selected_layers(...) yourself.
#nice_steps()