"""Mouse readout for the prediction's lower-origin imshow grids."""
import math


def grid_index(x, y, extent, shape):
    if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
        return None
    left, right, bottom, top = extent
    if not (left <= x <= right and bottom <= y <= top):
        return None
    rows, cols = shape
    return (min(rows - 1, int((y - bottom) / (top - bottom) * rows)),
            min(cols - 1, int((x - left) / (right - left) * cols)))


def attach_velocity_hover(fig, images, fields, theta_deg, z_m):
    """Display all four velocities for the cell under the mouse, without extra packages."""
    annotations = {}
    for image in images:
        ax = image.axes
        note = ax.annotate('', xy=(0, 0), xytext=(0.02, 0.98),
                           textcoords='axes fraction', va='top', fontsize=9,
                           bbox=dict(boxstyle='round', fc='white', alpha=0.9))
        note.set_visible(False)
        annotations[ax] = (image, note)

    def text_at(image, x, y):
        index = grid_index(x, y, image.get_extent(), fields['u'].shape)
        if index is None:
            return None
        row, col = index
        return (f'theta={theta_deg[col]:.2f} deg, z={z_m[row]:.4f} m\n'
                f'U={fields["u"][row, col]:.4f}, V={fields["v"][row, col]:.4f}\n'
                f'W={fields["w"][row, col]:.4f}, |V|={fields["magnitude"][row, col]:.4f} m/s')

    for image in images:
        image.axes.format_coord = lambda x, y, im=image: (text_at(im, x, y) or '').replace('\n', '  ')

    def on_move(event):
        changed = False
        for ax, (image, note) in annotations.items():
            text = text_at(image, event.xdata, event.ydata) if event.inaxes is ax else None
            visible = text is not None
            if note.get_visible() != visible or (visible and note.get_text() != text):
                changed = True
                note.set_visible(visible)
                if visible:
                    note.xy = (event.xdata, event.ydata)
                    note.set_text(text)
        if changed:
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('figure_leave_event', lambda event: on_move(
        type('Outside', (), dict(inaxes=None, xdata=None, ydata=None))()))
    return on_move
