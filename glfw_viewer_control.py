import glfw
import mujoco as mj
import numpy as np

class MouseState:
    def __init__(self):
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.last_pos = np.array([0.0, 0.0])

def setup_mouse_callbacks(window, model, scene, cam, width, height):
    mouse_state = MouseState()

    def mouse_button_callback(window, button, action, mods):
        if action == glfw.PRESS:
            if button == glfw.MOUSE_BUTTON_LEFT:
                mouse_state.button_left = True
            elif button == glfw.MOUSE_BUTTON_MIDDLE:
                mouse_state.button_middle = True
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                mouse_state.button_right = True
        elif action == glfw.RELEASE:
            if button == glfw.MOUSE_BUTTON_LEFT:
                mouse_state.button_left = False
            elif button == glfw.MOUSE_BUTTON_MIDDLE:
                mouse_state.button_middle = False
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                mouse_state.button_right = False

    def cursor_pos_callback(window, xpos, ypos):
        dx = xpos - mouse_state.last_pos[0]
        dy = ypos - mouse_state.last_pos[1]
        mouse_state.last_pos = np.array([xpos, ypos])

        if mouse_state.button_left:
            mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ROTATE_H, dx / width, dy / height, scene, cam)
        elif mouse_state.button_middle:
            mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, dx / width, dy / height, scene, cam)
        elif mouse_state.button_right:
            mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_PAN, dx / width, dy / height, scene, cam)

    def scroll_callback(window, xoffset, yoffset):
        mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, scene, cam)

    # 设置 GLFW 回调
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    return mouse_state