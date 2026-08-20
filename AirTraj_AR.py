import cv2
import numpy as np
import math
import AirTrajSimPy_main as At

class AirTrajAR:
    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        v0: float = 150.0,
        elevation_deg: float = 0,
        mass_g: float = 0.25,
        hop_up: float = 0.50,
        gravity: float = 9.81,
        wind_lateral_ms: float = 0.0,
        temp_c: float = 25.0,
        h0: float = 1.5,
        dt: float = 0.001,
        cam_height_offset: float = -0.05,
        cam_depth_offset: float = -0.10,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.v0 = v0
        self.elevation_deg = elevation_deg
        
        # Puxando variaveis globais diretamente do main original, se desejado
        self.mass_g = At.m_g if hasattr(At, 'm_g') else mass_g
        self.mass_kg = self.mass_g / 1000.0
        self.hop_up = At.hop_percent if hasattr(At, 'hop_percent') else hop_up
        
        self.gravity = gravity
        self.wind_lateral_ms = wind_lateral_ms
        self.temp_c = temp_c
        self.h0 = h0
        self.dt = dt
        self.cam_height_offset = cam_height_offset
        self.cam_depth_offset = cam_depth_offset

        # Matriz Intrínseca Genérica
        fov_horizontal_deg = 20.0
        fx = frame_width / (2.0 * math.tan(math.radians(fov_horizontal_deg / 2.0)))
        fy = fx
        cx = frame_width / 2.0
        cy = frame_height / 2.0

        self.K = np.array([
            [fx,  0.0, cx],
            [0.0, fy,  cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        self.rvec = np.zeros((3, 1), dtype=np.float64)
        self.tvec = np.array([
            [0.0],
            [0.0],
            [self.cam_depth_offset]
        ], dtype=np.float64)

        self.trajectory_3d = self._compute_trajectory_3d()
        self.trajectory_2d = None
        self._project_trajectory()
        self._compute_distance_markers()

    def _compute_trajectory_3d(self) -> np.ndarray:
        # Chamada direta para o motor físico modularizado no arquivo principal
        traj_main = At.calcular_fisica_3d(
            v0=self.v0,
            elevation_deg=self.elevation_deg,
            mass_kg=self.mass_kg,
            hop_percent=self.hop_up,
            temp_c=self.temp_c,
            wind_lateral_ms=self.wind_lateral_ms,
            h0=self.h0,
            dt=self.dt
        )
        
        # Conversão das coordenadas do Main para o OpenCV
        # OpenCV: X=direita, Y=baixo, Z=frente
        opencv_x = traj_main[:, 2]              
        opencv_y = -(traj_main[:, 1] - self.h0) 
        opencv_z = traj_main[:, 0]              

        return np.stack([opencv_x, opencv_y, opencv_z], axis=1)

    def _project_trajectory(self):
        if self.trajectory_3d is None or len(self.trajectory_3d) < 2:
            self.trajectory_2d = None
            return

        points_3d = self.trajectory_3d.reshape(-1, 1, 3)
        points_2d, _ = cv2.projectPoints(
            objectPoints=points_3d,
            rvec=self.rvec,
            tvec=self.tvec,
            cameraMatrix=self.K,
            distCoeffs=self.dist_coeffs
        )
        self.trajectory_2d = points_2d.astype(np.int32)

    def _compute_distance_markers(self):
        self.distance_markers = []
        if self.trajectory_3d is None or self.trajectory_2d is None:
            return

        z_values = self.trajectory_3d[:, 2]
        max_z = z_values[-1] if len(z_values) > 0 else 0

        for dist_m in range(10, int(max_z) + 1, 10):
            idx = np.argmin(np.abs(z_values - dist_m))
            if idx < len(self.trajectory_2d):
                pt_2d = self.trajectory_2d[idx, 0]
                self.distance_markers.append((pt_2d, f"{dist_m}m"))

    def update_parameters(self, **kwargs):
        recalc = False
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                recalc = True
                if key == 'mass_g':
                    self.mass_kg = value / 1000.0
                elif key == 'cam_height_offset':
                    self.tvec[1, 0] = value

        if recalc:
            self.trajectory_3d = self._compute_trajectory_3d()
            self._project_trajectory()
            self._compute_distance_markers()

    def render_overlay(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        if self.trajectory_2d is not None and len(self.trajectory_2d) > 1:
            n_pts = len(self.trajectory_2d)
            n_segments = min(n_pts - 1, 50)
            step = max(1, (n_pts - 1) // n_segments)

            for i in range(0, n_pts - 1, step):
                j = min(i + step, n_pts - 1)
                progress = i / max(n_pts - 1, 1)

                if progress < 0.5:
                    t_color = progress * 2.0
                    b, g, r = 0, int(255 * (1.0 - t_color * 0.3)), int(255 * t_color)
                else:
                    t_color = (progress - 0.5) * 2.0
                    b, g, r = 0, int(255 * (0.7 - t_color * 0.7)), int(255 * (1.0 - t_color * 0.3) + 80 * t_color)

                color = (b, g, min(r, 255))
                seg_pts = self.trajectory_2d[i:j+1]
                if len(seg_pts) >= 2:
                    cv2.polylines(frame, [seg_pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

            last_pt = tuple(self.trajectory_2d[-1, 0])
            if 0 <= last_pt[0] < w and 0 <= last_pt[1] < h:
                cv2.drawMarker(frame, last_pt, color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2, line_type=cv2.LINE_AA)

            first_pt = tuple(self.trajectory_2d[0, 0])
            if 0 <= first_pt[0] < w and 0 <= first_pt[1] < h:
                cv2.circle(frame, first_pt, 5, (0, 255, 0), -1, cv2.LINE_AA)

        for pt_2d, label in self.distance_markers:
            px, py = int(pt_2d[0]), int(pt_2d[1])
            if 0 <= px < w and 0 <= py < h:
                cv2.line(frame, (px, py - 6), (px, py + 6), (255, 255, 255), 1, cv2.LINE_AA)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (px - 2, py - th - 12), (px + tw + 2, py - 8), (0, 0, 0), -1)
                cv2.putText(frame, label, (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        cx, cy = w // 2, h // 2
        cross_size = 15
        cross_gap = 4
        cross_color = (0, 255, 0)
        cv2.line(frame, (cx - cross_size, cy), (cx - cross_gap, cy), cross_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + cross_gap, cy), (cx + cross_size, cy), cross_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - cross_size), (cx, cy - cross_gap), cross_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + cross_gap), (cx, cy + cross_size), cross_color, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 1, cross_color, -1, cv2.LINE_AA)

        hud_color = (200, 200, 200)
        hud_bg = (20, 20, 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        lines_top = [
            f"V0: {self.v0:.0f} m/s  |  Elev: {self.elevation_deg:.1f} deg",
            f"Massa: {self.mass_g:.2f}g  |  Hop: {self.hop_up:.0%}",
            f"Vento Lat: {self.wind_lateral_ms:.1f} m/s",
        ]

        if self.trajectory_3d is not None and len(self.trajectory_3d) > 0:
            alcance = self.trajectory_3d[-1, 2]
            queda = self.trajectory_3d[-1, 1]
            lines_top.append(f"Alcance: {alcance:.1f}m  |  Queda: {queda:.2f}m")

        hud_h = 18 * len(lines_top) + 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (310, hud_h + 5), hud_bg, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        for i, line in enumerate(lines_top):
            y_text = 22 + i * 18
            cv2.putText(frame, line, (10, y_text), font, 0.42, hud_color, 1, cv2.LINE_AA)

        instructions = ["[Q] Sair  [W/S] V0+/-  [A/D] Elev+/-  [E/R] Hop+/-  [Z/X] Vento+/-"]
        for i, line in enumerate(instructions):
            y_text = h - 12 - i * 18
            cv2.putText(frame, line, (11, y_text + 1), font, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, line, (10, y_text), font, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        return frame

    def run(self, camera_index: int = 0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[ERRO] Nao foi possivel abrir a camera.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_w != self.frame_width or actual_h != self.frame_height:
            self.frame_width = actual_w
            self.frame_height = actual_h
            fx = actual_w / (2.0 * math.tan(math.radians(60.0 / 2.0)))
            self.K = np.array([[fx, 0.0, actual_w/2.0], [0.0, fx, actual_h/2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            self._project_trajectory()
            self._compute_distance_markers()

        window_name = "AirTrajSim AR"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        while True:
            ret, frame = cap.read()
            if not ret: continue
            
            frame = self.render_overlay(frame)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]: break
            elif key == ord('w'): self.update_parameters(v0=self.v0 + 5.0)
            elif key == ord('s'): self.update_parameters(v0=max(10.0, self.v0 - 5.0))
            elif key == ord('d'): self.update_parameters(elevation_deg=self.elevation_deg + 0.5)
            elif key == ord('a'): self.update_parameters(elevation_deg=self.elevation_deg - 0.5)
            elif key == ord('e'): self.update_parameters(hop_up=min(1.0, self.hop_up + 0.05))
            elif key == ord('r'): self.update_parameters(hop_up=max(0.0, self.hop_up - 0.05))
            elif key == ord('z'): self.update_parameters(wind_lateral_ms=self.wind_lateral_ms + 0.5)
            elif key == ord('x'): self.update_parameters(wind_lateral_ms=self.wind_lateral_ms - 0.5)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ar = AirTrajAR(
        frame_width=640,
        frame_height=480,
        v0=110.0,              
        elevation_deg=1.5,     
        mass_g=0.20,           
        hop_up=0.35,           
        gravity=9.81,
        wind_lateral_ms=0.0,   
        temp_c=25.0,           
        h0=1.5,                
        dt=0.001,              
        cam_height_offset=-0.05,
        cam_depth_offset=0.0,
    )
    ar.run(camera_index=0)