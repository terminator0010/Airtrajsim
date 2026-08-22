import cv2
import numpy as np
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import AirTrajSimPy_main as At

class AirTrajAR:
    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        v0: float = 110.0,
        elevation_deg: float = 0,
        mass_g: float = 0.20,
        hop_up: float = 0.4,
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
        
        self.smoothed_distance = None
        self.alpha = 0.15        

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

        # --- INICIALIZAÇÃO DO MEDIAPIPE (TASKS API) ---
        base_options = python.BaseOptions(model_asset_path='face_detector.task')
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
        self.face_detector = vision.FaceDetector.create_from_options(options)
        
        self.REAL_FACE_HEIGHT_M = 0.23 

    def _compute_trajectory_3d(self) -> np.ndarray:
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

    def detect_and_draw_faces(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # Calcular a posição do retículo com base na elevação
        fy = self.K[1, 1]
        offset_y = fy * math.tan(math.radians(self.elevation_deg))
        reticle_x = w // 2
        reticle_y = int(h / 2 - offset_y)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.face_detector.detect(mp_image)

        target_detection = None
        min_dist_to_center = float('inf')

        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                xmin = bbox.origin_x
                ymin = bbox.origin_y
                box_w = bbox.width
                box_h = bbox.height
                
                if xmin <= reticle_x <= (xmin + box_w) and ymin <= reticle_y <= (ymin + box_h):
                    face_cx = xmin + box_w / 2
                    face_cy = ymin + box_h / 2
                    dist = math.hypot(face_cx - reticle_x, face_cy - reticle_y)
                    
                    if dist < min_dist_to_center:
                        min_dist_to_center = dist
                        target_detection = detection

            if target_detection:
                bbox = target_detection.bounding_box
                xmin = bbox.origin_x
                ymin = bbox.origin_y
                box_w = bbox.width
                box_h = bbox.height

                if box_h > 0:
                    distance_m = (self.REAL_FACE_HEIGHT_M * fy) / box_h

                    color = (0, 255, 0)
                    cv2.rectangle(frame, (xmin, ymin), (xmin + box_w, ymin + box_h), color, 2)
                    
                    text = f"Alvo: {distance_m:.1f}m"
                    cv2.putText(frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                    
                    cx_face, cy_face = int(xmin + box_w // 2), int(ymin + box_h // 2)
                    cv2.circle(frame, (cx_face, cy_face), 3, color, -1)

        return frame

# --- NOVO SISTEMA DE RENDERIZAÇÃO DA MIRA ---
    # --- NOVO SISTEMA DE RENDERIZAÇÃO DA MIRA ---
    def render_overlay(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # Ajustando o centro (cy) com base na elevação em graus
        fy = self.K[1, 1]
        offset_y = fy * math.tan(math.radians(self.elevation_deg))
        
        cx = w // 2
        cy = int(h / 2 - offset_y) 
        
        cross_color_base = (0, 255, 0) # Verde tático para a linha base
        zero_color = (0, 0, 255)       # Vermelho para o 0m (Centro)

        # Desenhar o retículo principal (Estilo PSO-1)
        # 1. Linhas horizontais principais
        cv2.line(frame, (cx - 150, cy), (cx - 20, cy), cross_color_base, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + 20, cy), (cx + 150, cy), cross_color_base, 1, cv2.LINE_AA)
        
        # 2. Marcações (hash marks) na linha horizontal
        for i in range(1, 6):
            cv2.line(frame, (cx - 20 - i*25, cy - 5), (cx - 20 - i*25, cy + 5), cross_color_base, 1, cv2.LINE_AA)
            cv2.line(frame, (cx + 20 + i*25, cy - 5), (cx + 20 + i*25, cy + 5), cross_color_base, 1, cv2.LINE_AA)

        # 3. Chevron central principal (Centro da mira - 0m) -> Destaque em Vermelho
        cv2.polylines(frame, [np.array([(cx - 8, cy + 8), (cx, cy), (cx + 8, cy + 8)])], isClosed=False, color=zero_color, thickness=2, lineType=cv2.LINE_AA)
        cv2.line(frame, (cx, cy + 12), (cx, cy + 30), zero_color, 2, cv2.LINE_AA)

        # 4. Desenhar as distâncias da trajetória balística com cores e tamanhos dinâmicos
        for pt_2d, label in self.distance_markers:
            px, py = int(pt_2d[0]), int(pt_2d[1])
            
            # Extrair apenas o número da string (ex: "10m" -> 10)
            try:
                dist_val = int(label.replace('m', ''))
            except:
                dist_val = 0

            # Definir a cor (BGR) e o tamanho da fonte baseados na distância
            
            if dist_val <= 1:
                marker_color = (0, 0, 255)   # Azul
                font_scale = 0.55          # Fonte maior            
            elif dist_val <= 10:
                marker_color = (0, 255, 0)   # Azul
                font_scale = 0.25          # Fonte maior
            elif dist_val == 20:
                marker_color = (0, 0, 255) # Amarelo
                font_scale = 0.55          # Fonte maior
            elif dist_val == 30:
                marker_color = (0, 255, 0)   # Verde
                font_scale = 0.25          # Fonte maior
            elif dist_val == 40:
                marker_color = (255, 100, 0) # Azul
                font_scale = 0.25          # Fonte menor
            else:
                marker_color = (255, 100, 0) # Azul (50m para cima)
                font_scale = 0.25          # Fonte menor

            if 0 <= px < w and 0 <= py < h:
                chev_size = 6 # Tamanho do chevron
                pt1 = (px - chev_size, py + chev_size)
                pt2 = (px, py)
                pt3 = (px + chev_size, py + chev_size)
                
                # Desenha o chevron com espessura 2 para destacar mais na tela
                cv2.polylines(frame, [np.array([pt1, pt2, pt3])], isClosed=False, color=marker_color, thickness=1, lineType=cv2.LINE_AA)
                
                # Calcula o tamanho do texto baseado na font_scale dinâmica
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                
                text_x = px - tw // 2  # Centraliza no eixo X
                text_y = py + chev_size + th + 4  # Desce no eixo Y (abaixo do chevron)
                
                # Renderiza o texto
                cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, marker_color, 1, cv2.LINE_AA)

        # Dados balísticos (HUD Superior Esquerdo)
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
            
            frame = self.detect_and_draw_faces(frame)
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
        self.face_detector.close()

if __name__ == "__main__":
    ar = AirTrajAR(
        frame_width=800,
        frame_height=600,
        v0=110.0,              
        elevation_deg=0.0,     
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