# =============================================================================
# AirTraj_AR.py — Módulo de Realidade Aumentada para o AirTrajSim
# =============================================================================
# Autor: Assistente de Engenharia (Antigravity)
# Dependências: cv2 (OpenCV), numpy, math  — Nenhum modelo de IA envolvido.
#
# Descrição:
#   Abre a câmera USB principal (índice 0), calcula uma trajetória balística
#   parabólica 3D (projétil leve tipo esfera de polímero 6mm) e renderiza
#   a curva projetada sobre o feed de vídeo em tempo real, funcionando como
#   um HUD de realidade aumentada.
#
# Fluxo:
#   1. Gera uma nuvem de pontos 3D [X, Y, Z] da trajetória (pré-cálculo).
#   2. Define uma câmera pinhole genérica (matriz intrínseca K 3×3).
#   3. Define pose estática da câmera (R = identidade, t = translação fixa).
#   4. Projeta os pontos 3D → 2D via cv2.projectPoints() a cada frame.
#   5. Desenha a curva com cv2.polylines() sobre o frame capturado.
# =============================================================================

import cv2
import numpy as np
import math


class AirTrajAR:
    """
    Classe principal do módulo de Realidade Aumentada.

    Responsável por:
      - Gerar a trajetória 3D balística (parabólica com arrasto simplificado).
      - Configurar o modelo de câmera pinhole (matriz intrínseca K).
      - Projetar pontos 3D → 2D via cv2.projectPoints().
      - Renderizar o overlay sobre o feed da webcam em tempo real.

    Pontos de Atenção para Ajuste:
      - self.K  (fx, fy, cx, cy)  → Ajuste se a linha "flutuar" estranhamente.
      - self.tvec  (eixo Y)       → Ajuste para simular offset de luneta/scope.
      - self.v0, self.elevation   → Velocidade e ângulo de saída do projétil.
    """

    # =========================================================================
    # CONSTRUTOR — Parâmetros configuráveis do sistema
    # =========================================================================
    def __init__(
        self,
        # --- Resolução da câmera ---
        frame_width: int = 640,
        frame_height: int = 480,
        # --- Parâmetros balísticos ---
        v0: float = 110.0,          # Velocidade inicial (m/s) — ex: M110 airsoft
        elevation_deg: float = 1.5, # Ângulo de elevação (graus) acima da horizontal
        mass_g: float = 0.20,       # Massa do projétil (gramas)
        hop_up: float = 0.35,       # Fator de hop-up (0.0–1.0) → sustentação Magnus
        # --- Ambiente ---
        gravity: float = 9.81,      # Aceleração gravitacional (m/s²)
        wind_lateral_ms: float = 0.0,  # Vento lateral (m/s) positivo = direita
        temp_c: float = 25.0,       # Temperatura ambiente (°C) — para cálculo de ρ do ar
        h0: float = 1.5,            # Altura inicial do cano (metros)
        # --- Simulação ---
        t_max: float = 3.0,         # Tempo máximo de voo (segundos)
        dt: float = 0.001,          # Passo de integração (segundos) — mesmo que AirTrajSimPy_main
        # --- Câmera (pose no mundo 3D) ---
        cam_height_offset: float = -0.05,  # Offset Y da câmera acima do cano (metros)
                                            # Negativo = câmera acima (eixo Y aponta pra baixo na conv. OpenCV)
        cam_depth_offset: float = 0.0,     # Offset Z para frente/trás
    ):
        # Armazenar parâmetros
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.v0 = v0
        self.elevation_deg = elevation_deg
        self.mass_g = mass_g
        self.mass_kg = mass_g / 1000.0
        self.hop_up = hop_up
        self.gravity = gravity
        self.wind_lateral_ms = wind_lateral_ms
        self.temp_c = temp_c
        self.h0 = h0
        self.t_max = t_max
        self.dt = dt
        self.cam_height_offset = cam_height_offset
        self.cam_depth_offset = cam_depth_offset

        # -----------------------------------------------------------------
        # MATRIZ INTRÍNSECA K (Câmera Pinhole Genérica)
        # -----------------------------------------------------------------
        # Para 640×480, valores focais típicos de webcam (FOV ~60°):
        #   fx = fy ≈ W / (2 * tan(FOV/2))
        #   Para FOV=60°: fx = 640 / (2 * tan(30°)) ≈ 554
        #
        # ATENÇÃO: Se a trajetória parecer "flutuando" ou com escala errada,
        #          calibre esta matriz com cv2.calibrateCamera() ou ajuste
        #          manualmente os valores de fx/fy.
        # -----------------------------------------------------------------
        fov_horizontal_deg = 60.0
        fx = frame_width / (2.0 * math.tan(math.radians(fov_horizontal_deg / 2.0)))
        fy = fx  # Pixels quadrados (aspecto 1:1)
        cx = frame_width / 2.0    # Centro óptico X
        cy = frame_height / 2.0   # Centro óptico Y

        self.K = np.array([
            [fx,  0.0, cx],
            [0.0, fy,  cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # -----------------------------------------------------------------
        # COEFICIENTES DE DISTORÇÃO (Nulos — sem distorção de lente)
        # -----------------------------------------------------------------
        # Se quiser modelar distorção radial/tangencial, preencha aqui:
        #   [k1, k2, p1, p2, k3]
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        # -----------------------------------------------------------------
        # POSE DA CÂMERA NO MUNDO 3D (R, t estáticos)
        # -----------------------------------------------------------------
        # Convenção:
        #   - Câmera apontando para +Z do mundo (para frente).
        #   - R = identidade (sem rotação) → eixos da câmera alinhados ao mundo.
        #   - t = translação da ORIGEM DO MUNDO no referencial da câmera.
        #
        # Para simular que a câmera está levemente ACIMA do cano (como uma
        # luneta montada), ajuste cam_height_offset (eixo Y).
        # No OpenCV, eixo Y aponta PARA BAIXO na imagem, então:
        #   - cam_height_offset negativo → câmera deslocada para cima.
        # -----------------------------------------------------------------
        self.rvec = np.zeros((3, 1), dtype=np.float64)  # Rodrigues: sem rotação
        self.tvec = np.array([
            [0.0],                    # X: sem offset lateral
            [self.cam_height_offset], # Y: offset vertical (luneta)
            [self.cam_depth_offset]   # Z: sem offset de profundidade
        ], dtype=np.float64)

        # -----------------------------------------------------------------
        # PRÉ-CALCULAR a trajetória 3D
        # -----------------------------------------------------------------
        self.trajectory_3d = self._compute_trajectory_3d()
        self.trajectory_2d = None  # Será calculado no primeiro frame

        # Pré-projetar os pontos (a pose é estática, então basta uma vez)
        self._project_trajectory()

        # Pré-calcular marcadores de distância (a cada 10m)
        self._compute_distance_markers()

    # =========================================================================
    # CÁLCULO DA TRAJETÓRIA 3D
    # =========================================================================
    # NOTA: Este método usa **exatamente** o mesmo motor físico de
    #       AirTrajSimPy_main.py (RK4 + derivatives). Nenhum modelo
    #       simplificado é utilizado.
    # =========================================================================
    def _compute_trajectory_3d(self) -> np.ndarray:
        """
        Gera um array Nx3 de pontos [X_cv, Y_cv, Z_cv] (coordenadas OpenCV)
        representando a trajetória balística 3D.

        O motor físico é IDÊNTICO ao de AirTrajSimPy_main.py:
          - Integração RK4 (Runge-Kutta 4ª ordem)
          - Estado = [x, y, z, vx, vy, vz, omega]
          - Arrasto quadrático (Cd_base = 0.50)
          - Efeito Magnus via spin_param e Cl limitado a 0.25
          - Decaimento de spin como derivada (dω/dt = -0.30·ω)
          - Densidade do ar calculada pela temperatura

        Sistema de coordenadas do MAIN (usado internamente):
          - X: distância para frente (alcance)
          - Y: altura (positivo = para cima)
          - Z: desvio lateral (positivo = direita)

        Conversão para coordenadas OpenCV (saída):
          - X_cv = Z_main       (lateral → eixo X da imagem)
          - Y_cv = h0 - Y_main  (altura invertida → eixo Y da imagem, positivo = baixo)
          - Z_cv = X_main       (alcance → profundidade/eixo Z da câmera)

        Returns:
            np.ndarray: Array de shape (N, 3) com pontos em coordenadas OpenCV.
        """
        # =================================================================
        # Parâmetros físicos — IDÊNTICOS a AirTrajSimPy_main.py L206–L223
        # =================================================================
        m_kg = self.mass_kg
        temp_c = self.temp_c
        h0 = self.h0

        # Densidade do ar (equação dos gases ideais, mesmo cálculo do main)
        rho = 101325.0 / (287.05 * (temp_c + 273.15))

        # Geometria da BB
        r_bb = (5.95 / 1000.0) / 2.0
        A_sec = np.pi * r_bb ** 2

        # Vento — mapeado para o sistema do main:
        #   Main X = frente (sem componente de vento frontal por ora)
        #   Main Z = lateral (self.wind_lateral_ms → Wz)
        Wx = 0.0
        Wy = 0.0
        Wz = self.wind_lateral_ms

        # Rotação (spin) — mesmo cálculo do main L217–L219
        rpm_max = 90000.0
        omega_0 = (rpm_max * self.hop_up) * (2.0 * np.pi / 60.0)

        # Coeficientes — mesmo do main L221–L223
        Cd_base = 0.50
        fator_eficiencia_magnus = 0.18

        # =================================================================
        # Velocidade inicial com elevação
        # =================================================================
        elev_rad = math.radians(self.elevation_deg)
        vx_init = self.v0 * math.cos(elev_rad)   # Componente X (para frente)
        vy_init = self.v0 * math.sin(elev_rad)    # Componente Y (para cima)
        # Sem componente lateral inicial (vz_init = 0)

        # =================================================================
        # Estado inicial [x, y, z, vx, vy, vz, omega]
        # Mesmo formato do main L225–L226
        # =================================================================
        state = np.array([0.0, h0, 0.0, vx_init, vy_init, 0.0, omega_0])

        # =================================================================
        # Função derivatives() — CÓPIA EXATA do main L228–L251
        # =================================================================
        def derivatives(t, s):
            x, y, z, vx, vy, vz, w = s

            v_rel_x = vx - Wx
            v_rel_y = vy - Wy
            v_rel_z = vz - Wz
            v_rel = np.sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2)
            if v_rel < 0.1:
                v_rel = 0.1

            spin_param = (r_bb * w) / v_rel
            Cl = min(0.25, fator_eficiencia_magnus * spin_param)

            Fd = 0.5 * rho * A_sec * Cd_base * v_rel**2
            Fl = 0.5 * rho * A_sec * Cl * v_rel**2

            ax_drag = -(Fd / m_kg) * (v_rel_x / v_rel)
            ay_drag = -(Fd / m_kg) * (v_rel_y / v_rel)
            az_drag = -(Fd / m_kg) * (v_rel_z / v_rel)

            ay_lift = (Fl / m_kg)
            ay_grav = -9.81
            decay = -0.30 * w

            return np.array([vx, vy, vz,
                             ax_drag,
                             ay_grav + ay_drag + ay_lift,
                             az_drag,
                             decay])

        # =================================================================
        # Loop RK4 — CÓPIA EXATA do main L253–L267
        # =================================================================
        dt = self.dt  # 0.001 (mesmo padrão do main)
        trajectory_states = [state.copy()]
        t = 0.0

        while True:
            k1 = derivatives(t, state)
            k2 = derivatives(t + 0.5 * dt, state + 0.5 * dt * k1)
            k3 = derivatives(t + 0.5 * dt, state + 0.5 * dt * k2)
            k4 = derivatives(t + dt, state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt

            # Condições de parada — mesmas do main L264
            if state[1] < 0 or state[0] > 100:
                trajectory_states.append(state.copy())
                break
            trajectory_states.append(state.copy())

        # =================================================================
        # Converter coordenadas: Main → OpenCV
        # =================================================================
        traj = np.array(trajectory_states)
        # traj colunas: [x_fwd, y_up, z_lat, vx, vy, vz, omega]

        # OpenCV: X=direita, Y=baixo, Z=frente
        opencv_x = traj[:, 2]              # Z_main (lateral)   → X_cv (direita)
        opencv_y = -(traj[:, 1] - h0)      # Y_main (cima)      → Y_cv (baixo, relativo a h0)
        opencv_z = traj[:, 0]              # X_main (frente)    → Z_cv (profundidade)

        return np.stack([opencv_x, opencv_y, opencv_z], axis=1)

    # =========================================================================
    # PROJEÇÃO 3D → 2D
    # =========================================================================
    def _project_trajectory(self):
        """
        Projeta os pontos 3D da trajetória em coordenadas 2D de pixel
        usando cv2.projectPoints() com a matriz intrínseca K e a pose (R, t).

        cv2.projectPoints() implementa:
            p_2d = K · [R | t] · P_3d

        Como R = I e t ≈ 0, simplifica para:
            p_2d = K · P_3d   (com divisão perspectiva por Z)

        O resultado é armazenado em self.trajectory_2d como array Nx1x2 int32
        (formato esperado por cv2.polylines).
        """
        if self.trajectory_3d is None or len(self.trajectory_3d) < 2:
            self.trajectory_2d = None
            return

        # cv2.projectPoints espera shape (N, 1, 3) ou (N, 3)
        points_3d = self.trajectory_3d.reshape(-1, 1, 3)

        # Projetar: 3D → 2D
        points_2d, _ = cv2.projectPoints(
            objectPoints=points_3d,
            rvec=self.rvec,
            tvec=self.tvec,
            cameraMatrix=self.K,
            distCoeffs=self.dist_coeffs
        )

        # points_2d tem shape (N, 1, 2) em float64
        # Converter para int32 para cv2.polylines
        self.trajectory_2d = points_2d.astype(np.int32)

    # =========================================================================
    # MARCADORES DE DISTÂNCIA
    # =========================================================================
    def _compute_distance_markers(self):
        """
        Pré-calcula posições 2D e labels para marcadores de distância
        ao longo da trajetória (a cada 10 metros de profundidade Z).
        """
        self.distance_markers = []

        if self.trajectory_3d is None or self.trajectory_2d is None:
            return

        # Encontrar pontos a cada 10m de profundidade (Z)
        z_values = self.trajectory_3d[:, 2]  # Eixo de profundidade
        max_z = z_values[-1] if len(z_values) > 0 else 0

        for dist_m in range(10, int(max_z) + 1, 10):
            # Encontrar o índice mais próximo de Z = dist_m
            idx = np.argmin(np.abs(z_values - dist_m))
            if idx < len(self.trajectory_2d):
                pt_2d = self.trajectory_2d[idx, 0]  # (x, y) em pixels
                self.distance_markers.append((pt_2d, f"{dist_m}m"))

    # =========================================================================
    # RECALCULAR TRAJETÓRIA (para ajuste em tempo real)
    # =========================================================================
    def update_parameters(self, **kwargs):
        """
        Atualiza parâmetros balísticos e recalcula a trajetória.

        Parâmetros aceitos (qualquer combinação):
            v0, elevation_deg, mass_g, hop_up, gravity,
            wind_lateral_ms, cam_height_offset

        Exemplo:
            ar.update_parameters(v0=120.0, hop_up=0.5)
        """
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

    # =========================================================================
    # RENDERIZAÇÃO DO OVERLAY
    # =========================================================================
    def render_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Desenha a trajetória projetada e o HUD informativo sobre o frame.

        Args:
            frame: Frame BGR capturado da câmera (np.ndarray HxWx3).

        Returns:
            Frame com overlay renderizado (mesmo array, modificado in-place).
        """
        h, w = frame.shape[:2]

        # =================================================================
        # 1. DESENHAR A CURVA DA TRAJETÓRIA
        # =================================================================
        if self.trajectory_2d is not None and len(self.trajectory_2d) > 1:
            # --- Linha principal (trajetória) ---
            # Gradiente de cor ao longo da trajetória: verde → amarelo → vermelho
            n_pts = len(self.trajectory_2d)

            # Segmentar a trajetória em trechos para colorir gradualmente
            n_segments = min(n_pts - 1, 50)  # Limitar segmentos para performance
            step = max(1, (n_pts - 1) // n_segments)

            for i in range(0, n_pts - 1, step):
                j = min(i + step, n_pts - 1)

                # Progresso normalizado (0.0 → 1.0)
                progress = i / max(n_pts - 1, 1)

                # Gradiente: Verde (0,255,0) → Amarelo (0,255,255) → Vermelho (0,0,255)
                if progress < 0.5:
                    t_color = progress * 2.0
                    b, g, r = 0, int(255 * (1.0 - t_color * 0.3)), int(255 * t_color)
                else:
                    t_color = (progress - 0.5) * 2.0
                    b, g, r = 0, int(255 * (0.7 - t_color * 0.7)), int(255 * (1.0 - t_color * 0.3) + 80 * t_color)

                color = (b, g, min(r, 255))

                # Sub-segmento para polylines
                seg_pts = self.trajectory_2d[i:j+1]
                if len(seg_pts) >= 2:
                    cv2.polylines(
                        frame,
                        [seg_pts],
                        isClosed=False,
                        color=color,
                        thickness=2,
                        lineType=cv2.LINE_AA  # Anti-aliasing para suavidade
                    )

            # --- Ponto de impacto (último ponto) ---
            last_pt = tuple(self.trajectory_2d[-1, 0])
            if 0 <= last_pt[0] < w and 0 <= last_pt[1] < h:
                cv2.drawMarker(
                    frame, last_pt,
                    color=(0, 0, 255),  # Vermelho
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=12,
                    thickness=2,
                    line_type=cv2.LINE_AA
                )

            # --- Ponto de origem (primeiro ponto) ---
            first_pt = tuple(self.trajectory_2d[0, 0])
            if 0 <= first_pt[0] < w and 0 <= first_pt[1] < h:
                cv2.circle(frame, first_pt, 5, (0, 255, 0), -1, cv2.LINE_AA)

        # =================================================================
        # 2. MARCADORES DE DISTÂNCIA
        # =================================================================
        for pt_2d, label in self.distance_markers:
            px, py = int(pt_2d[0]), int(pt_2d[1])
            if 0 <= px < w and 0 <= py < h:
                # Tick mark
                cv2.line(frame, (px, py - 6), (px, py + 6),
                         (255, 255, 255), 1, cv2.LINE_AA)
                # Label com fundo
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (px - 2, py - th - 12), (px + tw + 2, py - 8),
                              (0, 0, 0), -1)
                cv2.putText(frame, label, (px, py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                            cv2.LINE_AA)

        # =================================================================
        # 3. RETÍCULO CENTRAL (Crosshair simples)
        # =================================================================
        cx, cy = w // 2, h // 2
        cross_size = 15
        cross_gap = 4
        cross_color = (0, 255, 0)  # Verde
        # Linhas do crosshair com gap central
        cv2.line(frame, (cx - cross_size, cy), (cx - cross_gap, cy),
                 cross_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + cross_gap, cy), (cx + cross_size, cy),
                 cross_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - cross_size), (cx, cy - cross_gap),
                 cross_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + cross_gap), (cx, cy + cross_size),
                 cross_color, 1, cv2.LINE_AA)
        # Ponto central
        cv2.circle(frame, (cx, cy), 1, cross_color, -1, cv2.LINE_AA)

        # =================================================================
        # 4. HUD — Informações na tela
        # =================================================================
        hud_color = (200, 200, 200)
        hud_bg = (20, 20, 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1

        # Painel superior esquerdo
        lines_top = [
            f"V0: {self.v0:.0f} m/s  |  Elev: {self.elevation_deg:.1f} deg",
            f"Massa: {self.mass_g:.2f}g  |  Hop: {self.hop_up:.0%}",
            f"Vento Lat: {self.wind_lateral_ms:.1f} m/s",
        ]

        # Alcance estimado (distância Z do último ponto)
        if self.trajectory_3d is not None and len(self.trajectory_3d) > 0:
            alcance = self.trajectory_3d[-1, 2]
            queda = self.trajectory_3d[-1, 1]
            lines_top.append(f"Alcance: {alcance:.1f}m  |  Queda: {queda:.2f}m")

        # Fundo semi-transparente para o HUD
        hud_h = 18 * len(lines_top) + 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (310, hud_h + 5), hud_bg, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Texto
        for i, line in enumerate(lines_top):
            y_text = 22 + i * 18
            cv2.putText(frame, line, (10, y_text),
                        font, font_scale, hud_color, thickness, cv2.LINE_AA)

        # Painel inferior — Instruções
        instructions = [
            "[Q] Sair  [W/S] V0+/-  [A/D] Elev+/-  [E/R] Hop+/-  [Z/X] Vento+/-"
        ]
        for i, line in enumerate(instructions):
            y_text = h - 12 - i * 18
            # Sombra
            cv2.putText(frame, line, (11, y_text + 1),
                        font, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
            # Texto
            cv2.putText(frame, line, (10, y_text),
                        font, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        return frame

    # =========================================================================
    # LOOP PRINCIPAL — Captura + Processamento + Exibição
    # =========================================================================
    def run(self, camera_index: int = 0):
        """
        Inicia o loop principal de captura e renderização AR.

        Args:
            camera_index: Índice da câmera USB (padrão: 0).

        Controles (teclado):
            Q / ESC  → Sair
            W / S    → Aumentar / Diminuir velocidade inicial (±5 m/s)
            A / D    → Aumentar / Diminuir ângulo de elevação (±0.5°)
            E / R    → Aumentar / Diminuir hop-up (±0.05)
            Z / X    → Aumentar / Diminuir vento lateral (±0.5 m/s)
        """
        # --- Abrir câmera ---
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"[ERRO] Nao foi possivel abrir a camera no indice {camera_index}.")
            print("       Verifique se a webcam esta conectada e nao esta em uso.")
            return

        # Configurar resolução
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # Ler resolução real (pode diferir da solicitada)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Camera aberta: {actual_w}x{actual_h}")
        print(f"[INFO] Pressione 'Q' ou ESC para sair.")
        print(f"[INFO] Use W/S, A/D, E/R, Z/X para ajustar parametros em tempo real.")

        # Se a resolução real diferir, atualizar K
        if actual_w != self.frame_width or actual_h != self.frame_height:
            self.frame_width = actual_w
            self.frame_height = actual_h
            fov_horizontal_deg = 60.0
            fx = actual_w / (2.0 * math.tan(math.radians(fov_horizontal_deg / 2.0)))
            fy = fx
            cx = actual_w / 2.0
            cy = actual_h / 2.0
            self.K = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            # Reprojetar com a nova K
            self._project_trajectory()
            self._compute_distance_markers()
            print(f"[INFO] Matriz K recalculada para {actual_w}x{actual_h}")

        window_name = "AirTrajSim AR"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        # --- Loop principal ---
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[AVISO] Falha ao capturar frame. Tentando novamente...")
                continue

            # Renderizar overlay sobre o frame
            frame = self.render_overlay(frame)

            # Exibir
            cv2.imshow(window_name, frame)

            # --- Processar teclas ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q ou ESC
                print("[INFO] Encerrando...")
                break

            elif key == ord('w'):  # V0 +5
                self.update_parameters(v0=self.v0 + 5.0)
                print(f"  V0 = {self.v0:.0f} m/s")

            elif key == ord('s'):  # V0 -5
                self.update_parameters(v0=max(10.0, self.v0 - 5.0))
                print(f"  V0 = {self.v0:.0f} m/s")

            elif key == ord('d'):  # Elevação +0.5°
                self.update_parameters(elevation_deg=self.elevation_deg + 0.5)
                print(f"  Elevacao = {self.elevation_deg:.1f} deg")

            elif key == ord('a'):  # Elevação -0.5°
                self.update_parameters(elevation_deg=self.elevation_deg - 0.5)
                print(f"  Elevacao = {self.elevation_deg:.1f} deg")

            elif key == ord('e'):  # Hop-up +0.05
                self.update_parameters(hop_up=min(1.0, self.hop_up + 0.05))
                print(f"  Hop-up = {self.hop_up:.0%}")

            elif key == ord('r'):  # Hop-up -0.05
                self.update_parameters(hop_up=max(0.0, self.hop_up - 0.05))
                print(f"  Hop-up = {self.hop_up:.0%}")

            elif key == ord('z'):  # Vento +0.5 m/s
                self.update_parameters(wind_lateral_ms=self.wind_lateral_ms + 0.5)
                print(f"  Vento lateral = {self.wind_lateral_ms:.1f} m/s")

            elif key == ord('x'):  # Vento -0.5 m/s
                self.update_parameters(wind_lateral_ms=self.wind_lateral_ms - 0.5)
                print(f"  Vento lateral = {self.wind_lateral_ms:.1f} m/s")

        # --- Liberar recursos ---
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Recursos liberados. Programa encerrado.")


# =============================================================================
# PONTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # CONFIGURAÇÃO INICIAL — Edite estes valores para seu setup:
    # -------------------------------------------------------------------------
    ar = AirTrajAR(
        # Resolução da câmera
        frame_width=640,
        frame_height=480,
        # Balística
        v0=110.0,              # Velocidade inicial (m/s) — M110 típico
        elevation_deg=1.5,     # Ângulo de elevação (graus)
        mass_g=0.20,           # Massa da BB (gramas)
        hop_up=0.35,           # Hop-up (0.0–1.0)
        # Ambiente
        gravity=9.81,
        wind_lateral_ms=0.0,   # Vento lateral (m/s)
        temp_c=25.0,           # Temperatura ambiente (°C)
        h0=1.5,                # Altura inicial do cano (m)
        # Simulação
        t_max=3.0,             # Tempo máximo de voo (s)
        dt=0.001,              # Passo de integração (s) — mesmo que main
        # Pose da câmera
        cam_height_offset=-0.05,  # Câmera 5cm acima do cano (negativo = acima)
        cam_depth_offset=0.0,
    )

    # Iniciar o loop de captura e renderização
    ar.run(camera_index=0)
