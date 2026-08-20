import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

tipoCilindro = 72.0

def main():
    # ==============================================================================
    # 1. SETUP DO USUÁRIO & AMBIENTE
    # ==============================================================================
    
    # --- SELEÇÃO DE CILINDRO (EDITE AQUI) ---
    # Definição dos Tipos:
    A = 52.0 # TYPE_2_PORTED Cilindro com Janela (Padrão M4 / Cano Curto)
    B = 72.0 # TYPE_0_FULL Cilindro Fechado (Padrão DMR / Cano Longo)
         
    M190 = 90 # joules
    M100 = 100 # joules
    M110 = 110 # joules
    M120 = 120 # joules
    M130 = 130 # joules
    M140 = 140 # joules
    M150 = 150 # joules
    SP170 = 170 # 170 m/s
    
    
    # >>> ESCOLHA QUAL USAR ABAIXO <<<
    cilindro_escolhido = tipoCilindro 
    
    # Hardware Restante
    cano_len_mm = 501 
    cano_diam_mm = 6.10   
    mola_escolhida = M110 

    # Munição e Ajuste
    m_g = 0.20
    hop_percent = 0.4    
    h0 = 1.5              
    
    # Ambiente (Vento Lateral 5km/h)
    temp_c = 25.0
    vento_velocidade_kmh = 5.0  
    vento_angulo_graus = 90.0   # 90 graus (Esquerda -> Direita)

    # ==============================================================================
    # 2. BALÍSTICA INTERNA (Cálculo de Energia)
    # ==============================================================================
    
    m_kg = m_g / 1000.0
     
    '''
    v_base = mola_escolhida 
    e_base = 0.5 * m_kg * v_base**2
    
    # Usa o valor do cilindro escolhido acima
    vol_cilindro = math.pi * (23.8/2)**2 * cilindro_escolhido
    vol_cano = math.pi * (cano_diam_mm/2)**2 * cano_len_mm
    ratio = vol_cilindro / vol_cano 
    
    
    # 1. Penalidade por Falta de Ar
    if ratio < 1.9:
        fator_volume = (ratio / 1.9)**0.3
        fator_aceleracao = 1.0
    # 2. Bônus por Cano Longo (Se tiver ar sobrando)
    else:
        fator_volume = 1.0
        # Se tem ar, o cano longo ajuda a acelerar.
        # Base: 200mm. A cada mm extra, ganha energia.
        comp_util = min(cano_len_mm, 650.0) 
        fator_aceleracao = 1.0 + (comp_util - 200) * 0.0006
        
        
# 1. Penalidade por Falta de Ar (Undervolume - Cilindro pequeno para cano longo)
    if ratio < 1.9:
        fator_volume = (ratio / 1.9)**0.3
        # Se falta ar, o cano longo vira freio. Mantemos neutro ou leve penalidade.
        fator_aceleracao = 1.0 
        
    # 2. Lógica de Aceleração (Se tiver ar suficiente)
    else:
        fator_volume = 1.0
        
        # DEFININDO COMPRIMENTO MÍNIMO EFETIVO
        # Canos muito curtos (< 150mm) perdem muita eficiência exponencialmente.
        # Canos longos (> 200mm) ganham eficiência linearmente até o atrito atrapalhar.
        
        if cano_len_mm < 200:
             # Penalidade Exponencial para canos curtos
             # Ex: 10mm -> (10/200)^0.5 = 0.22 (Perde ~78% da força)
             # Ex: 100mm -> (100/200)^0.5 = 0.70 (Perde ~30% da força)
             fator_aceleracao = (cano_len_mm / 200.0)**0.6
        else:
             # Bônus Linear para canos longos
             # Base: 200mm. A cada mm extra, ganha energia.
             comp_util = min(cano_len_mm, 650.0) 
             fator_aceleracao = 1.0 + (comp_util - 200) * 0.0006        
        
    
    # Correção de Volume
    # fator_volume = (ratio / 1.9)**0.3 if ratio < 1.9 else 1.0
    
    # Correção de Cano Tightbore
    gap_std = (math.pi*(6.08/2)**2) - (math.pi*(5.95/2)**2)
    gap_new = (math.pi*(cano_diam_mm/2)**2) - (math.pi*(5.95/2)**2)
    fator_vedacao = 1.0 + ((gap_std - gap_new) / gap_std) * 0.14
    
    # Energia Final
    e_final = e_base * fator_vedacao * fator_volume * fator_aceleracao * 1.05 
    v0 = math.sqrt(2 * e_final / m_kg)
    v0_fps = v0 * 3.28084
    
    # Calibração Fina (Target 344 FPS para M100/Type2/363mm)
    # Se mudar o cilindro, o FPS vai mudar naturalmente devido ao 'fator_volume'
    # Manteremos a base de calibração fixa na M100 padrão para permitir variação.
    fator_calibracao_base = 344.0 / v0_fps
    #v0 = v0 * fator_calibracao_base
    #v0 = v0 * v0_fps
    #v0_fps = v0 * 3.28084
    #e_final = 0.5 * m_kg * v0**2 
    '''
    
    # 2. BALÍSTICA INTERNA (MÉTODO FÍSICO: TRABALHO DE EXPANSÃO)
    # ==============================================================================
    
    # A. Definir Pressão Inicial baseada na Mola
    # Uma mola M120 gera um pico de pressão.
    # Estimativa: M120 ~ 120m/s com 0.20g em cano padrão -> Pressão média necessária.
    # Vamos converter a "Etiqueta da Mola" em "Pressão Máxima do Cilindro (Pascal)"
    # Valor empírico de calibração: M100 ~ 2 bars, M120 ~ 3 bars médios efetivos na saída do nozzle
    # Fórmula aproximada para converter Rating em Pressão Inicial (Pa)
    pressao_inicial_pa = (mola_escolhida * 1920) # Ex: M120 * 3500 = 420.000 Pa (~4 atm)
    
    # B. Geometria
    area_bb = math.pi * ((5.95/1000)/2)**2      # Área onde o ar empurra
    area_cano = math.pi * ((cano_diam_mm/1000)/2)**2 
    vol_cilindro_m3 = (math.pi * (0.0238/2)**2 * (cilindro_escolhido/1000))
    
    # C. Simulação de Empuxo (Integração)
    energia_acumulada = 0.0
    posicao_bb = 0.0
    comprimento_cano_m = cano_len_mm / 1000.0
    passo_integracao = 0.001 # 1 milímetro por passo
    
    # Volume inicial é o do cilindro + nozzle (pequeno espaço morto)
    volume_atual = vol_cilindro_m3 + (area_cano * 0.005) 
    
    # Loop: Empurrar a BB até ela sair do cano
    while posicao_bb < comprimento_cano_m:
        # 1. Lei dos Gases (Adiabática): P1 * V1^gamma = P2 * V2^gamma
        # O volume aumenta conforme a BB anda. A pressão cai.
        # gamma do ar = 1.4
        
        # O volume "atrás" da BB aumenta
        volume_novo = volume_atual + (area_cano * passo_integracao)
        
        # Pressão cai conforme o volume expande
        # P_nova = P_inicial * (V_inicial / V_novo)^1.4
        # Mas atenção: O "V_inicial" aqui é o ar comprimido TOTAL disponível no cilindro
        # Se o cilindro é PORTED, temos menos ar comprimido inicial.
        
        pressao_atual = pressao_inicial_pa * (vol_cilindro_m3 / volume_novo)**1.4
        
        # 2. Força na BB (F = P * A)
        forca = pressao_atual * area_bb
        
        # 3. Trabalho (Energia) ganho neste milímetro (W = F * d)
        trabalho_passo = forca * passo_integracao
        
        # Somar energia
        energia_acumulada += trabalho_passo
        
        # Atualizar
        volume_atual = volume_novo
        posicao_bb += passo_integracao
        
        # SE O CILINDRO ACABAR (Pistão bateu no fim):
        # Na vida real, a pressão cai subitamente, mas ainda há ar residual expandindo.
        # O modelo adiabático acima já cuida disso (pressão cai rápido).

    # D. Perdas por Atrito e Vazamento (Gap)
    # Quanto maior o gap (6.08 vs 6.03), mais pressão vaza
    gap_area = area_cano - area_bb
    eficiencia_vedacao = 1.0 - (gap_area / area_bb) * 5.0 # Fator de perda por vazamento
    
    energia_cinetica_final = energia_acumulada * eficiencia_vedacao * 0.85 # 0.85 = Atrito cano/bucking
    
    # E. Resultados Finais
    v0 = math.sqrt(2 * energia_cinetica_final / m_kg)
    v0_fps = v0 * 3.28084
    e_final = energia_cinetica_final

    # Nome para o gráfico
    nome_cilindro_str = "Cilindro Ported (Type 2)" if cilindro_escolhido == 52.0 else "Cilindro Full (Type 0)"

    print(f"--- SETUP ---")
    print(f"Configuração: {mola_escolhida} | Cano: {cano_len_mm}mm ({cano_diam_mm}mm)")
    print(f"Cilindro: {nome_cilindro_str} (Comp. Útil: {cilindro_escolhido}mm)")
    #print(f"Ratio Volumétrico: {ratio:.2f}")
    print(f"Saída Estimada: {v0_fps:.1f} FPS | {e_final:.2f} Joules")

    # ==============================================================================
    # 3. FÍSICA 3D (COM VENTO)
    # ==============================================================================
    rho = 101325 / (287.05 * (temp_c + 273.15))
    r_bb = (5.95 / 1000) / 2
    A = np.pi * r_bb**2
    
    # Vento
    v_vento_mag = vento_velocidade_kmh / 3.6 
    theta_rad = math.radians(vento_angulo_graus)
    Wx = v_vento_mag * math.cos(theta_rad)
    Wz = v_vento_mag * math.sin(theta_rad)
    Wy = 0.0

    # Rotação
    rpm_max = 90000.0  
    omega_0 = (rpm_max * hop_percent) * (2 * np.pi / 60)
    
    # Coeficientes
    Cd_base = 0.50 
    fator_eficiencia_magnus = 0.18

    # Estado Inicial [x, y, z, vx, vy, vz, omega]
    state = np.array([0.0, h0, 0.0, v0, 0.0, 0.0, omega_0]) 
    
    def derivatives(t, s):
        x, y, z, vx, vy, vz, w = s
        
        v_rel_x = vx - Wx
        v_rel_y = vy - Wy
        v_rel_z = vz - Wz
        v_rel = np.sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2)
        if v_rel < 0.1: v_rel = 0.1
            
        spin_param = (r_bb * w) / v_rel
        Cl = min(0.25, fator_eficiencia_magnus * spin_param)
        
        Fd = 0.5 * rho * A * Cd_base * v_rel**2
        Fl = 0.5 * rho * A * Cl * v_rel**2
        
        ax_drag = -(Fd/m_kg) * (v_rel_x / v_rel)
        ay_drag = -(Fd/m_kg) * (v_rel_y / v_rel)
        az_drag = -(Fd/m_kg) * (v_rel_z / v_rel)
        
        ay_lift = (Fl/m_kg) 
        ay_grav = -9.81
        decay = -0.30 * w 
        
        return np.array([vx, vy, vz, ax_drag, ay_grav + ay_drag + ay_lift, az_drag, decay])

    # Loop RK4
    dt = 0.001
    trajectory = [state]
    t = 0.0
    while True:
        k1 = derivatives(t, state)
        k2 = derivatives(t + 0.5*dt, state + 0.5*dt*k1)
        k3 = derivatives(t + 0.5*dt, state + 0.5*dt*k2)
        k4 = derivatives(t + dt, state + dt*k3)
        state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        t += dt
        if state[1] < 0 or state[0] > 100:
            trajectory.append(state)
            break
        trajectory.append(state)
        
    traj = np.array(trajectory)
    dist_total = traj[-1, 0]
    drift_total = traj[-1, 2]
    
    # ==============================================================================
    # 4. PLOTAGEM (3 VISÕES)
    # ==============================================================================
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])

    # 1. LATERAL
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(traj[:,0], traj[:,1], color='#e74c3c', linewidth=2)
    ax1.axhline(y=h0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title(f"1. Visão Lateral | {nome_cilindro_str}\nSaída: {v0_fps:.1f} FPS", fontweight='bold')
    ax1.set_ylabel("Altura (m)")
    ax1.set_xlabel("Distância (m)")
    ax1.grid(True, alpha=0.4)
    ax1.set_ylim(0, max(2.5, np.max(traj[:,1]) + 0.5))
    ax1.set_xlim(0, dist_total + 5)

    # 2. SUPERIOR
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(traj[:,0], traj[:,2], color='#2980b9', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_title(f"2. Visão Superior (Vento {vento_velocidade_kmh}km/h Direita)", fontweight='bold')
    ax2.set_ylabel("Desvio (m) [Z]")
    ax2.set_xlabel("Distância (m) [X]")
    ax2.grid(True, alpha=0.4)
    limit_z = max(0.5, abs(drift_total)*1.2)
    ax2.set_ylim(-limit_z, limit_z)
    ax2.set_xlim(0, dist_total + 5)
    
    # 3. TRASEIRA
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(traj[:,2], traj[:,1], color='#8e44ad', linewidth=2.5)
    ax3.scatter([0], [h0], color='black', marker='+', s=100, label='Mira')
    
    for d in range(10, int(dist_total), 10):
        idx = (np.abs(traj[:,0] - d)).argmin()
        ax3.scatter(traj[idx, 2], traj[idx, 1], color='black', s=20)
        ax3.text(traj[idx, 2], traj[idx, 1]+0.1, f'{d}m', fontsize=8, ha='center')

    ax3.set_title("3. Visão de Trás (Shooter's View)", fontweight='bold')
    ax3.set_ylabel("Altura (m) [Y]")
    ax3.set_xlabel("Desvio Lateral (m) [Z]")
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.4)
    ax3.set_ylim(0, max(2.5, np.max(traj[:,1]) + 0.5))
    ax3.set_xlim(-limit_z, limit_z)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()