import numpy as np
import matplotlib.pyplot as plt

# --- Initialisierungen ---
human_weight = 135*9.81       # Gewicht des Patienten
mattress_weight = 0           # Gewicht der kompletten Matratze
balken_platten_weight = 85*9.81  # Masse der Holzplatten
accessories_weight = 0        # Sensor- bzw. Puffergewicht
safety_factor = 1.0
anteil = 5/7                  # Angehobener Anteil des Bettes (100cm / 140cm)

Lx = 1.0                      # Angehobene Breite in Meter (100cm von 140cm)

# --- Bereich der Winkel für den Graphen ---
alpha_range_deg = np.linspace(0, 40, 40)
alpha_range_rad = np.radians(alpha_range_deg)

def Berechne_Kraft_Verlauf(alphas_deg, alphas_rad):
    # Schwerpunkt liegt in der Mitte der angehobenen Breite
    schwerpunkt_profil = Lx / 2  # 0.5m von der Drehachse

    # Statisches Moment
    M_weight = (schwerpunkt_profil * balken_platten_weight * anteil +
                human_weight * (Lx - 0.2) +
                (mattress_weight * anteil) * (Lx / 2) +
                accessories_weight * (Lx / 2))

    force_vert = M_weight / Lx  # Kraft bei 0°

    # Sicherheitsfaktor anwenden
    force_vert_sec = force_vert * safety_factor

    # Reduktion durch Incline
    force_vert_sec_alpha = force_vert_sec * np.cos(alphas_rad)

    # Geometrie des Dump-Truck-Mechanismus
    beta = np.arctan2(0.05, 0.45)
    term1 = force_vert_sec_alpha * np.cos(np.radians(90 - alphas_deg)) * 0.1
    term2 = force_vert_sec_alpha * np.sin(np.radians(90 - alphas_deg)) * 0.15
    nenner = np.cos(beta) * 0.1

    forces = -(term1 - term2) / nenner
    return forces

def Berechne_Stroke():
    l_start = np.sqrt(0.45**2 + 0.05**2)
    l_ende = np.sqrt(0.10**2 + 0.45**2) + 0.15
    stroke = (l_ende - l_start)
    print(f"Absolut maximaler Stroke ist: {stroke*1000:.1f} mm")
    return stroke

# --- Berechnung ausführen ---
motor_forces = Berechne_Kraft_Verlauf(alpha_range_deg, alpha_range_rad)
stroke = Berechne_Stroke()

# Maximale Kraft ausgeben
max_f = np.max(motor_forces)
max_a = alpha_range_deg[np.argmax(motor_forces)]
print("\n=== Maximale Motorkraft ===")
print(f"Max. Kraft: {max_f:.1f} N  (bei alpha = {max_a:.1f}°)")

# --- Plotting ---
fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

ax1.plot(alpha_range_deg, motor_forces, label='Erforderliche Motorkraft', color='red', linewidth=2)
ax1.set_ylabel('Kraft [Newton]', fontsize=12)
ax1.set_xlabel('Kippwinkel alpha [°]', fontsize=12)
ax1.set_title(f'Analyse der Motordaten - Dump-Truck-Mechanismus (Safety Factor {safety_factor})', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

ax1.annotate(f'Max Kraft: {max_f:.0f} N', xy=(max_a, max_f), xytext=(max_a+2, max_f*0.95),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.show()
