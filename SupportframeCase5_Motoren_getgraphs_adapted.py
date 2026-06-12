import numpy as np
import matplotlib.pyplot as plt

# --- Konstanten ---
human_weight = 135*9.81
mattress_weight = 0
balken_platten_weight = 85*9.81
accessories_weight = 0
anzahl_linearM = 1
Lx = 0.7
Lz = 1.9
safety_factor = 1.0
L_hebel_mm = 500  # Abstand Lagerung zu Frame

# Statisches Moment (Maximales Moment bei alpha=0)
schwerpunkt_profil = (Lx/2*Lx + Lx/2*Lx + Lx*Lz)/(2*Lx+Lz)
M_weight = (schwerpunkt_profil * balken_platten_weight/2 +
            human_weight * (Lx - 0.2) +
            (mattress_weight / 2) * (Lx / 2) +
            accessories_weight * (Lx / 2))

# --- Kernfunktionen ---
def get_system_values(x_vec, alpha_deg, beta0_deg):
    a_rad = np.radians(alpha_deg)
    b0_rad = np.radians(beta0_deg)

    # Geometrie
    h0 = x_vec * np.sin(b0_rad)
    d0 = x_vec * np.cos(b0_rad)
    S1 = np.sqrt(h0**2 + d0**2)  # Startlänge

    h1 = h0 + x_vec * np.sin(a_rad)
    d1 = d0 + (x_vec - x_vec * np.cos(a_rad))
    S2 = np.sqrt(h1**2 + d1**2)  # Aktuelle Länge

    beta_now_rad = np.arctan2(h1, d1)
    gamma_deg = 180 - np.degrees(beta_now_rad) - alpha_deg

    # Kraftberechnung Motor (inkl. Safety)
    M_aktuell = M_weight * np.cos(a_rad)
    nenner_kraft = np.sin(np.radians(gamma_deg)) * x_vec * anzahl_linearM
    force = (M_aktuell * safety_factor) / nenner_kraft
    stroke = (S2 - S1) * 1000  # in mm

    # Wellenberechnung
    total_load_N = (human_weight + mattress_weight + balken_platten_weight + accessories_weight) * safety_factor

    F_welle_langs = (total_load_N / 2) * np.sin(a_rad) - force * np.cos(np.radians(gamma_deg))
    F_welle_quer = (total_load_N / 2) * np.cos(a_rad) - force * np.sin(np.radians(gamma_deg))
    F_res_welle = np.sqrt(F_welle_langs**2 + F_welle_quer**2)

    # Dimensionierung Durchmesser
    Re = 235
    safety_material = 1.5
    sigma_zul = Re / safety_material

    Mb_max = F_res_welle * L_hebel_mm
    welle_d = np.cbrt((32 * Mb_max) / (np.pi * sigma_zul))

    return force, stroke, welle_d

# --- Daten generieren ---
x_range = np.linspace(0.2, 0.7, 50)
beta0_test = 90

f0, s0, d0 = get_system_values(x_range, 0, beta0_test)
f40, s40, d40 = get_system_values(x_range, 40, beta0_test)

# --- Maximale Motorkraft ausgeben ---
max_force_0 = np.max(f0)
max_force_40 = np.max(f40)
x_at_max_0 = x_range[np.argmax(f0)]
x_at_max_40 = x_range[np.argmax(f40)]

print("=== Maximale Motorkraft ===")
print(f"alpha = 0°:  {max_force_0:.1f} N  (bei x = {x_at_max_0:.3f} m)")
print(f"alpha = 40°: {max_force_40:.1f} N  (bei x = {x_at_max_40:.3f} m)")

# --- Maximale Motorkraft am definierten Hebelarm ausgeben ---
f_hebel, _, _ = get_system_values(np.array([L_hebel_mm / 1000]), 0, beta0_test)
print("\n=== Maximale Motorkraft am definierten Hebelarm ===")
print(f"Hebelarm: {L_hebel_mm} mm")
print(f"Max. Kraft (alpha = 0°, Worst Case): {f_hebel[0]:.1f} N")
# --- Maximale Motorkraft am definierten Hebelarm ausgeben ---
f_hebel, s_hebel, _ = get_system_values(np.array([L_hebel_mm / 1000]), 0, beta0_test)
print("\n=== Maximale Motorkraft am definierten Hebelarm ===")
print(f"Hebelarm: {L_hebel_mm} mm")
print(f"Max. Kraft (alpha = 0°, Worst Case): {f_hebel[0]:.1f} N")

# --- Hub am definierten Hebelarm ausgeben ---
_, s_hebel_40, _ = get_system_values(np.array([L_hebel_mm / 1000]), 40, beta0_test)
print(f"Benötigter Hub (alpha = 0° -> 40°): {s_hebel_40[0]:.1f} mm")


# --- Plotting ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

# 1. Kraft-Plot
ax1.plot(x_range, f0, 'r-', label='alpha = 0° (Max. Lastmoment)')
ax1.plot(x_range, f40, 'r--', label='alpha = 40° (Incline)')
ax1.set_ylabel('Kraft pro Motor [N]')
ax1.set_title(f'Motorkraft (Safety Factor {safety_factor})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Hub-Plot
ax2.plot(x_range, s40, 'b-', label='Stroke für 40°')
ax2.set_ylabel('Benötigter Stroke [mm]')
ax2.set_title('Motor-Hub')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Wellen-Plot
ax3.plot(x_range, d0, 'g-', label='d_welle bei 0°')
ax3.plot(x_range, d40, 'g--', label='d_welle bei 40°')
ax3.axhline(y=20, color='gray', linestyle=':', label='Standardmaß 20mm')
ax3.set_ylabel('Mindestdurchmesser [mm]')
ax3.set_xlabel(
    'Abstand x: Angriffspunkt des Linearmotors zur Drehachse [m]\n'
    '(x = 0.2m: kurzer Hebelarm  |  x = 0.7m: langer Hebelarm)',
    fontsize=9
)
ax3.set_title(f'Wellen-Dimensionierung (Re=235MPa, Hebel={L_hebel_mm}mm)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()
