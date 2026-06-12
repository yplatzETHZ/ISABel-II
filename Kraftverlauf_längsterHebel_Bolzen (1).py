import numpy as np
import matplotlib.pyplot as plt

def solve_full_bed_mechanics():
    # --- 1. Parameter ---
    WINKEL_DEG = 40
    PATIENT_BREITE_CM = 51.5
    MASSE_PATIENT_KG = 135
    GELENKE_CM = [0, 40, 70, 100, 140] # Geometrie
    REALE_SCHARNIERE = [40, 70, 100]

    # Bolzen-Parameter (Abstand zum Scharnier für die Kraftberechnung)
    ABSTAND_BOLZEN_CM = 10.0

    # Material-Parameter (2cm Multiplex)
    DICKE_M = 0.02
    DICHTE_HOLZ = 700
    LAENGE_M = 2.20
    g_acc = 9.81

    # --- 2. Eingabe & Patienten-Grenzen ---
    limit_links = PATIENT_BREITE_CM / 2
    limit_rechts = 140 - limit_links

    print(f"--- DYNAMISCHER BETT-KONFIGURATOR ---")
    try:
        raw_input = input(f"Mitte des Patienten ({limit_links:.1f}-{limit_rechts:.1f} cm): ")
        pos_mitte = float(raw_input)
        pos_mitte = max(limit_links, min(limit_rechts, pos_mitte))
    except ValueError:
        pos_mitte = 70.0

    p_start = pos_mitte - (PATIENT_BREITE_CM / 2)
    p_ende = pos_mitte + (PATIENT_BREITE_CM / 2)

    # --- 3. Dynamische Gelenkwahl ---
    erlaubte_gelenke = [g for g in REALE_SCHARNIERE if g <= p_start or g >= p_ende]

    if not erlaubte_gelenke:
        drehachse = 70
        ende = 140
    else:
        optionen = []
        for g in erlaubte_gelenke:
            if g >= p_ende:
                laenge_links = g - 0
                optionen.append((g, 0, laenge_links))
            if g <= p_start:
                laenge_rechts = 140 - g
                optionen.append((g, 140, laenge_rechts))

        beste_option = min(optionen, key=lambda x: x[2])
        drehachse, ende, _ = beste_option

    hebel_punkt_x = ende

    # --- 4. Physikalische Vorbereitung ---
    L_kipp_m = abs(ende - drehachse) / 100.0
    pos_h_rel_m = abs(hebel_punkt_x - drehachse) / 100.0

    p_dist_1 = abs(p_start - drehachse) / 100.0
    p_dist_2 = abs(p_ende - drehachse) / 100.0
    p_start_rel = max(0, min(p_dist_1, p_dist_2))
    p_ende_rel = min(L_kipp_m, max(p_dist_1, p_dist_2))

    angle_rad = np.radians(WINKEL_DEG)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    q_p_normal = (MASSE_PATIENT_KG * g_acc * cos_a) / (PATIENT_BREITE_CM / 100.0)
    q_h_normal = (LAENGE_M * L_kipp_m * DICKE_M * DICHTE_HOLZ * g_acc * cos_a) / L_kipp_m

    M_h_lager = 0.5 * q_h_normal * (L_kipp_m**2)
    wirkbreite_p = (p_ende_rel - p_start_rel)
    pos_p_mitte_rel = (p_start_rel + p_ende_rel) / 2
    M_p_lager = (q_p_normal * wirkbreite_p) * pos_p_mitte_rel

    F_stütz_normal = (M_p_lager + M_h_lager) / pos_h_rel_m

    F_lager_y = (q_h_normal * L_kipp_m + (q_p_normal * wirkbreite_p)) - F_stütz_normal

    Gewicht_Holz_N = 220 * g_acc
    Gewicht_Patient_N = (MASSE_PATIENT_KG * g_acc) * (wirkbreite_p / (PATIENT_BREITE_CM / 100.0))
    F_lager_x = (Gewicht_Holz_N + Gewicht_Patient_N) * sin_a
    F_lager_gesamt = np.sqrt(F_lager_x**2 + F_lager_y**2)

    # --- 5. Verläufe berechnen ---
    x_rel = np.linspace(0, L_kipp_m, 500)
    V_vals, M_vals = [], []

    for x in x_rel:
        v = F_lager_y - (q_h_normal * x)
        if x > p_start_rel:
            v -= q_p_normal * (min(x, p_ende_rel) - p_start_rel)
        if x >= pos_h_rel_m:
            v += F_stütz_normal
        V_vals.append(v)

        m = F_lager_y * x - 0.5 * q_h_normal * (x**2)
        if x > p_start_rel:
            w = min(x, p_ende_rel) - p_start_rel
            h = x - (p_start_rel + min(x, p_ende_rel)) / 2
            m -= (q_p_normal * w) * h
        if x >= pos_h_rel_m:
            m += F_stütz_normal * (x - pos_h_rel_m)
        M_vals.append(m)

    versteifte_scharniere = []
    bolzen_ergebnisse = {}
    for g in REALE_SCHARNIERE:
        if min(drehachse, ende) < g < max(drehachse, ende):
            versteifte_scharniere.append(g)

    for g in versteifte_scharniere:
        pos_rel_g = abs(g - drehachse) / 100.0
        M_g = np.interp(pos_rel_g, x_rel, M_vals)
        F_bolzen = abs(M_g) / (ABSTAND_BOLZEN_CM / 100.0)
        bolzen_ergebnisse[g] = {'moment': M_g, 'kraft': F_bolzen, 'x_rel': pos_rel_g}

    if ende > drehachse:
        x_global = drehachse + (x_rel * 100)
    else:
        x_global = drehachse - (x_rel * 100)
    sort_idx = np.argsort(x_global)
    x_global_sorted = x_global[sort_idx]
    V_vals_sorted = np.array(V_vals)[sort_idx]
    M_vals_sorted = np.array(M_vals)[sort_idx]

    p_plot_start = max(min(drehachse, ende), p_start)
    p_plot_ende = min(max(drehachse, ende), p_ende)

    # --- 6. Darstellung ---
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(11, 15), gridspec_kw={'height_ratios': [1.2, 3, 3]})

    plot_start, plot_ende = min(drehachse, ende), max(drehachse, ende)
    for i in range(len(GELENKE_CM)-1):
        seg_mitte = (GELENKE_CM[i] + GELENKE_CM[i+1]) / 2
        farbe = 'red' if plot_start <= seg_mitte <= plot_ende else 'blue'
        ax0.add_patch(plt.Rectangle((GELENKE_CM[i], 0.2), GELENKE_CM[i+1]-GELENKE_CM[i], 0.6,
                                     color=farbe, alpha=0.3, ec='black', lw=2))
        ax0.text(seg_mitte, 0.5, f"S{i+1}", ha='center', va='center', fontweight='bold')

    ax0.add_patch(plt.Rectangle((p_start, 0.3), p_ende-p_start, 0.4, color='orange', alpha=0.7, label='Patient'))
    ax0.axvline(drehachse, color='black', lw=4, label='Drehachse')
    ax0.plot(hebel_punkt_x, 0.2, '^', color='green', markersize=12, label='Motor')

    ax0.annotate(f"Motor:\n{F_stütz_normal:.0f} N",
                 (hebel_punkt_x, 0.2), textcoords="offset points", xytext=(0,20),
                 ha='center', fontweight='bold', color='darkgreen',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='darkgreen'))

    ax0.annotate(f"Lager (lokal):\nFx: {F_lager_x:.0f} N\nFy: {F_lager_y:.0f} N",
                 (drehachse, 0.5), textcoords="offset points", xytext=(0,35),
                 ha='center', fontweight='bold', color='black',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='black'))

    for g in versteifte_scharniere:
        ax0.plot(g, 0.5, 'X', color='purple', markersize=10)
    if versteifte_scharniere:
        ax0.plot([], [], 'X', color='purple', markersize=10, label='Verbolzt')

    ax0.set_xlim(0, 140); ax0.set_ylim(-0.2, 1.4); ax0.set_yticks([]); ax0.legend(loc='upper right')
    #ax0.set_title(f"Konfiguration: Hebelarm {pos_h_rel_m*100:.1f} cm (Neigung: {WINKEL_DEG}°)", pad=15)

    ax1.plot(x_global_sorted, V_vals_sorted, color='blue', lw=2, label='Querkraft V [N]')
    if p_plot_start < p_plot_ende:
        ax1.axvspan(p_plot_start, p_plot_ende, color='orange', alpha=0.2, label='Patient')
    ax1.set_xlim(0, 140)
    ax1.set_ylabel("Querkraft (Lokal Y) [N]")
    ax1.grid(True, ls=':')
    ax1.legend()

    ax2.plot(x_global_sorted, M_vals_sorted, color='red', lw=2, label='Biegemoment M [Nm]')
    if p_plot_start < p_plot_ende:
        ax2.axvspan(p_plot_start, p_plot_ende, color='orange', alpha=0.2, label='Patient')

    for g, data in bolzen_ergebnisse.items():
        ax2.plot(g, data['moment'], 'X', color='purple', markersize=10, zorder=5)
        ax2.annotate(f"{data['kraft']:.0f} N\nauf Bolzen",
                     (g, data['moment']), textcoords="offset points", xytext=(0,20),
                     ha='center', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='purple'))

    ax2.set_xlim(0, 140)
    ax2.set_ylabel("Moment [Nm]")
    ax2.set_xlabel("Absolute Position im Bett [cm]")
    ax2.grid(True, ls=':')
    ax2.legend()

    plt.subplots_adjust(hspace=0.45)

    # --- 7. MAXIMAL-ANALYSE (Sweep) ---
    max_bolzen = {g: {'kraft': 0.0, 'pos': 0.0} for g in REALE_SCHARNIERE}
    test_positions = np.linspace(limit_links, limit_rechts, 200) # 200 Positionen testen

    for pos_test in test_positions:
        p_st = pos_test - (PATIENT_BREITE_CM / 2)
        p_en = pos_test + (PATIENT_BREITE_CM / 2)

        erl_gelenke = [g for g in REALE_SCHARNIERE if g <= p_st or g >= p_en]
        if not erl_gelenke:
            d_achse, end = 70, 140
        else:
            opt = []
            for g in erl_gelenke:
                if g >= p_en: opt.append((g, 0, g))
                if g <= p_st: opt.append((g, 140, 140-g))
            d_achse, end, _ = min(opt, key=lambda x: x[2])

        L_k = abs(end - d_achse) / 100.0
        p_s_rel = max(0, min(abs(p_st - d_achse)/100.0, abs(p_en - d_achse)/100.0))
        p_e_rel = min(L_k, max(abs(p_st - d_achse)/100.0, abs(p_en - d_achse)/100.0))

        M_h_l = 0.5 * q_h_normal * (L_k**2)
        w_p = (p_e_rel - p_s_rel)
        p_p_m_rel = (p_s_rel + p_e_rel) / 2
        M_p_l = (q_p_normal * w_p) * p_p_m_rel

        F_st_norm = (M_p_l + M_h_l) / max(0.01, L_k) # Vermeidung von div/0
        F_l_y_test = (q_h_normal * L_k + (q_p_normal * w_p)) - F_st_norm

        versteift = [g for g in REALE_SCHARNIERE if min(d_achse, end) < g < max(d_achse, end)]

        for g in versteift:
            x = abs(g - d_achse) / 100.0
            m = F_l_y_test * x - 0.5 * q_h_normal * (x**2)
            if x > p_s_rel:
                w = min(x, p_e_rel) - p_s_rel
                h = x - (p_s_rel + min(x, p_e_rel)) / 2
                m -= (q_p_normal * w) * h

            F_b = abs(m) / (ABSTAND_BOLZEN_CM / 100.0)
            if F_b > max_bolzen[g]['kraft']:
                max_bolzen[g]['kraft'] = F_b
                max_bolzen[g]['pos'] = pos_test

    # Terminal Output am Ende, nach Schließen des Plots
    plt.show()

    print(f"\n" + "="*60)
    print(f"AKTUELLE GEOMETRIE: Drehachse bei {drehachse} cm | Motor greift bei {ende} cm an")
    print("-" * 60)
    print(f"MOTORKRAFT (bei {WINKEL_DEG}° Neigung):")
    print(f" -> Kraft (senkrecht zum Bett):     {F_stütz_normal:.1f} N  (~ {F_stütz_normal/9.81:.1f} kg)")

    print("-" * 60)
    print(f"LAGERKRÄFTE (Lokal am gekippten Brett):")
    print(f" -> Lokal X (Axialkraft parallel):  {F_lager_x:.1f} N  (~ {F_lager_x/9.81:.1f} kg)")
    print(f" -> Lokal Y (Querkraft senkrecht):  {F_lager_y:.1f} N  (~ {F_lager_y/9.81:.1f} kg)")
    print(f" -> Resultierender Gesamt-Betrag:   {F_lager_gesamt:.1f} N  (~ {F_lager_gesamt/9.81:.1f} kg)")

    if bolzen_ergebnisse:
        print("-" * 60)
        print(f"AKTUELLE BOLZEN-BELASTUNG (für Position {pos_mitte:.1f} cm):")
        for g, data in bolzen_ergebnisse.items():
            print(f" -> Bolzen an Scharnier {g:3} cm:     {data['kraft']:.1f} N  (~ {data['kraft']/9.81:.1f} kg)")

    # NEUER OUTPUT: Worst-Case Analyse
    print("=" * 60)
    print("WORST-CASE ANALYSE: Maximal mögliche Bolzenkräfte")
    print("  (Berechnet über alle möglichen Patienten-Positionen)")
    for g, data in max_bolzen.items():
        if data['kraft'] > 0: # Nur ausgeben, wenn das Scharnier jemals verbolzt wird
            print(f" -> Scharnier {g:3} cm: MAX {data['kraft']:.1f} N  (~ {data['kraft']/9.81:.1f} kg)")
            print(f"                   (Tritt auf, wenn Patientenmitte bei {data['pos']:.1f} cm liegt)")
    print("=" * 60)

if __name__ == "__main__":
    solve_full_bed_mechanics()
