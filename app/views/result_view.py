import flet as ft

def crear_result_view(page: ft.Page, resultados, on_nueva_analisis):
    
    # Datos Híbridos
    res_hibrido = resultados.get('hibrido', {})
    diag_hibrido = res_hibrido.get('diagnostico', 'Desconocido')
    probs = res_hibrido.get('cnn_probs', {'Corrosion': 0, 'Punzonamiento': 0})
    
    # Datos NLP
    res_texto = resultados.get('texto', {})
    diag_texto = res_texto.get('diagnostico', 'N/A')
    score_texto = res_texto.get('score', 0.0)
    input_texto = res_texto.get('input_text', '')

    # Estilos dinámicos
    is_corrosion = diag_hibrido == "Corrosión"
    main_color = "#FF9800" if is_corrosion else "#F44336" # Naranja vs Rojo
    icon_main = ft.Icons.WATER_DROP if is_corrosion else ft.Icons.WARNING_AMBER

    # --- TARJETA 1: DIAGNÓSTICO PRINCIPAL (HÍBRIDO) ---
    card_hibrido = ft.Container(
        expand=1,
        padding=25,
        bgcolor="#1c2430",
        border_radius=15,
        border=ft.border.all(2, main_color),
        content=ft.Column([
            ft.Row([
                ft.Icon(icon_main, color=main_color, size=40),
                ft.Text("DIAGNÓSTICO ESTRUCTURAL", font_family="Bebas Neue", size=28, color="white")
            ]),
            ft.Divider(),
            ft.Text("Resultado Final (Imagen + Datos):", size=12, color="grey"),
            ft.Text(
                diag_hibrido.upper(), 
                font_family="Bebas Neue", size=50, color=main_color, weight="bold"
            ),
            ft.Container(height=15),
            ft.Text("Factores Visuales (CNN):", weight="bold"),
            
            # Barras de probabilidad CNN
            ft.Column([
                ft.Row([
                    ft.Text("Corrosión", width=100),
                    ft.ProgressBar(value=probs['Corrosion'], color="#FF9800", bgcolor="#263238", expand=True),
                    ft.Text(f"{probs['Corrosion']:.0%}")
                ]),
                ft.Row([
                    ft.Text("Punzonam.", width=100),
                    ft.ProgressBar(value=probs['Punzonamiento'], color="#F44336", bgcolor="#263238", expand=True),
                    ft.Text(f"{probs['Punzonamiento']:.0%}")
                ])
            ])
        ])
    )

    # --- TARJETA 2: SEGUNDA OPINIÓN (NLP) ---
    nlp_color = "#66BB6A" # Verde
    
    card_nlp = ft.Container(
        expand=1,
        padding=25,
        bgcolor="#1c2430",
        border_radius=15,
        border=ft.border.only(left=ft.border.BorderSide(4, nlp_color)),
        content=ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.TEXT_FIELDS, color=nlp_color, size=35),
                ft.Text("ANÁLISIS SEMÁNTICO", font_family="Bebas Neue", size=28, color="white")
            ]),
            ft.Divider(),
            ft.Text("Interpretación del Texto:", size=12, color="grey"),
            ft.Text(
                diag_texto.upper(), 
                font_family="Bebas Neue", size=40, color=nlp_color
            ),
            ft.Row([
                ft.Text("Confianza del modelo:", italic=True),
                ft.Text(f"{score_texto:.1%}", weight="bold", color=nlp_color)
            ]),
            ft.Container(height=10),
            ft.Container(
                padding=10, bgcolor="#263238", border_radius=5,
                content=ft.Text(f"Input: \"{input_texto[:80]}...\"", size=12, italic=True)
            )
        ])
    )

    # --- RECOMENDACIONES ---
    rec_list = []
    if not is_corrosion: # Punzonamiento
        rec_list = [
            "🛑 RIESGO DE COLAPSO: Apuntalar la losa inmediatamente.",
            "Evacuar el área bajo la falla.",
            "Consultar urgente a ingeniero patólogo.",
            "No aumentar cargas en pisos superiores."
        ]
    else: # Corrosión
        rec_list = [
            "🛠️ MANTENIMIENTO CORRECTIVO: Picar zona afectada.",
            "Limpiar acero con cepillo o chorro de arena.",
            "Aplicar inhibidor de corrosión y puente de adherencia.",
            "Restituir recubrimiento con mortero de reparación."
        ]

    card_recs = ft.Container(
        padding=20, bgcolor="#1c2430", border_radius=10,
        content=ft.Column([
            ft.Text("RECOMENDACIONES DE INTERVENCIÓN", font_family="Bebas Neue", size=26, color="#90CAF9"),
            ft.Column([
                ft.Row([ft.Icon(ft.Icons.CHECK, color="#90CAF9"), ft.Text(txt, size=14)]) for txt in rec_list
            ])
        ])
    )

    # --- LAYOUT FINAL ---
    return ft.Container(
        padding=30, bgcolor="#0f1419", expand=True,
        content=ft.Column([
            ft.Row([
                ft.TextButton("Volver al Inicio", icon=ft.Icons.ARROW_BACK, on_click=lambda _: on_nueva_analisis()),
                ft.Text("RESULTADOS DEL ANÁLISIS", font_family="Bebas Neue", size=35)
            ]),
            ft.Divider(),
            
            ft.Row([card_hibrido, ft.Container(width=20), card_nlp], alignment="start"),
            
            ft.Container(height=20),
            card_recs,
            
            ft.Container(height=30),
            ft.ElevatedButton("NUEVA INSPECCIÓN", on_click=lambda _: on_nueva_analisis(), bgcolor="#1976D2", color="white", height=50)
        ], scroll=ft.ScrollMode.AUTO)
    )