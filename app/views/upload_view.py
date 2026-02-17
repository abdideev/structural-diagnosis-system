import flet as ft
import os

def crear_upload_view(page: ft.Page, on_analizar, on_back):
    # Estado local para la imagen
    imagen_path_ref = ft.Ref[ft.Text]()
    
    # --- 1. Preview de Imagen ---
    img_preview = ft.Image(
        src="", width=400, height=250, 
        fit=ft.ImageFit.CONTAIN, visible=False, 
        border_radius=10
    )
    
    def on_file_selected(e: ft.FilePickerResultEvent):
        if e.files:
            path = e.files[0].path
            imagen_path_ref.current.value = path
            img_preview.src = path
            img_preview.visible = True
            img_preview.update()
            btn_analizar.disabled = False
            btn_analizar.update()

    file_picker = ft.FilePicker(on_result=on_file_selected)
    page.overlay.append(file_picker)

    btn_upload = ft.ElevatedButton(
        "Subir Fotografía",
        icon=ft.Icons.CAMERA_ALT,
        on_click=lambda _: file_picker.pick_files(allow_multiple=False, allowed_extensions=["jpg", "png", "jpeg"]),
        bgcolor="#263238", color="white", height=45
    )

    # --- 2. Formulario ID3 (Campos del CSV actualizado) ---
    
    # Ubicación
    dd_ubicacion = ft.Dropdown(
        label="Ubicación del Elemento",
        options=[ft.dropdown.Option("viga"), ft.dropdown.Option("columna"), ft.dropdown.Option("losa"), ft.dropdown.Option("zapata")],
        width=400, border_color="#42A5F5"
    )

    # Dirección (Agregada opción 'radial' de tu dataset)
    dd_direccion = ft.Dropdown(
        label="Dirección de la Fisura",
        options=[ft.dropdown.Option("longitudinal"), ft.dropdown.Option("vertical"), ft.dropdown.Option("inclinada"), ft.dropdown.Option("radial")],
        width=400, border_color="#42A5F5"
    )

    # Forma (Agregadas 'piramidal', 'anillo', 'poligonal')
    dd_forma = ft.Dropdown(
        label="Forma Visual / Patrón",
        options=[
            ft.dropdown.Option("lineal"),
            ft.dropdown.Option("piramidal"),
            ft.dropdown.Option("anillo"),
            ft.dropdown.Option("difusa"),
            ft.dropdown.Option("poligonal")
        ],
        width=400, border_color="#42A5F5"
    )

    # Switches (Características binarias del CSV)
    sw_oxido = ft.Switch(label="¿Manchas de Óxido?", active_color="#42A5F5")
    sw_eflo = ft.Switch(label="¿Eflorescencias (Sal)?", active_color="#42A5F5")
    sw_desp = ft.Switch(label="¿Recubrimiento Desprendido?", active_color="#42A5F5")
    sw_hume = ft.Switch(label="¿Ambiente Húmedo?", active_color="#42A5F5")
    sw_recub = ft.Switch(label="¿Recubrimiento < 25mm?", active_color="#42A5F5")
    sw_acero = ft.Switch(label="¿Acero Visible Corroído?", active_color="#42A5F5")

    # --- 3. Descripción NLP ---
    txt_descripcion = ft.TextField(
        label="Descripción del Inspector (Análisis de Texto)",
        multiline=True, min_lines=3, max_lines=5,
        hint_text="Describa la geometría, el entorno y las observaciones técnicas...",
        width=400, border_color="green"
    )

    # --- Botón Final ---
    def on_click_analizar(e):
        path = imagen_path_ref.current.value
        if not path:
            page.snack_bar = ft.SnackBar(ft.Text("⚠️ Debes cargar una imagen obligatoriamente"))
            page.snack_bar.open = True
            page.update()
            return
        
        # Validar dropdowns
        if not (dd_ubicacion.value and dd_direccion.value and dd_forma.value):
             page.snack_bar = ft.SnackBar(ft.Text("⚠️ Por favor selecciona Ubicación, Dirección y Forma"))
             page.snack_bar.open = True
             page.update()
             return

        # Mapeo de datos exacto para predictor.py
        structural_data = {
            'ubicacion': dd_ubicacion.value,
            'direccion_fisura': dd_direccion.value,
            'forma_fisura': dd_forma.value,
            'manchas_oxido': 1 if sw_oxido.value else 0,
            'eflorescencias_sal': 1 if sw_eflo.value else 0,
            'recubrimiento_desprendido': 1 if sw_desp.value else 0,
            'ambiente_humedo': 1 if sw_hume.value else 0,
            'recubrimiento_bajo25mm': 1 if sw_recub.value else 0,
            'acero_visible_corrosion': 1 if sw_acero.value else 0
        }
        
        on_analizar(path, structural_data, txt_descripcion.value)

    btn_analizar = ft.ElevatedButton(
        "EJECUTAR DIAGNÓSTICO",
        icon=ft.Icons.CHECK_CIRCLE,
        on_click=on_click_analizar,
        bgcolor="#1976D2", color="white", height=55, width=300,
        disabled=True, # Se activa al subir foto
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10))
    )

    # --- LAYOUT ---
    return ft.Container(
        padding=30, bgcolor="#0f1419", expand=True,
        content=ft.Column([
            # Header
            ft.Row([
                ft.IconButton(icon=ft.Icons.ARROW_BACK, on_click=lambda _: on_back(), icon_color="grey"),
                ft.Text("NUEVA INSPECCIÓN TÉCNICA", font_family="Bebas Neue", size=35, color="white")
            ]),
            ft.Divider(color="grey"),
            
            ft.Row([
                # Columna Izquierda: Imagen (Visual)
                ft.Container(
                    expand=1,
                    padding=20,
                    content=ft.Column([
                        ft.Text("EVIDENCIA VISUAL (CNN)", font_family="Bebas Neue", size=24, color="#90CAF9"),
                        ft.Container(
                            content=ft.Column([
                                ft.Icon(ft.Icons.ADD_PHOTO_ALTERNATE, size=50, color="#37474F"),
                                btn_upload,
                                ft.Text("Sin imagen", ref=imagen_path_ref, size=10, italic=True, color="grey"),
                                img_preview
                            ], horizontal_alignment="center"),
                            padding=20, border=ft.border.all(1, "#37474F"), border_radius=15,
                            alignment=ft.Alignment(0,0)
                        )
                    ])
                ),
                
                # Columna Derecha: Datos (Lógica + NLP)
                ft.Container(
                    expand=1,
                    padding=20,
                    content=ft.Column([
                        ft.Text("DATOS ESTRUCTURALES (ID3)", font_family="Bebas Neue", size=24, color="#90CAF9"),
                        dd_ubicacion,
                        dd_direccion,
                        dd_forma,
                        ft.Text("Características Físicas:", weight="bold", size=12),
                        
                        # --- CORRECCIÓN: Usamos Row con wrap=True ---
                        ft.Row(
                            controls=[sw_oxido, sw_eflo, sw_desp, sw_hume, sw_recub, sw_acero], 
                            wrap=True, 
                            spacing=10
                        ),
                        
                        ft.Container(height=20),
                        ft.Text("ANÁLISIS SEMÁNTICO (NLP)", font_family="Bebas Neue", size=24, color="#A5D6A7"),
                        txt_descripcion
                    ], scroll=ft.ScrollMode.AUTO)
                )
            ], expand=True, alignment=ft.MainAxisAlignment.START),
            
            ft.Divider(color="grey"),
            ft.Container(content=btn_analizar, alignment=ft.Alignment(0,0), padding=10)
        ])
    )