import flet as ft
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Oculta INFO y WARNINGS de TF
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Desactiva el aviso de oneDNN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from views.upload_view import crear_upload_view
from views.result_view import crear_result_view
from services.predictor import FissurePredictor

class FisuraApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.predictor = None
        self.configurar_pagina()
        self.inicializar_app()

    def configurar_pagina(self):
        """Configura ventana, tema y FUENTES (Bebas Neue + Montserrat)."""
        self.page.title = "Sistema Experto de Fisuras Estructurales"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.window_width = 1200
        self.page.window_height = 900
        self.page.padding = 0
        self.page.bgcolor = "#0f1419"

        # --- CARGAR FUENTES ONLINE ---
        self.page.fonts = {
            "Bebas Neue": "https://github.com/google/fonts/raw/main/ofl/bebasneue/BebasNeue-Regular.ttf",
            "JetBrains Mono": "https://github.com/google/fonts/raw/main/ofl/jetbrainsmono/JetBrainsMono-Regular.ttf",
            "JetBrains Mono Bold": "https://github.com/google/fonts/raw/main/ofl/jetbrainsmono/JetBrainsMono-Bold.ttf"
        }

        self.page.theme = ft.Theme(
            font_family="JetBrains Mono",
            use_material3=True
        )

    def inicializar_app(self):
        """Muestra pantalla de carga mientras levanta los modelos."""
        loading = ft.Column(
            [
                ft.ProgressRing(color="blue"),
                ft.Text("Cargando Modelos de IA (CNN + ID3 + NLP)...", size=16, weight="bold")
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )
        self.page.add(ft.Container(loading, alignment=ft.Alignment(0,0), expand=True))
        self.page.update()

        try:
            # Instanciar predictor (Carga los 3 modelos)
            self.predictor = FissurePredictor()
            self.mostrar_pantalla_inicio()
        except Exception as e:
            self.page.controls.clear()
            self.page.add(ft.Container(
                ft.Column([
                    ft.Icon(ft.Icons.ERROR, color="red", size=50),
                    ft.Text(f"Error crítico cargando modelos:\n{e}", color="red", size=16)
                ], alignment="center"),
                alignment=ft.Alignment(0,0), expand=True
            ))
            self.page.update()

    def mostrar_pantalla_inicio(self):
        """Pantalla de Bienvenida."""
        self.page.controls.clear()
        
        content = ft.Container(
            content=ft.Column([
                ft.Icon(ft.Icons.DOMAIN, size=80, color="#1976D2"),
                ft.Text(
                    "SISTEMA EXPERTO DE FISURAS ESTRUCTURALES", 
                    font_family="Bebas Neue", 
                    size=50, 
                    color="white",
                    text_align="center"
                ),
                ft.Text(
                    "Diagnóstico inteligente mediante Visión Artificial (CNN), Árboles de Decisión (ID3) y Procesamiento de Lenguaje Natural (LLM)",
                    size=16,
                    color="grey",
                    text_align="center"
                ),
                ft.Container(height=30),
                ft.ElevatedButton(
                    "INICIAR NUEVA INSPECCIÓN",
                    on_click=lambda _: self.mostrar_vista_upload(),
                    bgcolor="#1976D2",
                    color="white",
                    height=60,
                    style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10))
                )
            ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            alignment=ft.alignment.center,
            expand=True,
            padding=20
        )
        
        self.page.add(content)
        self.page.update()

    def mostrar_vista_upload(self):
        self.page.controls.clear()
        view = crear_upload_view(
            self.page,
            on_analizar=self.ejecutar_analisis,
            on_back=self.mostrar_pantalla_inicio
        )
        self.page.add(view)
        self.page.update()

    def ejecutar_analisis(self, img_path, struct_data, text_desc):
        """Callback que ejecuta la predicción."""
        
        # Modal de carga
        progress_dialog = ft.AlertDialog(
            modal=True,
            content=ft.Container(
                height=100,
                content=ft.Column([
                    ft.ProgressRing(), 
                    ft.Text("Procesando imagen y datos...", font_family="Montserrat")
                ], alignment="center", horizontal_alignment="center")
            )
        )
        self.page.dialog = progress_dialog
        progress_dialog.open = True
        self.page.update()

        try:
            resultados = self.predictor.predict(img_path, struct_data, text_desc)
            
            progress_dialog.open = False
            self.mostrar_resultados(resultados)
            
        except Exception as e:
            progress_dialog.open = False
            self.page.snack_bar = ft.SnackBar(ft.Text(f"Error en análisis: {e}"))
            self.page.snack_bar.open = True
            self.page.update()

    def mostrar_resultados(self, resultados):
        self.page.controls.clear()
        view = crear_result_view(
            self.page,
            resultados,
            on_nueva_analisis=self.mostrar_vista_upload
        )
        self.page.add(view)
        self.page.update()

def main(page: ft.Page):
    FisuraApp(page)

if __name__ == "__main__":
    ft.app(target=main)