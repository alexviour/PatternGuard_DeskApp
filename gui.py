import os
import time
import threading
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageOps

from evaluation_metrics import calculate_restoration_metrics
from cultural_data import TEXTILE_METADATA
from dsp_engine import (
    run_algorithm_i_autocorrelation,
    run_algorithm_ii_defect_isolation,
    run_algorithm_iii_template_alignment,
)
from pdf_exporter import generate_patternguard_pdf
from pattern_classifier import PatternClassifier


def open_tips_modal(parent):
    modal = tk.Toplevel(parent)
    modal.title("📸 Capture Tips for Best Results")
    modal.resizable(False, False)
    modal.configure(bg="#F4EFEA")
    modal.transient(parent.winfo_toplevel())

    content = tk.Frame(
        modal,
        bg="#FFFFFF",
        highlightbackground="#E3DCD6",
        highlightthickness=1,
    )
    content.pack(fill="both", expand=True, padx=20, pady=15)

    title = tk.Label(
        content,
        text="📸 Capture Tips for Best Results",
        bg="#FFFFFF",
        fg="#1D265C",
        font=("Segoe UI", 15, "bold"),
        anchor="w",
        wraplength=440,
    )
    title.pack(fill="x", padx=20, pady=(20, 15))

    tips = (
        ("📐 Capture Overhead", "Take photos directly from above to prevent perspective skew."),
        ("💡 Ensure Even Lighting", "Use balanced light to eliminate unnecessary shadows."),
        ("🧵 Keep Fabric Flat", "Smooth out wrinkles so the geometric grid stays aligned."),
        ("🔍 Frame Full Context", "Ensure intact pattern motifs surround the damaged area."),
        ("📷 Use High-Res Images", "Clear focus allows a more precise restoration."),
    )

    tips_frame = tk.Frame(content, bg="#FFFFFF")
    tips_frame.pack(fill="both", expand=True, padx=20)

    for heading, description in tips:
        tip_row = tk.Frame(
            tips_frame,
            bg="#FAF6F0",
            highlightbackground="#E3DCD6",
            highlightthickness=1,
        )
        tip_row.pack(fill="x", pady=4)
        tk.Label(
            tip_row,
            text=heading,
            bg="#FAF6F0",
            fg="#1D265C",
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(8, 2))
        tk.Label(
            tip_row,
            text=description,
            bg="#FAF6F0",
            fg="#2E2E30",
            font=("Segoe UI", 9),
            anchor="w",
            justify="left",
            wraplength=420,
        ).pack(fill="x", padx=12, pady=(0, 8))

    close_button = tk.Button(
        content,
        text="Got It",
        bg="#1D265C",
        fg="#FFFFFF",
        activebackground="#283882",
        activeforeground="#FFFFFF",
        font=("Segoe UI", 10, "bold"),
        relief="flat",
        bd=0,
        padx=24,
        pady=8,
        cursor="hand2",
        command=modal.destroy,
    )
    close_button.pack(pady=(15, 20))
    modal.bind("<Escape>", lambda event: modal.destroy())

    modal.update_idletasks()
    parent_window = parent.winfo_toplevel()
    x = parent_window.winfo_rootx() + (parent_window.winfo_width() - modal.winfo_width()) // 2
    y = parent_window.winfo_rooty() + (parent_window.winfo_height() - modal.winfo_height()) // 2
    modal.geometry(f"+{max(0, x)}+{max(0, y)}")
    modal.grab_set()
    modal.focus_set()


class PatternGuardApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PatternGuard Desktop")
        self.geometry("1280x880")
        self.minsize(1200, 840)

        # State Variables
        self.classifier = PatternClassifier()
        self.uploaded_file_path = None
        self.original_bgr = None
        self.original_image_pil = None
        self.original_image_hires = None
        self.tracking_pattern_pil = None
        self.heatmap_hires = None
        self.reconstructed_image_pil = None
        self.reconstructed_image_hires = None
        self.img_tk_cache = {}
        self.active_subclass_id = None
        self.classification_result = None
        self.reference_window = None
        self.current_page = "upload"
        self._wheel_remainder = 0.0

        self.configure_styles()
        self.create_layout()

    def configure_styles(self):
        self.colors = {
            "bg": "#F4EFEA",
            "card": "#FFFDF8",
            "border": "#D9CBB8",
            "panel": "#F8F0E8",
            "text_primary": "#392019",
            "text_dark": "#2F2521",
            "text_grey": "#76685F",
            "maroon": "#8C2D2D",
            "maroon_hover": "#A63A3A",
            "navy_btn": "#2B1712",
            "navy_btn_hover": "#4A2A20",
            "gold": "#D69A70",
            "banner_bg": "#F5E9DB",
            "banner_border": "#9B6B47",
            "emerald": "#5C7453",
            "soft_gold": "#E6D4B5",
        }
        self.configure(bg=self.colors["bg"])
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(
            "TCombobox",
            fieldbackground="#FFFFFF",
            background="#FFFFFF",
            foreground=self.colors["text_dark"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["border"],
            darkcolor=self.colors["border"],
        )

    def create_layout(self):
        # HEADER BAR
        header_bar = tk.Frame(self, bg="#2B1712", height=88, bd=0, relief="flat")
        header_bar.pack(fill="x", side="top")
        header_bar.pack_propagate(False)

        brand_frame = tk.Frame(header_bar, bg="#2B1712")
        brand_frame.pack(side="left", padx=(25, 16), pady=18)

        self.logo_frame = tk.Frame(
            brand_frame,
            bg="#4A2A20",
            highlightbackground="#D69A70",
            highlightthickness=1,
            bd=0,
            padx=12,
            pady=8,
        )
        self.logo_frame.pack(side="left")

        self.logo_lbl = tk.Label(
            self.logo_frame,
            text="PG",
            bg="#4A2A20",
            fg="#F4EFEA",
            font=("Segoe UI", 12, "bold"),
            cursor="hand2",
        )
        self.logo_lbl.pack()
        self.logo_frame.configure(cursor="hand2")
        self.logo_frame.bind("<Button-1>", lambda event: self.show_page("results"))
        self.logo_lbl.bind("<Button-1>", lambda event: self.show_page("results"))

        title_frame = tk.Frame(header_bar, bg="#2B1712")
        title_frame.pack(side="left", fill="y", pady=18)

        title_lbl = tk.Label(
            title_frame,
            text="PATTERNGUARD",
            bg="#2B1712",
            fg="#FFFFFF",
            font=("Segoe UI", 17, "bold"),
        )
        title_lbl.pack(anchor="w")

        sub_title_lbl = tk.Label(
            title_frame,
            text="FILIPINO TEXTILE AUDITING & RESTORATION FRAMEWORK",
            bg="#2B1712",
            fg="#E9D7C6",
            font=("Segoe UI", 8, "bold"),
        )
        sub_title_lbl.pack(anchor="w", pady=(2, 0))

        nav_frame = tk.Frame(header_bar, bg="#2B1712")
        nav_frame.pack(side="right", padx=(0, 25), pady=20)

        self.dashboard_tab = tk.Button(
            nav_frame,
            text="Dashboard",
            bg="#2B1712",
            fg="#F4EFEA",
            activebackground="#4A2A20",
            activeforeground="#F4EFEA",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            bd=0,
            padx=14,
            pady=8,
            cursor="hand2",
            command=lambda: self.show_page("results"),
        )
        self.dashboard_tab.pack(side="left")

        self.reference_tab = tk.Button(
            nav_frame,
            text="Reference Guide",
            bg="#2B1712",
            fg="#F4EFEA",
            activebackground="#4A2A20",
            activeforeground="#F4EFEA",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            bd=0,
            padx=14,
            pady=8,
            cursor="hand2",
            command=lambda: self.show_reference_guide(),
        )
        self.reference_tab.pack(side="left", padx=(10, 0))

        sep = tk.Frame(self, bg=self.colors["gold"], height=3)
        sep.pack(fill="x", side="top")

        workspace_wrapper = tk.Frame(self, bg=self.colors["bg"])
        workspace_wrapper.pack(fill="both", expand=True)

        self.upload_action_bar = tk.Frame(
            workspace_wrapper,
            bg="#EAE1D7",
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )
        tk.Label(
            self.upload_action_bar,
            text="AUDIT ACTION",
            bg="#EAE1D7",
            fg=self.colors["text_grey"],
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=18, pady=10)
        self.upload_reconstruct_btn = self.create_flat_button(
            self.upload_action_bar,
            text="Run audit pipeline",
            bg=self.colors["navy_btn"],
            fg="#FFFFFF",
            hover_bg=self.colors["navy_btn_hover"],
            command=self.run_reconstruction_pipeline,
            font=("Segoe UI", 10, "bold"),
            height=42,
        )
        self.upload_reconstruct_btn.pack(side="right", padx=12, pady=7)
        self.upload_reconstruct_btn.configure(state="disabled")
        self.upload_action_bar.pack(fill="x", side="top")

        self.scrollbar = ttk.Scrollbar(workspace_wrapper, orient="vertical")
        self.scrollbar.pack(side="right", fill="y")

        self.canvas = tk.Canvas(
            workspace_wrapper,
            bg=self.colors["bg"],
            bd=0,
            highlightthickness=0,
            yscrollcommand=self.scrollbar.set,
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.config(command=self.canvas.yview)

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

        main_container = tk.Frame(self.canvas, bg=self.colors["bg"])
        self.canvas_frame_id = self.canvas.create_window(
            (0, 0), window=main_container, anchor="nw"
        )

        main_container.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self.canvas_frame_id, width=e.width),
        )

        self.upload_page = tk.Frame(main_container, bg=self.colors["bg"])
        self.result_page = tk.Frame(main_container, bg=self.colors["bg"])
        self.reference_page = tk.Frame(main_container, bg=self.colors["bg"])
        self.upload_page.pack(fill="both", expand=True)
        self.result_page.pack(fill="both", expand=True)
        self.reference_page.pack(fill="both", expand=True)

        self.show_page("upload")

        self.upload_left = tk.Frame(self.upload_page, bg=self.colors["bg"], width=620)
        self.upload_left.pack(side="left", fill="both", padx=(25, 12), pady=20)
        self.upload_left.pack_propagate(False)

        self.upload_right = tk.Frame(self.upload_page, bg=self.colors["bg"], width=360)
        self.upload_right.pack(side="left", fill="both", expand=True, padx=(12, 25), pady=20)

        self.build_upload_page(self.upload_left, self.upload_right)

        self.result_left = tk.Frame(self.result_page, bg=self.colors["bg"], width=380)
        self.result_left.pack(side="left", fill="both", padx=(25, 0), pady=20)
        self.result_left.pack_propagate(False)

        self.result_right = tk.Frame(self.result_page, bg=self.colors["bg"])
        self.result_right.pack(side="left", fill="both", expand=True, padx=(25, 25), pady=20)

        self.build_left_column(self.result_left)
        self.build_right_column(self.result_right)

        self.result_page.pack_forget()

    def _on_mousewheel(self, event):
        if hasattr(event, "delta") and event.delta:
            self._wheel_remainder -= event.delta
            delta = int(self._wheel_remainder / 120)
            self._wheel_remainder -= delta * 120
        elif getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            return
        if not delta:
            return
        self.canvas.yview_scroll(delta, "units")

    def show_page(self, page_name):
        self.upload_page.pack_forget()
        self.result_page.pack_forget()
        self.reference_page.pack_forget()

        if page_name == "upload":
            self.upload_action_bar.pack(fill="x", side="top", before=self.canvas)
            self.upload_page.pack(fill="both", expand=True)
            self.dashboard_tab.configure(bg="#2B1712", fg="#F4EFEA")
        elif page_name == "results":
            self.upload_action_bar.pack_forget()
            self.result_page.pack(fill="both", expand=True)
            self.dashboard_tab.configure(bg="#2B1712", fg="#F4EFEA")
        elif page_name == "reference":
            self.upload_action_bar.pack_forget()
            self.reference_page.pack(fill="both", expand=True)
            self.dashboard_tab.configure(bg="#4A2A20", fg="#FFFFFF")
        else:
            raise ValueError(f"Unknown page: {page_name}")

        self.current_page = page_name
        self.canvas.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.yview_moveto(0)

    def build_upload_page(self, left, right):
        title_block = tk.Frame(left, bg=self.colors["bg"])
        title_block.pack(fill="x", pady=(10, 12))
        tk.Label(
            title_block,
            text="RESTORATION WORKSPACE / NEW AUDIT",
            bg=self.colors["bg"],
            fg=self.colors["maroon"],
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_block,
            text="Begin a Textile Audit",
            bg=self.colors["bg"],
            fg=self.colors["text_primary"],
            font=("Segoe UI", 28, "bold"),
            anchor="w",
        ).pack(anchor="w", pady=(6, 4))
        tk.Label(
            title_block,
            text="Upload one clear textile image. PatternGuard will classify the weave, inspect its symmetry, and reconstruct detected anomalies.",
            bg=self.colors["bg"],
            fg=self.colors["text_dark"],
            font=("Segoe UI", 12),
            justify="left",
            wraplength=520,
        ).pack(anchor="w")

        stepper = tk.Frame(left, bg=self.colors["bg"])
        stepper.pack(anchor="e", pady=(10, 18))
        self.upload_stepper = stepper

        steps = ["Upload image", "Review results"]
        for idx, step in enumerate(steps):
            dot = tk.Label(
                stepper,
                text=str(idx + 1) if idx == 0 else str(idx + 1),
                bg="#D8D1C9" if idx == 0 else "#F3EFEA",
                fg="#2B1712" if idx == 0 else "#76685F",
                font=("Segoe UI", 9, "bold"),
                justify="center",
                width=3,
                relief="flat",
            )
            dot.grid(row=0, column=idx * 2, padx=(0, 4), sticky="n")
            label = tk.Label(
                stepper,
                text=step,
                bg=self.colors["bg"],
                fg="#2B1712" if idx == 0 else "#76685F",
                font=("Segoe UI", 10),
            )
            label.grid(row=0, column=idx * 2 + 1, padx=(0, 12))
        if len(steps) > 1:
            tk.Label(stepper, text="", bg=self.colors["bg"]).grid(row=0, column=2, sticky="n")

        upload_card = tk.Frame(left, bg="#F5F1EB", highlightbackground=self.colors["border"], highlightthickness=1, bd=0)
        upload_card.pack(fill="both", expand=True)

        title_small = tk.Label(
            upload_card,
            text="STEP 01 / SOURCE IMAGE",
            bg="#F5F1EB",
            fg=self.colors["text_grey"],
            font=("Segoe UI", 9, "bold"),
        )
        title_small.pack(anchor="w", padx=18, pady=(18, 4))

        tk.Label(
            upload_card,
            text="Select a textile image",
            bg="#F5F1EB",
            fg=self.colors["text_primary"],
            font=("Segoe UI", 18, "bold"),
            anchor="w",
        ).pack(anchor="w", padx=18, pady=(0, 10))

        upload_inner = tk.Frame(upload_card, bg="#F5F1EB")
        upload_inner.pack(fill="both", expand=True)

        upload_panel = tk.Frame(upload_inner, bg="#F5F1EB", highlightbackground=self.colors["border"], highlightthickness=1, bd=0)
        upload_panel.pack(fill="x", padx=18, pady=(0, 12))
        upload_panel.configure(height=250)

        self.upload_page_canvas = tk.Canvas(
            upload_panel,
            height=230,
            bg="#F5F1EB",
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        self.upload_page_canvas.pack(fill="both", expand=True)
        self.upload_page_canvas.bind(
            "<Button-1>", lambda e: self.trigger_file_upload()
        )
        self.upload_page_canvas.bind(
            "<Configure>",
            lambda event: self.draw_upload_slot(self.upload_page_canvas),
        )
        self.draw_upload_slot(self.upload_page_canvas)

        selected_frame = tk.Frame(upload_card, bg="#F1E4D5", highlightbackground=self.colors["border"], highlightthickness=1, bd=0)
        selected_frame.pack(fill="x", padx=18, pady=(0, 18))

        self.file_row = tk.Frame(selected_frame, bg="#F1E4D5")
        self.file_row.pack(fill="x", padx=12, pady=10)

        self.upload_file_name_lbl = tk.Label(
            self.file_row,
            text="No file selected",
            bg="#F1E4D5",
            fg=self.colors["text_dark"],
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        )
        self.upload_file_name_lbl.pack(side="left", fill="x", expand=True)

        self.upload_re_upload_btn = tk.Button(
            self.file_row,
            text="Re-Upload",
            bg="#F1E4D5",
            fg=self.colors["maroon"],
            activebackground="#F9EDE5",
            activeforeground=self.colors["maroon"],
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            bd=0,
            highlightbackground=self.colors["maroon"],
            highlightthickness=1,
            cursor="hand2",
            command=self.trigger_file_upload,
        )
        self.upload_re_upload_btn.pack(side="right", padx=6)

        self.upload_banner_frame = tk.Frame(
            selected_frame,
            bg="#ECF2E8",
            highlightbackground="#7BA06E",
            highlightthickness=1,
            bd=0,
        )
        self.upload_banner_frame.pack(fill="x", padx=12, pady=(0, 12))
        tk.Label(
            self.upload_banner_frame,
            text="✓",
            bg="#ECF2E8",
            fg="#4E7045",
            font=("Segoe UI", 14, "bold"),
        ).grid(row=0, column=0, padx=(10, 8), pady=10, sticky="n")
        self.upload_banner_msg_lbl = tk.Label(
            self.upload_banner_frame,
            text="Choose an image to begin.",
            bg="#ECF2E8",
            fg="#4E7045",
            font=("Segoe UI", 10, "bold"),
            wraplength=300,
            justify="left",
        )
        self.upload_banner_msg_lbl.grid(row=0, column=1, padx=(0, 12), pady=10, sticky="w")

        footer = tk.Frame(left, bg=self.colors["bg"])
        footer.pack(fill="x", pady=(18, 8))
        tk.Label(
            footer,
            text="Your image is processed only for this audit session.",
            bg=self.colors["bg"],
            fg=self.colors["text_grey"],
            font=("Segoe UI", 10),
            anchor="w",
        ).pack(side="left")

        btns = tk.Frame(left, bg=self.colors["bg"])
        btns.pack(fill="x")
        tk.Button(
            btns,
            text="View capture guide",
            bg="#F5F1EB",
            fg=self.colors["text_primary"],
            activebackground="#EFE2D6",
            activeforeground=self.colors["text_primary"],
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            bd=1,
            highlightbackground=self.colors["border"],
            cursor="hand2",
            command=lambda: open_tips_modal(self),
        ).pack(side="left")

        right_panel = tk.Frame(right, bg="#F5F1EB", highlightbackground=self.colors["border"], highlightthickness=1, bd=0)
        right_panel.pack(fill="both", expand=True)
        preview = tk.Label(right_panel, bg="#E8DCC7", text="", width=26, height=12)
        preview.pack(fill="both", expand=True, padx=16, pady=16)
        self.upload_preview = preview
        self.upload_preview_photo = None

        info_box = tk.Frame(right_panel, bg="#F3E7DA", highlightbackground=self.colors["border"], highlightthickness=1, bd=0)
        info_box.pack(fill="x", padx=16, pady=(0, 16))
        tk.Label(
            info_box,
            text="FOR A RELIABLE AUDIT",
            bg="#F3E7DA",
            fg=self.colors["text_primary"],
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(anchor="w", padx=14, pady=(12, 6))
        tk.Label(
            info_box,
            text="Capture the full repeat. Use even lighting and position the camera directly above the textile. Include at least four complete motif repetitions.",
            bg="#F3E7DA",
            fg=self.colors["text_dark"],
            font=("Segoe UI", 10),
            justify="left",
            wraplength=290,
        ).pack(anchor="w", padx=14, pady=(0, 12))

        checks = [
            "Pattern is flat and in focus",
            "Minimal glare or heavy shadows",
            "Camera is parallel to the surface",
        ]
        for text in checks:
            row = tk.Frame(info_box, bg="#F3E7DA")
            row.pack(fill="x", padx=14, pady=2)
            tk.Label(row, text="✓", bg="#F3E7DA", fg="#4E7045", font=("Segoe UI", 10, "bold")).pack(side="left")
            tk.Label(row, text=text, bg="#F3E7DA", fg=self.colors["text_dark"], font=("Segoe UI", 10)).pack(side="left", padx=(6, 0))

        self.upload_preview.configure(text="Upload preview")

    def build_left_column(self, parent):
        self.card_input = self.create_card(parent, "Input", "")
        self.card_input.pack(fill="x", pady=(0, 20))

        self.upload_container = tk.Frame(self.card_input, bg="#FFFFFF")
        self.upload_container.pack(fill="x", padx=15, pady=(5, 15))

        self.upload_canvas = tk.Canvas(
            self.upload_container,
            height=155,
            bg="#FFFFFF",
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        self.upload_canvas.pack(fill="x")
        self.upload_canvas.bind(
            "<Button-1>", lambda e: self.trigger_file_upload()
        )
        self.draw_upload_slot()

        self.uploaded_state_frame = tk.Frame(
            self.upload_container, bg="#FFFFFF"
        )

        self.file_row = tk.Frame(
            self.uploaded_state_frame,
            bg="#FFFFFF",
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            bd=0,
        )
        self.file_row.pack(fill="x", pady=(0, 12))

        self.file_name_lbl = tk.Label(
            self.file_row,
            text="No file selected",
            bg="#FFFFFF",
            fg=self.colors["text_dark"],
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        )
        self.file_name_lbl.pack(
            side="left", padx=12, pady=10, fill="x", expand=True
        )

        self.re_upload_btn = tk.Button(
            self.file_row,
            text="Re-Upload",
            bg="#FFFFFF",
            fg=self.colors["maroon"],
            activebackground="#FAF2EE",
            font=("Segoe UI", 9, "bold"),
            activeforeground=self.colors["maroon"],
            relief="flat",
            bd=0,
            highlightbackground=self.colors["maroon"],
            highlightthickness=1,
            cursor="hand2",
            command=self.trigger_file_upload,
        )
        self.re_upload_btn.pack(side="right", padx=10, pady=6)

        self.banner_frame = tk.Frame(
            self.uploaded_state_frame,
            bg=self.colors["banner_bg"],
            highlightbackground=self.colors["banner_border"],
            highlightthickness=1,
            bd=0,
        )
        self.banner_frame.pack(fill="x")
        self.banner_frame.columnconfigure(1, weight=1)

        check_lbl = tk.Label(
            self.banner_frame,
            text="✓",
            bg=self.colors["banner_bg"],
            fg=self.colors["banner_border"],
            font=("Segoe UI", 14, "bold"),
        )
        check_lbl.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="n")

        banner_text = "Image ready for processing."
        self.banner_msg_lbl = tk.Label(
            self.banner_frame,
            text=banner_text,
            bg=self.colors["banner_bg"],
            fg=self.colors["banner_border"],
            font=("Segoe UI", 9, "bold"),
            wraplength=270,
            justify="left",
        )
        self.banner_msg_lbl.grid(
            row=0, column=1, padx=(0, 10), pady=10, sticky="w"
        )

        btn_row = tk.Frame(self.card_input, bg="#FFFFFF")
        btn_row.pack(fill="x", padx=15, pady=(0, 15))

        self.reconstruct_btn = self.create_flat_button(
            btn_row,
            text="Run Pipeline",
            bg=self.colors["maroon"],
            fg="#FFFFFF",
            hover_bg=self.colors["maroon_hover"],
            command=self.run_reconstruction_pipeline,
            font=("Segoe UI", 10, "bold"),
            height=38,
        )
        self.reconstruct_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.rerun_pipeline_btn = self.create_flat_button(
            btn_row,
            text="RE-RUN",
            bg=self.colors["navy_btn"],
            fg="#FFFFFF",
            hover_bg=self.colors["navy_btn_hover"],
            command=self.run_reconstruction_pipeline,
            font=("Segoe UI", 10, "bold"),
            height=38,
        )
        self.rerun_pipeline_btn.pack(side="left", fill="x", padx=(5, 0))

        tips_btn = tk.Button(
            self.card_input,
            text="Capture Tips",
            bg="#FFFFFF",
            fg=self.colors["navy_btn"],
            activebackground="#E8EAF6",
            activeforeground=self.colors["navy_btn"],
            font=("Segoe UI", 11, "bold", "italic", "underline"),
            relief="flat",
            bd=0,
            cursor="hand2",
            command=lambda: open_tips_modal(self),
        )
        tips_btn.pack(anchor="e", padx=15, pady=(0, 10))

        #CLASSIFICATION CARD
        self.card_identified_textile = self.create_card(
            parent, "Textile Classification", "", has_border=True
        )
        self.card_identified_textile.pack(fill="x", pady=(0, 12))

        identified_inner = tk.Frame(
            self.card_identified_textile, bg=self.colors["banner_bg"]
        )
        identified_inner.pack(fill="x", padx=12, pady=10)

        self.summary_classification_lbl = tk.Label(
            identified_inner,
            text="Identified Textile: --",
            bg="#FFFFFF",
            fg=self.colors["text_dark"],
            font=("Segoe UI", 11, "bold"),
            anchor="w",
            justify="left",
            wraplength=300,
        )
        self.summary_classification_lbl.pack(fill="x", pady=(0, 5))

        self.summary_confidence_lbl = tk.Label(
            identified_inner,
            text="Confidence: --",
            bg="#FFFFFF",
            fg=self.colors["text_dark"],
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        )
        self.summary_confidence_lbl.pack(fill="x")

        # DESCRIPTION CARD
        self.card_description = self.create_card(parent, "Description", "")
        self.card_description.pack(fill="x", pady=(12, 20))

        desc_inner = tk.Frame(self.card_description, bg="#FFFFFF")
        desc_inner.pack(fill="x", padx=15, pady=(8, 12))

        self.description_text = tk.Text(
            desc_inner,
            bg="#FFFFFF",
            fg=self.colors["text_dark"],
            font=("Segoe UI", 10, "italic"),
            wrap="word",
            relief="flat",
            bd=0,
            height=4,
            highlightthickness=0,
        )
        self.description_text.pack(fill="x")
        self.description_text.insert("1.0", "Upload an image to identify the textile.")
        self.description_text.configure(state="disabled")

        self.card_reference = self.create_card(
            parent, "Educational Textile Reference Guide", "📚"
        )
        self.card_reference.pack(fill="x", pady=(0, 20))
        self._build_reference_guide_cards(self.card_reference)

        # TRADITIONAL CONTEXT & CULTURAL ORIGIN
        self.card_context = tk.Frame(
            parent,
            bg="#FAF6F0",
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            bd=0,
        )
        self.card_context.pack(fill="x", pady=(12, 0))

        gold_accent_bar = tk.Frame(
            self.card_context, bg=self.colors["gold"], width=4
        )
        gold_accent_bar.pack(side="left", fill="y")

        context_inner = tk.Frame(self.card_context, bg="#FAF6F0")
        context_inner.pack(
            side="left", fill="both", expand=True, padx=15, pady=12
        )

        context_title = tk.Label(
            context_inner,
            text="Traditional Context & Cultural Origin",
            bg="#FAF6F0",
            fg=self.colors["text_primary"],
            font=("Segoe UI", 11, "bold"),
        )
        context_title.pack(anchor="w", pady=(0, 5))

        cultural_label = tk.Label(
            context_inner,
            text="Cultural Background:",
            bg="#FAF6F0",
            fg=self.colors["text_primary"],
            font=("Segoe UI", 9, "bold"),
        )
        cultural_label.pack(anchor="w", pady=(0, 2))

        self.cultural_text = tk.Text(
            context_inner,
            bg="#FAF6F0",
            fg=self.colors["text_dark"],
            font=("Segoe UI", 9, "italic"),
            wrap="word",
            relief="flat",
            bd=0,
            height=3,
            highlightthickness=0,
        )
        self.cultural_text.pack(fill="x", anchor="w", pady=(0, 8))
        self.cultural_text.insert("1.0", "Cultural context will appear after classification.")
        self.cultural_text.configure(state="disabled")

        self.source_lbl = tk.Label(
            context_inner,
            text="Source: --",
            bg="#FAF6F0",
            fg=self.colors["text_grey"],
            font=("Segoe UI", 8, "italic"),
        )
        self.source_lbl.pack(anchor="w")

    def draw_upload_slot(self, canvas=None):
        canvas = canvas or self.upload_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), canvas.winfo_reqwidth())
        height = max(canvas.winfo_height(), canvas.winfo_reqheight())
        center_x = width / 2
        center_y = height / 2
        canvas.create_rectangle(
            2, 2, width - 2, height - 2,
            outline=self.colors["border"], width=1.5, dash=(6, 4)
        )
        canvas.create_polygon(
            center_x - 20, center_y - 42,
            center_x - 6, center_y - 42,
            center_x - 2, center_y - 36,
            center_x + 20, center_y - 36,
            center_x + 20, center_y - 12,
            center_x - 20, center_y - 12,
            fill="#E8EAF6", outline="#3F51B5", width=2,
        )
        canvas.create_line(
            center_x, center_y - 17, center_x, center_y - 34,
            arrow=tk.FIRST, fill="#EF4444", width=3
        )
        canvas.create_text(
            center_x,
            center_y + 16,
            text="Upload Textile Image",
            font=("Segoe UI", 11, "bold"),
            fill=self.colors["text_primary"],
        )
        canvas.create_text(
            center_x,
            center_y + 38,
            text="PNG, JPG up to 10MB",
            font=("Segoe UI", 9),
            fill=self.colors["text_grey"],
        )

    def _build_reference_guide_cards(self, parent_card):
        container = tk.Frame(parent_card, bg="#FFFDF8")
        container.pack(fill="x", padx=12, pady=(8, 12))

        reference_items = [
            {
                "title": "Pinilian Bayabas",
                "subtitle": "Dense botanical lattice",
                "description": "A traditionally woven geometric pattern recognized for its clustered diagonal rhythm and botanical-like spacing.",
                "image_path": os.path.join(
                    os.path.dirname(__file__),
                    "dataset",
                    "pinilian_sinan_bayabas",
                    "sinan_bayabas_01.png",
                ),
            },
            {
                "title": "Pinilian Kabalyo",
                "subtitle": "Balanced repeating rhythm",
                "description": "Compact motifs with a consistent textile rhythm that highlight disciplined lattice spacing and woven repetition.",
                "image_path": os.path.join(
                    os.path.dirname(__file__),
                    "dataset",
                    "pinilian_sinan_kabalyo",
                    "sinan_kabalyo_01.png",
                ),
            },
            {
                "title": "Binakol Kinkinelleng",
                "subtitle": "Strong diagonal symmetry",
                "description": "A bold binakol textile pattern with a pronounced directional flow and a visible repeating weave continuity.",
                "image_path": os.path.join(
                    os.path.dirname(__file__),
                    "dataset",
                    "binakol_kinkinelleng",
                    "kinkinelleng_01.png",
                ),
            },
            {
                "title": "Binakol Kulalabang",
                "subtitle": "High-contrast pattern field",
                "description": "A highly textured textile field with rhythmic alternation and strong lattice regularity used as a visual benchmark.",
                "image_path": os.path.join(
                    os.path.dirname(__file__),
                    "dataset",
                    "binakol_kulalabang",
                    "kulalabang_01.png",
                ),
            },
        ]

        grid = tk.Frame(container, bg="#FFFDF8")
        grid.pack(fill="x")
        for index, item in enumerate(reference_items):
            col = index % 2
            row = index // 2
            item_frame = tk.Frame(grid, bg="#F8F2EA", highlightbackground=self.colors["border"], highlightthickness=1, bd=0)
            item_frame.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
            item_frame.grid_columnconfigure(0, weight=1)
            item_frame.grid_rowconfigure(1, weight=1)

            thumb = self._render_reference_thumbnail(item["image_path"], (110, 110))
            image_label = tk.Label(item_frame, image=thumb, bg="#F8F2EA")
            image_label.image = thumb
            image_label.pack(fill="x", padx=8, pady=(8, 4))

            title = tk.Label(item_frame, text=item["title"], bg="#F8F2EA", fg=self.colors["text_primary"], font=("Segoe UI", 9, "bold"), anchor="w")
            title.pack(fill="x", padx=8)

            subtitle = tk.Label(item_frame, text=item["subtitle"], bg="#F8F2EA", fg=self.colors["text_grey"], font=("Segoe UI", 8), anchor="w")
            subtitle.pack(fill="x", padx=8, pady=(0, 4))

            btn = tk.Button(
                item_frame,
                text="View Details",
                bg=self.colors["navy_btn"],
                fg="#FFFFFF",
                activebackground=self.colors["navy_btn_hover"],
                activeforeground="#FFFFFF",
                font=("Segoe UI", 8, "bold"),
                relief="flat",
                bd=0,
                padx=12,
                pady=6,
                cursor="hand2",
                command=lambda item=item: self.show_reference_detail(item),
            )
            btn.pack(fill="x", padx=8, pady=(0, 8))

    def _render_reference_thumbnail(self, image_path, size):
        try:
            source = Image.open(image_path)
            source = source.convert("RGB")
            source = ImageOps.fit(source, size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(source)
        except Exception:
            placeholder = Image.new("RGB", size, color=(236, 227, 216))
            return ImageTk.PhotoImage(placeholder)

    def show_reference_detail(self, item):
        if not hasattr(self, "reference_grid_page"):
            self.show_reference_guide()

        self.reference_grid_page.pack_forget()
        for child in self.reference_detail_page.winfo_children():
            child.destroy()

        header = tk.Frame(self.reference_detail_page, bg=self.colors["bg"])
        header.pack(fill="x", pady=(0, 14))
        tk.Button(
            header,
            text="← Back to reference guide",
            bg=self.colors["bg"],
            fg=self.colors["maroon"],
            activebackground=self.colors["bg"],
            activeforeground=self.colors["maroon_hover"],
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            cursor="hand2",
            command=self.show_reference_grid,
        ).pack(side="left")

        detail_card = tk.Frame(
            self.reference_detail_page,
            bg="#FFFDF8",
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )
        detail_card.pack(fill="both", expand=True)

        try:
            with Image.open(item["image_path"]) as source:
                image = ImageOps.contain(
                    source.convert("RGB"), (820, 470), Image.Resampling.LANCZOS
                )
        except (OSError, ValueError):
            image = Image.new("RGB", (820, 360), color=(236, 227, 216))
        photo = ImageTk.PhotoImage(image)
        image_label = tk.Label(detail_card, image=photo, bg="#FFFDF8")
        image_label.image = photo
        image_label.pack(fill="x", padx=24, pady=(24, 16))

        tk.Label(
            detail_card,
            text=item["title"],
            bg="#FFFDF8",
            fg=self.colors["text_primary"],
            font=("Segoe UI", 22, "bold"),
            anchor="w",
        ).pack(fill="x", padx=24)
        tk.Label(
            detail_card,
            text=item["subtitle"],
            bg="#FFFDF8",
            fg=self.colors["maroon"],
            font=("Segoe UI", 12, "bold"),
            anchor="w",
        ).pack(fill="x", padx=24, pady=(4, 12))
        tk.Label(
            detail_card,
            text=item["description"],
            bg="#FFFDF8",
            fg=self.colors["text_dark"],
            font=("Segoe UI", 11),
            anchor="w",
            justify="left",
            wraplength=900,
        ).pack(fill="x", padx=24, pady=(0, 24))
        self.reference_detail_page.pack(fill="both", expand=True)
        self.show_page("reference")
        self.canvas.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def show_reference_guide(self):
        if not hasattr(self, "reference_grid_page"):
            heading = tk.Frame(self.reference_page, bg=self.colors["bg"])
            heading.pack(fill="x", padx=28, pady=(24, 18))
            tk.Label(
                heading,
                text="EDUCATIONAL ARCHIVE / TEXTILE FIELD GUIDE",
                bg=self.colors["bg"],
                fg=self.colors["maroon"],
                font=("Segoe UI", 9, "bold"),
            ).pack(anchor="w")
            tk.Label(
                heading,
                text="Educational Textile Reference Guide",
                bg=self.colors["bg"],
                fg=self.colors["text_primary"],
                font=("Segoe UI", 25, "bold"),
            ).pack(anchor="w", pady=(5, 4))
            tk.Label(
                heading,
                text="Select a textile to view a larger sample and its pattern notes.",
                bg=self.colors["bg"],
                fg=self.colors["text_grey"],
                font=("Segoe UI", 11),
            ).pack(anchor="w")

            self.reference_grid_page = tk.Frame(
                self.reference_page, bg=self.colors["bg"]
            )
            self.reference_grid_page.pack(fill="both", expand=True, padx=22, pady=8)
            self.reference_detail_page = tk.Frame(
                self.reference_page, bg=self.colors["bg"]
            )

            items = [
                {
                    "title": "Pinilian Bayabas",
                    "subtitle": "Dense botanical lattice",
                    "description": "A traditionally woven geometric pattern recognized for its clustered diagonal rhythm and botanical-like spacing.",
                    "image_path": os.path.join(os.path.dirname(__file__), "dataset", "pinilian_sinan_bayabas", "sinan_bayabas_01.png"),
                },
                {
                    "title": "Pinilian Kabalyo",
                    "subtitle": "Balanced repeating rhythm",
                    "description": "Compact motifs with a consistent textile rhythm that highlight disciplined lattice spacing and woven repetition.",
                    "image_path": os.path.join(os.path.dirname(__file__), "dataset", "pinilian_sinan_kabalyo", "sinan_kabalyo_01.png"),
                },
                {
                    "title": "Binakol Kinkinelleng",
                    "subtitle": "Strong diagonal symmetry",
                    "description": "A bold binakol textile pattern with a pronounced directional flow and a visible repeating weave continuity.",
                    "image_path": os.path.join(os.path.dirname(__file__), "dataset", "binakol_kinkinelleng", "kinkinelleng_01.png"),
                },
                {
                    "title": "Binakol Kulalabang",
                    "subtitle": "High-contrast pattern field",
                    "description": "A highly textured textile field with rhythmic alternation and strong lattice regularity used as a visual benchmark.",
                    "image_path": os.path.join(os.path.dirname(__file__), "dataset", "binakol_kulalabang", "kulalabang_01.png"),
                },
            ]

            for index, item in enumerate(items):
                row, column = divmod(index, 2)
                card = tk.Frame(
                    self.reference_grid_page,
                    bg="#FFFDF8",
                    highlightbackground=self.colors["border"],
                    highlightthickness=1,
                    cursor="hand2",
                )
                card.grid(row=row, column=column, padx=10, pady=10, sticky="nsew")
                self.reference_grid_page.grid_columnconfigure(column, weight=1)
                image = self._render_reference_thumbnail(
                    item["image_path"], (320, 220)
                )
                image_label = tk.Label(card, image=image, bg="#FFFDF8", cursor="hand2")
                image_label.image = image
                image_label.pack(fill="x", padx=14, pady=(14, 10))
                title = tk.Label(
                    card,
                    text=item["title"],
                    bg="#FFFDF8",
                    fg=self.colors["text_primary"],
                    font=("Segoe UI", 13, "bold"),
                    anchor="w",
                    cursor="hand2",
                )
                title.pack(fill="x", padx=14)
                tk.Label(
                    card,
                    text=item["subtitle"],
                    bg="#FFFDF8",
                    fg=self.colors["text_grey"],
                    font=("Segoe UI", 10),
                    anchor="w",
                ).pack(fill="x", padx=14, pady=(3, 10))
                for widget in (card, image_label, title):
                    widget.bind(
                        "<Button-1>",
                        lambda event, selected=item: self.show_reference_detail(selected),
                    )

        self.show_reference_grid()

    def show_reference_grid(self):
        if not hasattr(self, "reference_grid_page"):
            self.show_reference_guide()
            return
        self.reference_detail_page.pack_forget()
        self.reference_grid_page.pack(
            fill="both", expand=True, padx=22, pady=8
        )
        self.show_page("reference")

    def _bind_button_hover(self, button, normal_bg, hover_bg):
        button.bind(
            "<Enter>",
            lambda event: button.configure(
                bg=hover_bg
                if button.cget("state") == "normal"
                else normal_bg
            ),
        )
        button.bind(
            "<Leave>", lambda event: button.configure(bg=normal_bg)
        )

    def build_right_column(self, parent):
        self.card_visuals = self.create_card(
            parent, "Visual Results", ""
        )
        self.card_visuals.pack(fill="x", pady=(0, 20))

        visuals_inner = tk.Frame(self.card_visuals, bg="#FFFFFF")
        visuals_inner.pack(fill="x", padx=15, pady=(5, 15))
        visuals_inner.columnconfigure(0, weight=1)
        visuals_inner.columnconfigure(1, weight=1)
        visuals_inner.columnconfigure(2, weight=1)

        self.panel_orig = self.create_image_panel(
            visuals_inner, 0, "ORIGINAL IMAGE"
        )
        self.panel_track = self.create_image_panel(
            visuals_inner, 1, "DETECTED ERROR HEATMAP"
        )
        self.panel_recon = self.create_image_panel(
            visuals_inner, 2, "RECONSTRUCTED TEXTILE"
        )

        # Buttons row below panels
        buttons_inner = tk.Frame(self.card_visuals, bg="#FFFFFF")
        buttons_inner.pack(fill="x", padx=15, pady=(0, 15))
        buttons_inner.columnconfigure(0, weight=1)
        buttons_inner.columnconfigure(1, weight=1)
        buttons_inner.columnconfigure(2, weight=1)

        # Heatmap + Reconstruction export buttons (centered)
        center_btns = tk.Frame(buttons_inner, bg="#FFFFFF")
        center_btns.grid(row=0, column=0, columnspan=4, pady=(10, 0))

        self.export_heatmap_btn = self.create_flat_button(
            center_btns,
            text="Export Heatmap",
            bg=self.colors["navy_btn"],
            fg="#EFBF04",
            hover_bg=self.colors["navy_btn_hover"],
            command=self.export_heatmap,
            font=("Segoe UI", 12, "bold"),
            height=20,
        )
        self.export_heatmap_btn.pack(side="left", padx=6)
        self.export_heatmap_btn.configure(state="disabled")
        # Slightly enlarge export button visuals
        self.export_heatmap_btn.configure(font=("Segoe UI", 10, "bold"), padx=12, pady=8)

        self.export_btn = self.create_flat_button(
            center_btns,
            text="Export Reconstruction",
            bg=self.colors["navy_btn"],
            fg="#EFBF04",
            hover_bg=self.colors["navy_btn_hover"],
            command=self.export_restored_photo,
            font=("Segoe UI", 12, "bold"),
            height=20,
        )
        self.export_btn.pack(side="left", padx=6)
        self.export_btn.configure(state="disabled")
        # Slightly enlarge export button visuals
        self.export_btn.configure(font=("Segoe UI", 10, "bold"), padx=12, pady=8)

        metrics_container = tk.Frame(parent, bg=self.colors["bg"])
        metrics_container.pack(fill="x", pady=(0, 20))
        metrics_container.columnconfigure(0, weight=1)
        metrics_container.columnconfigure(1, weight=1)

        self.card_confidence = self.create_card(
            metrics_container,
            "SYSTEM CONFIDENCE METRICS",
            "",
            has_border=True,
        )
        self.card_confidence.grid(
            row=0, column=0, sticky="nsew", padx=(0, 10)
        )

        conf_inner = tk.Frame(self.card_confidence, bg="#FFFFFF")
        conf_inner.pack(fill="both", expand=True, padx=15, pady=(5, 15))

        self.group_lbl = tk.Label(
            conf_inner,
            text="Wallpaper Group Identified: --",
            bg="#FFFFFF",
            fg=self.colors["text_primary"],
            font=("Segoe UI", 11, "italic", "bold", "underline"),
            cursor="hand2",
        )
        self.group_lbl.pack(anchor="w", pady=(0, 10))
        self.group_lbl.bind(
            "<Button-1>", lambda e: self.show_wallpaper_guide_window()
        )
        self.guide_window = None

        self.progress_canvas = tk.Canvas(
            conf_inner,
            height=10,
            bg="#E3DCD6",
            bd=0,
            highlightthickness=0,
        )
        self.progress_canvas.pack(fill="x", pady=(0, 15))
        self.draw_progress_bar(0)

        stats_row = tk.Frame(conf_inner, bg="#FFFFFF")
        stats_row.pack(fill="x")
        stats_row.columnconfigure(0, weight=1)
        stats_row.columnconfigure(1, weight=1)
        stats_row.columnconfigure(2, weight=1)

        self.box_acc = self.create_stat_box(stats_row, 0, "Accuracy")
        self.box_rec = self.create_stat_box(stats_row, 1, "a1 Lattice")
        self.box_f1 = self.create_stat_box(stats_row, 2, "a2 Lattice")

        self.card_sfs = self.create_card(
            metrics_container,
            "SYMMETRY FIDELITY SCORE (%)",
            "",
            has_border=True,
        )
        self.card_sfs.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        sfs_inner = tk.Frame(self.card_sfs, bg="#FFFFFF")
        sfs_inner.pack(fill="both", expand=True, padx=15, pady=(5, 15))

        sfs_header_row = tk.Frame(sfs_inner, bg="#FFFFFF")
        sfs_header_row.pack(fill="x", pady=(0, 10))

        self.sfs_num_lbl = tk.Label(
            sfs_header_row,
            text="--",
            bg="#FFFFFF",
            fg=self.colors["text_grey"],
            font=("Segoe UI", 28, "bold"),
        )
        self.sfs_num_lbl.pack(side="left")

        threshold_lbl = tk.Label(
            sfs_header_row,
            text="(Threshold: 95%)",
            bg="#FFFFFF",
            fg=self.colors["text_grey"],
            font=("Segoe UI", 10),
        )
        threshold_lbl.pack(side="left", padx=(10, 0), pady=(15, 0))

        grid_frame = tk.Frame(sfs_inner, bg="#FFFFFF")
        grid_frame.pack(fill="x", pady=(0, 15))
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)

        self.grid_acc = self.create_grid_cell(
            grid_frame, 0, 0, "Accuracy", "--"
        )
        self.grid_rec = self.create_grid_cell(
            grid_frame, 0, 1, "Recall", "--"
        )
        self.grid_fpr = self.create_grid_cell(
            grid_frame, 1, 0, "False Positive Rate", "--"
        )
        self.grid_iou = self.create_grid_cell(
            grid_frame, 1, 1, "Intersection over Union", "--"
        )
        self.grid_ssim = self.create_grid_cell(
            grid_frame, 2, 0, "Structural Similarity", "--"
        )

        # Explainability / Transparency Card
        self.card_explain = self.create_card(
            parent, "Interpretable Output", "", has_border=True
        )
        self.card_explain.pack(fill="x", pady=(0, 12))

        exp_inner = tk.Frame(self.card_explain, bg="#FFFFFF")
        exp_inner.pack(fill="x", padx=15, pady=(8, 12))

        self.classification_lbl = tk.Label(
            exp_inner,
            text="Identified Textile: --",
            bg="#FFFFFF",
            fg=self.colors["text_primary"],
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        )
        self.classification_lbl.pack(fill="x", pady=(0, 4))

        self.confidence_lbl = tk.Label(
            exp_inner,
            text="Confidence: --",
            bg="#FFFFFF",
            fg=self.colors["text_grey"],
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.confidence_lbl.pack(fill="x", pady=(0, 8))

        self.explain_text = tk.Text(
            exp_inner,
            bg="#FFFFFF",
            fg=self.colors["text_dark"],
            font=("Segoe UI", 10, "italic"),
            wrap="word",
            relief="flat",
            bd=0,
            height=12,
            highlightthickness=0,
        )
        self.explain_text.pack(fill="both", expand=True)
        self.explain_text.insert(
            "1.0",
            "After a process, an interpretable output explanation will be displayed here.",
        )
        self.explain_text.configure(state="disabled")

        # PDF Export button placed just below the Output Details card
        export_pdf_container = tk.Frame(parent, bg="#FFFFFF")
        export_pdf_container.pack(fill="x", padx=15, pady=(8, 12))

        self.export_pdf_btn = self.create_flat_button(
            export_pdf_container,
            text="Export PDF Audit Report",
            bg=self.colors["navy_btn"],
            fg=self.colors["text_primary"],
            hover_bg=self.colors["navy_btn_hover"],
            command=self.export_to_pdf,
            font=("Segoe UI", 9, "bold"),
            height=20,
        )
        self.export_pdf_btn.pack(anchor="center")
        self.export_pdf_btn.configure(state="disabled")
        
        self.export_pdf_btn.configure(font=("Segoe UI", 10, "bold"), padx=12, pady=8)

    def create_card(self, parent, title_text, icon_prefix="", has_border=True):
        card = tk.Frame(parent, bg="#FFFFFF")
        if has_border:
            card.configure(
                highlightbackground=self.colors["border"],
                highlightthickness=1,
                bd=0,
            )

        header_container = tk.Frame(card, bg="#FFFFFF")
        header_container.pack(fill="x")
        card.header_container = header_container

        header_txt = (
            f"{icon_prefix} {title_text}" if icon_prefix else title_text
        )
        header_lbl = tk.Label(
            header_container,
            text=header_txt,
            bg="#FFFFFF",
            fg=self.colors["text_primary"],
            font=("Segoe UI", 12, "bold"),
            anchor="w",
        )
        header_lbl.pack(side="left", padx=15, pady=(15, 8))
        return card

    def create_image_panel(self, parent, col, label_text):
        frame = tk.Frame(parent, bg="#FFFFFF")
        frame.grid(row=0, column=col, padx=8, sticky="nsew")

        lbl = tk.Label(
            frame,
            text=label_text,
            bg="#FFFFFF",
            fg=self.colors["text_grey"],
            font=("Segoe UI", 9, "bold"),
        )
        lbl.pack(anchor="center", pady=(5, 6))

        slot = tk.Frame(
            frame,
            bg="#F5F3EF",
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            bd=0,
            width=280,
            height=280,
        )
        slot.pack(fill="both", expand=True)
        slot.pack_propagate(False)

        placeholder = tk.Label(
            slot,
            text="Awaiting processing" if col > 0 else "No image uploaded",
            bg="#F5F3EF",
            fg=self.colors["text_grey"],
            font=("Segoe UI", 10),
        )
        placeholder.pack(expand=True)
        slot.placeholder = placeholder
        return slot

    def create_stat_box(self, parent, col, name):
        box = tk.Frame(
            parent,
            bg="#FAF6F0",
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            bd=0,
        )
        box.grid(row=0, column=col, padx=5, pady=5, sticky="nsew")

        name_lbl = tk.Label(
            box,
            text=name,
            bg="#FAF6F0",
            fg=self.colors["text_grey"],
            font=("Segoe UI", 9),
        )
        name_lbl.pack(anchor="center", pady=(8, 2))

        num_lbl = tk.Label(
            box,
            text="--",
            bg="#FAF6F0",
            fg=self.colors["text_grey"],
            font=("Segoe UI", 14, "bold"),
        )
        num_lbl.pack(anchor="center", pady=(0, 8))
        box.num_lbl = num_lbl
        return box

    def create_grid_cell(self, parent, row, col, label_text, default_val):
        cell = tk.Frame(
            parent,
            bg="#FAF6F0",
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            bd=0,
        )
        cell.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")

        lbl = tk.Label(
            cell,
            text=f"{label_text}: ",
            bg="#FAF6F0",
            fg=self.colors["text_grey"],
            font=("Segoe UI", 9),
        )
        lbl.pack(side="left", padx=(10, 2), pady=6)

        val_lbl = tk.Label(
            cell,
            text=default_val,
            bg="#FAF6F0",
            fg=self.colors["text_dark"],
            font=("Segoe UI", 10, "bold"),
        )
        val_lbl.pack(side="left")
        cell.val_lbl = val_lbl
        return cell

    def draw_progress_bar(self, percentage):
        self.progress_canvas.delete("all")
        self.progress_canvas.update()
        w = self.progress_canvas.winfo_width()
        if w <= 1:
            w = 340
        self.create_rounded_rect(self.progress_canvas, 0, 0, w, 10, fill="#E3DCD6")
        if percentage > 0:
            fill_w = max(10, int((percentage / 100.0) * w))
            bar_color = (
                self.colors["emerald"]
                if percentage >= 95
                else self.colors["maroon"]
            )
            self.create_rounded_rect(
                self.progress_canvas, 0, 0, fill_w, 10, fill=bar_color
            )

    def create_rounded_rect(self, canvas, x1, y1, x2, y2, fill):
        r = y2 - y1
        canvas.create_oval(x1, y1, x1 + r, y2, fill=fill, outline="")
        canvas.create_oval(x2 - r, y1, x2, y2, fill=fill, outline="")
        canvas.create_rectangle(x1 + r / 2, y1, x2 - r / 2, y2, fill=fill, outline="")

    def create_flat_button(
        self, parent, text, bg, fg, hover_bg, command, font, height
    ):
        btn = tk.Button(
            parent,
            text=text,
            bg=bg,
            fg="#FFFFFF",
            font=font,
            relief="flat",
            bd=0,
            activebackground=hover_bg,
            activeforeground="#FFFFFF",
            command=command,
            cursor="hand2",
        )
        btn.configure(pady=6)
        self._bind_button_hover(btn, bg, hover_bg)
        return btn

    def create_flat_border_button(
        self, parent, text, bg, fg, hover_bg, border_color, command
    ):
        btn = tk.Button(
            parent,
            text=text,
            bg=bg,
            fg="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            bd=0,
            highlightbackground=border_color,
            highlightthickness=1,
            activebackground=hover_bg,
            activeforeground="#FFFFFF",
            command=command,
            cursor="hand2",
        )
        btn.configure(pady=6)
        self._bind_button_hover(btn, bg, hover_bg)
        return btn

    def _set_text_widget(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _apply_classification_result(self, result):
        self.classification_result = result
        self.active_subclass_id = None

        if result:
            confidence = result["confidence"]
            supported = result.get("is_supported", True)
            category = result.get("main_category", "Unknown")
            subclass = result.get("subclass", "Unknown")
            confidence_text = f"Confidence: {confidence:.2f}%"
            if supported:
                classification_text = f"Identified Textile: {category} ({subclass})"
                subclass_id = f"{category.lower()}_{subclass.lower().replace(' ', '_')}"
                metadata = TEXTILE_METADATA.get(subclass_id)
                if metadata:
                    self.active_subclass_id = subclass_id
                    description = metadata["description"]
                    cultural_background = metadata["cultural_background"]
                    source = metadata["source"]
                else:
                    description = "No description is available for this textile."
                    cultural_background = "No cultural background is available for this textile."
                    source = "N/A"
            else:
                classification_text = (
                    "Unsupported or uncertain input: not confidently "
                    "recognized as Pinilian or Binakol."
                )
                confidence_text += (
                    f" (minimum supported: "
                    f"{result.get('support_threshold', 80.0):.0f}%)"
                )
                description = "The image was not confidently identified as a supported textile."
                cultural_background = "Cultural context is unavailable for this prediction."
                source = "N/A"

            classification_details = (
                f"Classification Result:\n{classification_text}\n"
                f"{confidence_text}\n"
            )
        else:
            category = "Classification unavailable"
            subclass = ""
            confidence = None
            supported = False
            classification_text = f"Category: {category}"
            confidence_text = "Confidence: --"
            description = "Classification metadata is unavailable."
            cultural_background = "Cultural context is unavailable."
            source = "N/A"
            classification_details = "Classification Result:\nClassification unavailable."

        self.classification_lbl.configure(text=classification_text)
        self.confidence_lbl.configure(text=confidence_text)
        self.summary_classification_lbl.configure(text=classification_text)
        self.summary_confidence_lbl.configure(text=confidence_text)
        self._set_text_widget(self.description_text, description)
        self._set_text_widget(self.cultural_text, cultural_background)
        self.source_lbl.configure(text=f"Source: {source}")
        self._set_text_widget(self.explain_text, classification_details)

    def trigger_file_upload(self):
        file_path = filedialog.askopenfilename(
            title="Upload Textile Image for Auditing",
            filetypes=[("Textile Images", "*.png *.jpg *.jpeg")],
        )
        if not file_path:
            return

        try:
            with Image.open(file_path) as source_image:
                pil_img = source_image.convert("RGB")
            original_bgr = cv2.imread(file_path)
            if original_bgr is None:
                raise ValueError("OpenCV could not decode the selected image.")
            preview = ImageOps.contain(
                pil_img, (440, 360), Image.Resampling.LANCZOS
            )
        except Exception as e:
            messagebox.showerror(
                "Error Reading Image", f"Failed to open image:\n{str(e)}"
            )
            return

        self.uploaded_file_path = file_path
        self.original_bgr = original_bgr
        self.original_image_hires = pil_img.copy()
        self.original_image_pil = ImageOps.fit(
            pil_img, (270, 270), Image.Resampling.LANCZOS
        )
        self._apply_classification_result(None)

        file_name = os.path.basename(file_path)
        short_name = file_name if len(file_name) < 22 else f"{file_name[:18]}..."
        self.upload_file_name_lbl.configure(text=short_name)
        self.upload_page_canvas.pack_forget()
        self.upload_reconstruct_btn.configure(state="normal")
        self.upload_banner_msg_lbl.configure(
            text="Image loaded and ready for the audit pipeline."
        )
        self.upload_preview_photo = ImageTk.PhotoImage(preview)
        self.upload_preview.configure(
            image=self.upload_preview_photo,
            text="",
            width=0,
            height=0,
        )

        self.file_name_lbl.configure(text=short_name)
        self.upload_canvas.pack_forget()
        self.uploaded_state_frame.pack(fill="x")
        self.display_pil_in_panel(
            self.panel_orig, self.original_image_pil, "orig"
        )

        try:
            self._apply_classification_result(
                self.classifier.predict(file_path)
            )
        except Exception as e:
            self._apply_classification_result(None)
            messagebox.showerror(
                "Classification Error", f"Failed to classify image:\n{str(e)}"
            )

    def display_pil_in_panel(self, panel_widget, pil_image, cache_key):
        panel_widget.placeholder.pack_forget()
        for child in panel_widget.winfo_children():
            if child != panel_widget.placeholder:
                child.destroy()

        img_tk = ImageTk.PhotoImage(pil_image)
        self.img_tk_cache[cache_key] = img_tk

        lbl = tk.Label(panel_widget, image=img_tk, bg="#F5F3EF")
        lbl.pack(fill="both", expand=True)

    def run_reconstruction_pipeline(self):
        if self.original_bgr is None or self.uploaded_file_path is None:
            messagebox.showwarning(
                "Image Required", "Please upload a damaged textile image first!"
            )
            return

        self.upload_reconstruct_btn.configure(
            state="disabled", text="Processing audit..."
        )
        self.reconstruct_btn.configure(state="disabled")
        image_path = self.uploaded_file_path
        input_bgr = self.original_bgr.copy()

        def worker():
            start_time = time.time()
            orig_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
            orig_hires = Image.fromarray(orig_rgb)
            orig_pil = orig_hires.resize((270, 270), Image.Resampling.LANCZOS)

            I_bgr_resized = cv2.resize(input_bgr, (256, 256))
            I_gray_eval = cv2.cvtColor(I_bgr_resized, cv2.COLOR_BGR2GRAY)

            R_xx, a1, a2, theta_dom, detected_group = run_algorithm_i_autocorrelation(
                I_gray_eval
            )
            M_mask, diff_map = run_algorithm_ii_defect_isolation(
                I_bgr_resized, a1, a2
            )

            defect_pixel_count = int((M_mask == 255).sum())
            no_defect_detected = defect_pixel_count == 0

            I_final_bgr = run_algorithm_iii_template_alignment(
                I_bgr_resized, M_mask, a1, a2, symmetry_group=detected_group
            )

            runtime_ms = (time.time() - start_time) * 1000
            I_final_gray = cv2.cvtColor(I_final_bgr, cv2.COLOR_BGR2GRAY)

            res = calculate_restoration_metrics(
                I_gray_eval, I_final_gray, M_mask, None
            )
            (
                ssim_val,
                accuracy_val,
                recall_val,
                fpr_val,
                iou_val,
                sfs_score,
            ) = res if res else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

            color_heatmap = cv2.applyColorMap(
                cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX).astype(
                    np.uint8
                ),
                cv2.COLORMAP_JET,
            )
            color_heatmap[M_mask == 255] = [255, 0, 0]

            heatmap_rgb = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
            final_rgb = cv2.cvtColor(I_final_bgr, cv2.COLOR_BGR2RGB)

            track_pil = Image.fromarray(heatmap_rgb).resize(
                (270, 270), Image.Resampling.LANCZOS
            )
            recon_pil = Image.fromarray(final_rgb).resize(
                (270, 270), Image.Resampling.LANCZOS
            )
            recon_hires = Image.fromarray(final_rgb)
            heatmap_hires = Image.fromarray(heatmap_rgb)
            try:
                classification_result = self.classifier.predict(image_path)
            except Exception:
                classification_result = None

            self.after(
                0,
                lambda: self.update_pipeline_ui(
                    orig_pil,
                    orig_hires,
                    track_pil,
                    recon_pil,
                    recon_hires,
                    heatmap_hires,
                    sfs_score,
                    ssim_val,
                    accuracy_val,
                    recall_val,
                    fpr_val,
                    iou_val,
                    a1,
                    a2,
                    theta_dom,
                    detected_group,
                    runtime_ms,
                    classification_result,
                    no_defect_detected=no_defect_detected,
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    def update_pipeline_ui(
        self,
        orig_pil,
        orig_hires,
        track_pil,
        recon_pil,
        recon_hires,
        heatmap_hires,
        sfs_score,
        ssim_val,
        accuracy_val,
        recall_val,
        fpr_val,
        iou_val,
        a1,
        a2,
        theta_dom,
        detected_group,
        runtime_ms,
        classification_result=None,
        no_defect_detected=False,
    ):
        self.upload_reconstruct_btn.configure(
            state="normal", text="Run audit pipeline"
        )
        self.reconstruct_btn.configure(state="normal")

        # 1. Store images on instance
        self.original_image_pil = orig_pil
        self.original_image_hires = orig_hires
        self.tracking_pattern_pil = track_pil
        self.heatmap_hires = heatmap_hires
        self.reconstructed_image_pil = recon_pil
        self.reconstructed_image_hires = recon_hires

        # Display in UI panels
        self.display_pil_in_panel(self.panel_track, track_pil, "track")
        self.display_pil_in_panel(self.panel_recon, recon_pil, "recon")

        # 2. Update Wallpaper Group & SFS UI Labels
        group_str = str(detected_group) if detected_group else "N/A"
        if detected_group:
            self.group_lbl.configure(text=f"Wallpaper Group: {group_str}")

        if no_defect_detected:
            sfs_str = "N/A"
            self.sfs_num_lbl.configure(
                text=sfs_str, fg=self.colors["text_grey"]
            )
            self.draw_progress_bar(0)
        else:
            sfs_str = f"{sfs_score:.2f}%"
            self.draw_progress_bar(sfs_score)
            self.sfs_num_lbl.configure(
                text=sfs_str,
                fg=(
                    self.colors["maroon"]
                    if sfs_score < 95.0
                    else self.colors["text_primary"]
                ),
            )

        # 3. Update Metric Grid Widgets
        self.box_acc.num_lbl.configure(
            text=f"{accuracy_val:.1f}%", fg=self.colors["text_primary"]
        )
        self.box_rec.num_lbl.configure(
            text=f"{a1:.1f}", fg=self.colors["text_primary"]
        )
        self.box_f1.num_lbl.configure(
            text=f"{a2:.1f}", fg=self.colors["text_primary"]
        )

        self.grid_acc.val_lbl.configure(text=f"{accuracy_val:.1f}%")
        self.grid_rec.val_lbl.configure(text=f"{recall_val:.1f}%")
        self.grid_fpr.val_lbl.configure(text=f"{fpr_val:.1f}%")
        self.grid_iou.val_lbl.configure(text=f"{iou_val:.1f}%")
        self.grid_ssim.val_lbl.configure(text=f"{ssim_val:.1f}%")

        # Enable export buttons safely
        if hasattr(self, "export_btn"):
            self.export_btn.configure(state="normal")
        if hasattr(self, "export_heatmap_btn"):
            self.export_heatmap_btn.configure(state="normal")
        if hasattr(self, "export_pdf_btn"):
            self.export_pdf_btn.configure(state="normal")
        if classification_result:
            confidence = classification_result["confidence"]
            classification_supported = classification_result.get(
                "is_supported", True
            )
            if classification_supported:
                category = classification_result["main_category"]
                subclass = classification_result["subclass"]
                classification_text = (
                    f"Identified Textile: {category} ({subclass})"
                )
                confidence_text = f"Confidence: {confidence:.2f}%"
                classification_details = (
                    f"Classification Result:\n{classification_text}\n"
                    f"{confidence_text}\n\n"
                )
            else:
                category = "Unsupported or uncertain input"
                subclass = "Not recognized as Pinilian or Binakol"
                threshold = classification_result.get(
                    "support_threshold", 80.0
                )
                classification_text = (
                    "Unsupported or uncertain input: not confidently "
                    "recognized as Pinilian or Binakol."
                )
                confidence_text = (
                    f"Confidence: {confidence:.2f}% "
                    f"(minimum supported: {threshold:.0f}%)"
                )
                classification_details = (
                    f"Classification Result:\n{classification_text}\n"
                    f"{confidence_text}\n\n"
                )
            self.classification_lbl.configure(text=classification_text)
            self.confidence_lbl.configure(text=confidence_text)
        else:
            category = "Classification unavailable"
            subclass = ""
            confidence = None
            classification_supported = False
            classification_text = f"Category: {category}"
            confidence_text = "Confidence: --"
            self.classification_lbl.configure(text=classification_text)
            self.confidence_lbl.configure(text=confidence_text)
            classification_details = (
                "Classification Result:\nClassification unavailable.\n\n"
            )

        self._apply_classification_result(classification_result)

        # 4. Generate Dynamic Explainability Text
        if no_defect_detected:
            expl = classification_details + (
                "No defect detected. The system analyzed the repeating pattern and local pixel "
                "differences and did not find damaged areas.\n\n"
                "What was checked: pattern lattice (FFT autocorrelation), local deviation map, "
                "and adaptive thresholds.\n\nResult: No repair was applied."
            )
        else:
            expl = classification_details + (
                f"How this result was produced:\n\n"
                f"1) Pattern Analysis: the engine detected repeating tile sizes A1 (Horizontal) = {a1:.1f}, "
                f"A2 (Vertical) = {a2:.1f} and symmetry '{group_str}'.\n\n"
                f"2) Defect detection: Damaged fabric is localized by combining texture chaos (highlighted in Red/Yellow) "
                f"and physical holes/dark spots (appearing as Dark Blue). Both conditions are merged into the final repair mask.\n\n"
                f"3) Repair: Based on Exemplar healthy patches matching the textile's symmetry are copied over damaged zones "
                f"with feathered edge-blending to prevent halos, using a Telea inpainting fallback for any leftover gaps.\n\n"
                f"Key outputs:\n"
                f" - SSIM (clean areas): {ssim_val:.2f}%\n"
                f" - Symmetry Fidelity Score (SFS): {sfs_str}\n"
                f" - Accuracy (repair + clean preservation): {accuracy_val:.2f}%\n"
                f" - Recall (defect coverage): {recall_val:.2f}%\n"
                f" - False Positive Rate (unintended fabric alterations): {fpr_val:.2f}%\n"
                f" - IoU (mask alignment): {iou_val:.2f}%\n\n"
                f"Processing time: {runtime_ms:.0f} ms\n\n"
                f"Simple Interpretation: Higher SFS/SSIM means the repaired image preserves the original fabric appearance; "
                f"low FPR means healthy fabric was not changed.\n"
            )

        try:
            self.explain_text.configure(state="normal")
            self.explain_text.delete("1.0", tk.END)
            self.explain_text.insert("1.0", expl)
            self.explain_text.configure(state="disabled")
        except Exception:
            pass

        # 5. Save explicit instance attributes for fallback lookup
        self.sfs_val = sfs_str
        self.group_val = group_str
        self.details_val = expl

        # Read cultural description
        cultural_desc = ""
        try:
            cultural_desc = self.cultural_text.get("1.0", "end-1c").strip()
        except Exception:
            pass

        # 6. Store structured pipeline_results dictionary directly on self
        self.pipeline_results = {
            "original_img": orig_hires,
            "heatmap_img": heatmap_hires,
            "restored_img": recon_hires,
            "sfs_score": sfs_str,
            "wallpaper_group": group_str,
            "a1_vector": f"{a1:.1f}",
            "a2_vector": f"{a2:.1f}",
            "accuracy": f"{accuracy_val:.1f}%",
            "recall": f"{recall_val:.1f}%",
            "fpr": f"{fpr_val:.1f}%",
            "iou": f"{iou_val:.1f}%",
            "ssim": f"{ssim_val:.1f}%",
            "classification_category": category,
            "classification_subclass": subclass,
            "classification_confidence": (
                f"{confidence:.2f}%" if confidence is not None else "N/A"
            ),
            "classification_supported": classification_supported,
            "category": category,
            "subclass": subclass,
            "subclass_id": self.active_subclass_id,
            "cultural_description": cultural_desc,
            "pipeline_details": expl,
            "classification_confirmed": False,
        }
        self.show_page("results")

    def export_restored_photo(self):
        if not self.reconstructed_image_hires:
            messagebox.showwarning("No Image", "Please run the pipeline first!")
            return
        save_path = filedialog.asksaveasfilename(
            title="Export Reconstructed Textile Photo",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
        )
        if not save_path:
            return
        try:
            export_image = self.reconstructed_image_hires.resize(
                (500, 500), Image.Resampling.LANCZOS
            )
            export_image.save(save_path, format="PNG", quality=95)
            messagebox.showinfo(
                "Export Success", f"Reconstructed image saved to:\n{save_path}"
            )
        except Exception as e:
            messagebox.showerror(
                "Export Failed", f"Could not save file:\n{str(e)}"
            )

    def export_heatmap(self):
        if not self.heatmap_hires:
            messagebox.showwarning("No Image", "Please run the pipeline first!")
            return
        save_path = filedialog.asksaveasfilename(
            title="Export Error Heatmap",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
        )
        if not save_path:
            return
        try:
            export_image = self.heatmap_hires.resize(
                (500, 500), Image.Resampling.LANCZOS
            )
            export_image.save(save_path, format="PNG", quality=95)
            messagebox.showinfo(
                "Export Success", f"Heatmap saved to:\n{save_path}"
            )
        except Exception as e:
            messagebox.showerror(
                "Export Failed", f"Could not save file:\n{str(e)}"
            )

    def show_wallpaper_guide_window(self):
        guide_path = os.path.join(
            os.path.dirname(__file__), "assets", "wallpaper_group_guide.png"
        )
        if not os.path.exists(guide_path):
            messagebox.showwarning(
                "Guide Not Found", "Wallpaper group guide image not found!"
            )
            return

        if self.guide_window is not None and self.guide_window.winfo_exists():
            self.guide_window.destroy()

        self.guide_window = tk.Toplevel(self)
        self.guide_window.title("Wallpaper Group Classification Guide")
        self.guide_window.geometry("600x750")
        self.guide_window.configure(bg=self.colors["bg"])
        self.guide_window.geometry(
            f"+{self.winfo_x() + 100}+{self.winfo_y() + 100}"
        )

        header_frame = tk.Frame(
            self.guide_window, bg=self.colors["navy_btn"]
        )
        header_frame.pack(fill="x", side="top")

        header_label = tk.Label(
            header_frame,
            text="Wallpaper Group Classification",
            bg=self.colors["navy_btn"],
            fg="#FFFFFF",
            font=("Segoe UI", 12, "bold"),
            padx=15,
            pady=10,
        )
        header_label.pack(side="left", fill="x", expand=True)

        close_btn = tk.Button(
            header_frame,
            text="X",
            bg=self.colors["navy_btn"],
            fg="#FFFFFF",
            font=("Segoe UI", 14, "bold"),
            relief="flat",
            bd=0,
            padx=10,
            pady=5,
            command=self.close_wallpaper_window,
            cursor="hand2",
            activebackground="#001541",
        )
        close_btn.pack(side="right", padx=10, pady=5)

        try:
            guide_img = Image.open(guide_path)
            img_width = 580
            ratio = img_width / guide_img.width
            img_height = int(guide_img.height * ratio)
            guide_img_resized = guide_img.resize(
                (img_width, img_height), Image.Resampling.LANCZOS
            )
            guide_tk = ImageTk.PhotoImage(guide_img_resized)

            img_label = tk.Label(
                self.guide_window, image=guide_tk, bg=self.colors["bg"]
            )
            img_label.image = guide_tk
            img_label.pack(fill="both", expand=True, padx=10, pady=10)
        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to load guide image:\n{str(e)}"
            )
            self.guide_window.destroy()
            self.guide_window = None

    def close_wallpaper_window(self):
        if self.guide_window is not None and self.guide_window.winfo_exists():
            self.guide_window.destroy()
            self.guide_window = None

    def export_to_pdf(self):
        # 1. Check pipeline results dictionary
        results = getattr(self, "pipeline_results", None)

        if not results:
            orig = getattr(self, "original_image_hires", None) or getattr(
                self, "original_image_pil", None
            )
            heatmap = getattr(self, "heatmap_hires", None) or getattr(
                self, "tracking_pattern_pil", None
            )
            restored = getattr(self, "reconstructed_image_hires", None) or getattr(
                self, "reconstructed_image_pil", None
            )

            if orig is None or heatmap is None or restored is None:
                messagebox.showwarning(
                    "Export Warning",
                    "Please run the optimization pipeline first to generate all outputs before exporting.",
                )
                return

            def get_val(widget_attr):
                widget = getattr(self, widget_attr, None)
                if widget is None:
                    return "N/A"
                if hasattr(widget, "get") and hasattr(widget, "delete"):
                    return widget.get("1.0", "end-1c").strip() or "N/A"
                if hasattr(widget, "get"):
                    return str(widget.get()).strip() or "N/A"
                if hasattr(widget, "cget"):
                    return str(widget.cget("text")).strip() or "N/A"
                return "N/A"

            results = {
                "original_img": orig,
                "heatmap_img": heatmap,
                "restored_img": restored,
                "sfs_score": get_val("sfs_num_lbl"),
                "wallpaper_group": get_val("group_lbl"),
                "a1_vector": getattr(self.box_rec.num_lbl, "cget")("text") if hasattr(self, "box_rec") else "N/A",
                "a2_vector": getattr(self.box_f1.num_lbl, "cget")("text") if hasattr(self, "box_f1") else "N/A",
                "accuracy": getattr(self.grid_acc.val_lbl, "cget")("text") if hasattr(self, "grid_acc") else "N/A",
                "recall": getattr(self.grid_rec.val_lbl, "cget")("text") if hasattr(self, "grid_rec") else "N/A",
                "fpr": getattr(self.grid_fpr.val_lbl, "cget")("text") if hasattr(self, "grid_fpr") else "N/A",
                "iou": getattr(self.grid_iou.val_lbl, "cget")("text") if hasattr(self, "grid_iou") else "N/A",
                "ssim": getattr(self.grid_ssim.val_lbl, "cget")("text") if hasattr(self, "grid_ssim") else "N/A",
                "category": getattr(self, "pipeline_results", {}).get(
                    "category", "N/A"
                ),
                "subclass": getattr(self, "pipeline_results", {}).get(
                    "subclass", "N/A"
                ),
                "cultural_description": get_val("cultural_text"),
                "pipeline_details": get_val("explain_text"),
            }

        # 2. Ask user for save file path
        save_filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Documents", "*.pdf")],
            title="Save PatternGuard Audit Report",
        )
        if not save_filename:
            return

        def save_temp_img(img_obj, target_path):
            if isinstance(img_obj, str) and os.path.exists(img_obj):
                shutil.copy(img_obj, target_path)
            elif hasattr(img_obj, "save"):
                img_obj.save(target_path)
            elif isinstance(img_obj, np.ndarray):
                cv2.imwrite(target_path, img_obj)
            else:
                raise ValueError(f"Unable to process image type: {type(img_obj)}")

        temp_orig = "temp_orig.png"
        temp_heatmap = "temp_heatmap.png"
        temp_restored = "temp_restored.png"

        try:
            save_temp_img(results["original_img"], temp_orig)
            save_temp_img(results["heatmap_img"], temp_heatmap)
            save_temp_img(results["restored_img"], temp_restored)

            generate_patternguard_pdf(
                pdf_filename=save_filename,
                orig_img_path=temp_orig,
                heatmap_img_path=temp_heatmap,
                restored_img_path=temp_restored,
                sfs_score=results.get("sfs_score", "N/A"),
                wallpaper_group=results.get("wallpaper_group", "N/A"),
                a1_vector=results.get("a1_vector", "N/A"),
                a2_vector=results.get("a2_vector", "N/A"),
                accuracy=results.get("accuracy", "N/A"),
                recall=results.get("recall", "N/A"),
                fpr=results.get("fpr", "N/A"),
                iou=results.get("iou", "N/A"),
                ssim=results.get("ssim", "N/A"),
                category_name=results.get("category", "N/A"),
                subclass_name=results.get("subclass", "N/A"),
                cultural_description=results.get("cultural_description", ""),
                output_details_text=results.get("pipeline_details", ""),
            )

            messagebox.showinfo(
                "Export Successful", f"PDF Report saved to:\n{save_filename}"
            )
        except Exception as e:
            messagebox.showerror(
                "Export Error", f"Failed to save PDF report:\n{str(e)}"
            )
        finally:
            for temp_file in [temp_orig, temp_heatmap, temp_restored]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass


if __name__ == "__main__":
    app = PatternGuardApp()
    app.mainloop()