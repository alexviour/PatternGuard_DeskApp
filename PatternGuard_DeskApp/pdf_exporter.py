from datetime import datetime
from fpdf import FPDF


def clean_pdf_text(text):
  if not isinstance(text, str):
    return str(text) if text is not None else ""
  replacements = {
      "’": "'",
      "‘": "'",
      "“": '"',
      "”": '"',
      "—": "-",
      "–": "-",
      "…": "...",
  }
  for unicode_char, ascii_char in replacements.items():
    text = text.replace(unicode_char, ascii_char)
  return text.encode("latin-1", "replace").decode("latin-1")


class PatternGuardPDFReport(FPDF):

  def header(self):
    self.set_font("Helvetica", "B", 15)
    self.set_text_color(139, 0, 0)
    self.cell(
        0, 8, "PATTERNGUARD: FILIPINO TEXTILE AUDIT REPORT", ln=True, align="L"
    )

    self.set_font("Helvetica", "B", 8)
    self.set_text_color(100, 110, 120)
    self.cell(
        0,
        4,
        "AUTOMATED TEXTILE AUDITING & RESTORATION FRAMEWORK",
        ln=True,
        align="L",
    )

    self.set_draw_color(200, 200, 200)
    self.line(10, 22, 200, 22)
    self.ln(6)

  def footer(self):
    self.set_y(-12)
    self.set_font("Helvetica", "I", 8)
    self.set_text_color(128, 128, 128)
    self.cell(
        0,
        10,
        f"PatternGuard DeskApp Framework  |  Generated on:"
        f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Page"
        f" {self.page_no()}",
        align="C",
    )


def generate_patternguard_pdf(
    pdf_filename,
    orig_img_path,
    heatmap_img_path,
    restored_img_path,
    sfs_score="",
    wallpaper_group="",
    a1_vector="",
    a2_vector="",
    accuracy="",
    recall="",
    fpr="",
    iou="",
    ssim="",
    category_name="",
    subclass_name="",
    cultural_description="",
    output_details_text="",
):
  # Sanitize dynamic strings
  sfs_score = clean_pdf_text(sfs_score)
  wallpaper_group = clean_pdf_text(wallpaper_group)
  a1_vector = clean_pdf_text(a1_vector)
  a2_vector = clean_pdf_text(a2_vector)
  accuracy = clean_pdf_text(accuracy)
  recall = clean_pdf_text(recall)
  fpr = clean_pdf_text(fpr)
  iou = clean_pdf_text(iou)
  ssim = clean_pdf_text(ssim)
  category_name = clean_pdf_text(category_name)
  subclass_name = clean_pdf_text(subclass_name)
  cultural_description = clean_pdf_text(cultural_description)
  output_details_text = clean_pdf_text(output_details_text)

  pdf = PatternGuardPDFReport()
  pdf.add_page()
  pdf.set_auto_page_break(auto=True, margin=12)

  # 1. Visual Outputs
  pdf.set_font("Helvetica", "B", 11)
  pdf.set_text_color(26, 54, 93)
  pdf.cell(0, 6, "1. Visual Audit & Reconstruction Outputs", ln=True)
  pdf.ln(2)

  y_start = pdf.get_y()
  img_w, img_h = 58, 58

  pdf.image(orig_img_path, x=10, y=y_start, w=img_w, h=img_h)
  pdf.image(heatmap_img_path, x=76, y=y_start, w=img_w, h=img_h)
  pdf.image(restored_img_path, x=142, y=y_start, w=img_w, h=img_h)

  pdf.set_y(y_start + img_h + 2)
  pdf.set_font("Helvetica", "B", 8)
  pdf.set_text_color(60, 60, 60)
  pdf.text(24, y_start + img_h + 6, "ORIGINAL IMAGE")
  pdf.text(84, y_start + img_h + 6, "DETECTED ERROR HEATMAP")
  pdf.text(148, y_start + img_h + 6, "RECONSTRUCTED TEXTILE")

  pdf.set_y(y_start + img_h + 10)

  # 2. Quantitative Evaluation & Metrics
  pdf.set_font("Helvetica", "B", 11)
  pdf.set_text_color(26, 54, 93)
  pdf.cell(0, 6, "2. Quantitative Evaluation & Lattice Metrics", ln=True)
  pdf.ln(2)

  pdf.set_font("Helvetica", "B", 9)
  pdf.set_fill_color(245, 247, 250)

  pdf.cell(
      95, 7, f" Symmetry Fidelity Score (SFS): {sfs_score}", border=1, fill=True
  )
  pdf.cell(
      95,
      7,
      f" Auto-Detected Wallpaper Group: {wallpaper_group}",
      border=1,
      fill=True,
  )
  pdf.ln()

  pdf.set_font("Helvetica", "", 8.5)
  pdf.cell(47.5, 6, f" A1 Lattice: {a1_vector}", border=1, align="C")
  pdf.cell(47.5, 6, f" A2 Lattice: {a2_vector}", border=1, align="C")
  pdf.cell(47.5, 6, f" Accuracy: {accuracy}", border=1, align="C")
  pdf.cell(47.5, 6, f" Recall: {recall}", border=1, align="C")
  pdf.ln()

  pdf.cell(47.5, 6, f" FPR: {fpr}", border=1, align="C")
  pdf.cell(47.5, 6, f" IoU: {iou}", border=1, align="C")
  pdf.cell(
      95, 6, f" Structural Similarity (SSIM): {ssim}", border=1, align="C"
  )
  pdf.ln(8)

  # 3. Cultural Metadata
  pdf.set_font("Helvetica", "B", 11)
  pdf.set_text_color(26, 54, 93)
  pdf.cell(0, 6, "3. Cultural Background & Motif Metadata", ln=True)

  pdf.set_font("Helvetica", "B", 8.5)
  pdf.set_text_color(139, 0, 0)
  pdf.cell(
      0,
      5,
      f"Textile Category: {category_name}   |   Subclass: {subclass_name}",
      ln=True,
  )

  pdf.set_font("Helvetica", "I", 8.5)
  pdf.set_text_color(50, 50, 50)
  pdf.multi_cell(0, 4.5, cultural_description, border=0)
  pdf.ln(4)

  # 4. Detailed Pipeline Instructions
  pdf.set_font("Helvetica", "B", 11)
  pdf.set_text_color(26, 54, 93)
  pdf.cell(
      0, 6, "4. Pipeline Execution Details (How Result Was Produced)", ln=True
  )

  pdf.set_font("Helvetica", "", 8)
  pdf.set_text_color(40, 40, 40)
  pdf.set_fill_color(250, 250, 252)
  pdf.multi_cell(0, 4, output_details_text, border=1, fill=True)

  pdf.output(pdf_filename)