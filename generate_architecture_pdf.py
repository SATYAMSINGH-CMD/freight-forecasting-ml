import os
import sys
from playwright.sync_api import sync_playwright
import pypdfium2 as pdfium

def generate_pdf():
    workspace_dir = os.path.abspath(r"d:\freight forecasting")
    html_file = os.path.join(workspace_dir, "freight_forecasting_flowchart.html")
    pdf_file = os.path.join(workspace_dir, "freight_forecasting_system_architecture.pdf")
    png_file = os.path.join(workspace_dir, "freight_forecasting_system_architecture.png")

    print(f"Loading HTML: {html_file}")
    file_url = f"file:///{html_file.replace(os.sep, '/')}"

    width_px = 4250
    height_px = 2200

    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=[
                "--font-render-hinting=none",
                "--enable-font-antialiasing",
                "--force-color-profile=srgb"
            ]
        )
        page = browser.new_page(
            viewport={"width": width_px, "height": height_px},
            device_scale_factor=2
        )
        
        # Navigate to local HTML file
        page.goto(file_url, wait_until="networkidle")
        
        # Wait extra second for Google Fonts and SVG connector script to execute
        page.wait_for_timeout(2000)

        # 1. Capture high-res PNG for pitch decks / quick review
        print(f"Generating high-res PNG preview: {png_file}")
        page.screenshot(
            path=png_file,
            full_page=True
        )

        # 2. Render vector PDF
        print(f"Generating vector PDF: {pdf_file}")
        page.pdf(
            path=pdf_file,
            width=f"{width_px}px",
            height=f"{height_px}px",
            print_background=True,
            page_ranges="1",
            margin={"top": "0px", "right": "0px", "bottom": "0px", "left": "0px"}
        )

        browser.close()

    print("Checking generated PDF with pypdfium2...")
    pdf = pdfium.PdfDocument(pdf_file)
    page_count = len(pdf)
    print(f"PDF Page Count: {page_count}")
    
    first_page = pdf[0]
    w, h = first_page.get_size()
    print(f"PDF Dimensions: {w:.1f} pt x {h:.1f} pt (equivalent to {w*96/72:.0f}px x {h*96/72:.0f}px)")
    
    # Render page 1 to check for render errors
    image = first_page.render(scale=0.5).to_pil()
    sample_preview = os.path.join(workspace_dir, "pdf_render_verification.png")
    image.save(sample_preview)
    print(f"PDF verification render saved: {sample_preview}")
    
    print("\nSUCCESS! High-detail architecture flowchart PDF generated successfully.")

if __name__ == "__main__":
    generate_pdf()
