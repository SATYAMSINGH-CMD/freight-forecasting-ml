import os
from playwright.sync_api import sync_playwright
import pypdfium2 as pdfium

def render():
    workspace = os.path.abspath(r"d:\freight forecasting")
    html_path = os.path.join(workspace, "what_are_we_predicting.html")
    pdf_path = os.path.join(workspace, "what_are_we_predicting.pdf")
    png_path = os.path.join(workspace, "what_are_we_predicting.png")
    
    file_url = f"file:///{html_path.replace(os.sep, '/')}"
    print(f"Loading {file_url}...")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=[
                "--font-render-hinting=none",
                "--enable-font-antialiasing",
                "--force-color-profile=srgb"
            ]
        )
        page = browser.new_page(
            viewport={"width": 1280, "height": 1300},
            device_scale_factor=2
        )
        page.goto(file_url, wait_until="networkidle")
        page.wait_for_timeout(1000)
        
        # Get content height for perfect 1-page fit
        content_height = page.evaluate("() => document.querySelector('.page-container').offsetHeight + 60")
        print(f"Calculated page container height: {content_height}px")
        
        # 1. Save High-Res PNG
        print(f"Saving high-res PNG to {png_path}...")
        page.screenshot(path=png_path, full_page=True)
        
        # 2. Save crisp PDF
        print(f"Saving PDF to {pdf_path}...")
        page.pdf(
            path=pdf_path,
            width="1280px",
            height=f"{content_height}px",
            print_background=True,
            margin={"top": "0px", "right": "0px", "bottom": "0px", "left": "0px"}
        )
        browser.close()
        
    print("Verifying PDF with pypdfium2...")
    doc = pdfium.PdfDocument(pdf_path)
    print(f"PDF page count: {len(doc)}")
    print("Done! High quality PDF and PNG generated.")

if __name__ == "__main__":
    render()
