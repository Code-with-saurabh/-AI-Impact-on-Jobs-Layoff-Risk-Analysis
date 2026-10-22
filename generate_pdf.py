from fpdf import FPDF
import os

def sanitize_text(text):
    replacements = {
        '\u2013': '-', '\u2014': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2022': '*', '\u2026': '...',
        '\u2713': 'x', '\u2717': 'x', '\u25cf': '*', '\u25cb': 'o',
        '\u25ba': '>', '\u25b6': '>',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode('latin-1', errors='ignore').decode('latin-1')

class CodePDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 8, 'AI Impact on Jobs & Layoff Risk Analysis - Complete Code', 0, 1, 'C')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    def add_file_section(self, filename, content):
        if self.get_y() > 240:
            self.add_page()
        
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(52, 73, 94)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, f'  {filename}', 0, 1, 'L', True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

        self.set_font('Courier', '', 7)
        lines = content.split('\n')
        for line in lines:
            clean_line = sanitize_text(line)
            if len(clean_line) > 110:
                chunks = [clean_line[i:i+110] for i in range(0, len(clean_line), 110)]
                for chunk in chunks:
                    self.cell(0, 3.5, chunk, 0, 1)
            else:
                self.cell(0, 3.5, clean_line, 0, 1)
        
        self.ln(5)

project_path = r"D:\Saurabh Sir\Clg\py\AI-Impact-Jobs-Layoff-Risk"

files = [
    ("config.py", os.path.join(project_path, "config.py")),
    ("requirements.txt", os.path.join(project_path, "requirements.txt")),
    ("main.py", os.path.join(project_path, "main.py")),
    ("src/__init__.py", os.path.join(project_path, "src", "__init__.py")),
    ("src/data_loading.py", os.path.join(project_path, "src", "data_loading.py")),
    ("src/data_cleaning.py", os.path.join(project_path, "src", "data_cleaning.py")),
    ("src/exploratory_data_analysis.py", os.path.join(project_path, "src", "exploratory_data_analysis.py")),
    ("src/feature_engineering.py", os.path.join(project_path, "src", "feature_engineering.py")),
    ("src/visualization.py", os.path.join(project_path, "src", "visualization.py")),
    ("src/preprocessing.py", os.path.join(project_path, "src", "preprocessing.py")),
    ("src/model_training.py", os.path.join(project_path, "src", "model_training.py")),
    ("src/model_evaluation.py", os.path.join(project_path, "src", "model_evaluation.py")),
    ("src/prediction.py", os.path.join(project_path, "src", "prediction.py")),
    ("src/utils.py", os.path.join(project_path, "src", "utils.py")),
    ("README.md", os.path.join(project_path, "README.md")),
]

pdf = CodePDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=15)

pdf.add_page()
pdf.set_font('Helvetica', 'B', 24)
pdf.ln(30)
pdf.cell(0, 15, 'AI Impact on Jobs &', 0, 1, 'C')
pdf.cell(0, 15, 'Layoff Risk Analysis', 0, 1, 'C')
pdf.ln(8)
pdf.set_font('Helvetica', '', 14)
pdf.cell(0, 10, 'Using Machine Learning', 0, 1, 'C')
pdf.ln(10)
pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 8, 'Complete Project Code', 0, 1, 'C')
pdf.cell(0, 8, 'All 3 Phases (Month-1, Month-2, Month-3)', 0, 1, 'C')
pdf.ln(8)
pdf.set_font('Helvetica', '', 11)
pdf.cell(0, 8, 'GTU Semester-5', 0, 1, 'C')
pdf.cell(0, 8, 'BE05000231 - Python for Data Science', 0, 1, 'C')
pdf.ln(10)
pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 8, 'Student Details', 0, 1, 'C')
pdf.ln(3)
pdf.set_font('Helvetica', '', 11)
pdf.cell(0, 8, 'Name: Sharma Saurabh', 0, 1, 'C')
pdf.cell(0, 8, 'Enrollment: 250863131007', 0, 1, 'C')
pdf.ln(10)
pdf.set_font('Helvetica', '', 10)
pdf.cell(0, 8, 'Project Duration: 3 Months', 0, 1, 'C')
pdf.cell(0, 8, 'Status: Completed', 0, 1, 'C')

pdf.add_page()
pdf.set_font('Helvetica', 'B', 16)
pdf.cell(0, 10, 'Table of Contents', 0, 1, 'L')
pdf.line(10, pdf.get_y(), 200, pdf.get_y())
pdf.ln(5)

pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 8, 'Month-1: Foundation Phase', 0, 1)
pdf.set_font('Helvetica', '', 10)
toc_items_m1 = [
    "  1. config.py - Project Configuration",
    "  2. requirements.txt - Dependencies",
    "  3. main.py - Main Entry Point",
    "  4. src/__init__.py - Package Initializer",
    "  5. src/data_loading.py - Data Loading Module",
    "  6. src/data_cleaning.py - Data Cleaning Module",
]
for item in toc_items_m1:
    pdf.cell(0, 6, item, 0, 1)

pdf.ln(3)
pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 8, 'Month-2: Analysis Phase', 0, 1)
pdf.set_font('Helvetica', '', 10)
toc_items_m2 = [
    "  7. src/exploratory_data_analysis.py - EDA Module",
    "  8. src/feature_engineering.py - Feature Engineering",
    "  9. src/visualization.py - Data Visualization",
    "  10. src/preprocessing.py - Data Preprocessing",
]
for item in toc_items_m2:
    pdf.cell(0, 6, item, 0, 1)

pdf.ln(3)
pdf.set_font('Helvetica', 'B', 12)
pdf.cell(0, 8, 'Month-3: Modeling Phase', 0, 1)
pdf.set_font('Helvetica', '', 10)
toc_items_m3 = [
    "  11. src/model_training.py - Model Training",
    "  12. src/model_evaluation.py - Model Evaluation",
    "  13. src/prediction.py - Prediction System",
    "  14. src/utils.py - Utility Functions",
    "  15. README.md - Project Documentation",
]
for item in toc_items_m3:
    pdf.cell(0, 6, item, 0, 1)

for filename, filepath in files:
    pdf.add_page()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        pdf.add_file_section(filename, content)
    except Exception as e:
        pdf.add_file_section(filename, f"Error reading file: {e}")

output_path = os.path.join(project_path, "AI_Impact_Jobs_Complete_Code.pdf")
pdf.output(output_path)
print(f"PDF generated successfully: {output_path}")
