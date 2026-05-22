import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from bidi.algorithm import get_display
import arabic_reshaper

# Register Arabic font
font_path = "Amiri-Regular.ttf"
try:
    pdfmetrics.registerFont(TTFont('ArabicFont', font_path))
    font_name = 'ArabicFont'
except:
    font_name = 'Helvetica'
    print("تحذير: تعذر تحميل الخط العربي، يتم استخدام خط افتراضي.")

def format_arabic(text):
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text

def draw_text(c, text, x, y, font_size=12, align="right"):
    c.setFont(font_name, font_size)
    formatted_text = format_arabic(text)
    if align == "right":
        c.drawRightString(x, y, formatted_text)
    else:
        c.drawString(x, y, formatted_text)

def create_pdf():
    pdf_file = "smart_invoice_erd.pdf"
    c = canvas.Canvas(pdf_file, pagesize=letter)
    width, height = letter

    # Page 1: Title and ERD Introduction
    draw_text(c, "نظام إدارة الفواتير الذكي", width - 40, height - 50, 24)
    draw_text(c, "1. مخطط العلاقات (ERD)", width - 40, height - 90, 20)
    draw_text(c, "يتكون مخطط العلاقات من الكيانات التالية وسماتها:", width - 40, height - 120, 14)

    y_pos = height - 150
    entities = [
        "- العملاء (Client_ID كمفتاح أساسي، الاسم، البريد الإلكتروني، الهاتف، العنوان)",
        "- المنتجات (Product_ID كمفتاح أساسي، الاسم، السعر، كمية المخزون)",
        "- الفواتير (Invoice_ID كمفتاح أساسي، Client_ID، تاريخ الإصدار، المبلغ الإجمالي، الحالة)",
        "- عناصر الفاتورة (Invoice_ID، Product_ID، الكمية، سعر الوحدة)",
        "- المدفوعات (Payment_ID كمفتاح أساسي، Invoice_ID، المبلغ، تاريخ الدفع، طريقة الدفع)"
    ]
    for entity in entities:
        draw_text(c, entity, width - 40, y_pos, 12)
        y_pos -= 20

    c.showPage()

    # Page 2: Relationships and Visual Representation
    draw_text(c, "تشمل العلاقات إصدار الفواتير من العملاء، احتواء الفواتير على عناصر، وجود المنتجات في عناصر الفاتورة، وتسجيل المدفوعات للفواتير.", width - 40, height - 50, 12)
    draw_text(c, "التمثيل المرئي (نصي):", width - 40, height - 80, 14)

    y_pos = height - 110
    visual = [
        "[العملاء] ----(1:N)---- [الفواتير] ----(1:N)---- [عناصر الفاتورة] ----(N:1)---- [المنتجات]",
        "                         |",
        "                         +--------(1:N)---- [المدفوعات]",
        "",
        "[العملاء]           [الفواتير]          [عناصر الفاتورة]        [المنتجات]         [المدفوعات]",
        "+---------+         +----------+         +------------+        +----------+        +----------+",
        "| Client_ID | PK |---->| Client_ID | FK    | Invoice_ID | FK <---| Product_ID | PK |  | Payment_ID | PK |",
        "| الاسم      |       | | Invoice_ID| PK |---->| Product_ID | FK ----| الاسم       |     | | Invoice_ID | FK |",
        "| البريد     |       | | تاريخ الإصدار |   | الكمية    |        | السعر      |     | | المبلغ     |     |",
        "| الهاتف     |       | | المبلغ الإجمالي| | سعر الوحدة |        | كمية المخزون|  | | تاريخ الدفع|    |",
        "| العنوان   |       | | الحالة     |     +------------+        +-------------+   | | الطريقة    |     |",
        "+-----------+        +------------+                                            +-------------+"
    ]
    for line in visual:
        draw_text(c, line, width - 40, y_pos, 10, align="left")
        y_pos -= 15

    c.showPage()

    # Page 3: Relational Schema (Part 1)
    draw_text(c, "2. المخطط العلائقي", width - 40, height - 50, 20)
    
    y_pos = height - 80
    schema_part1 = [
        "العملاء",
        "CLIENTS (",
        "    Client_ID INTEGER PRIMARY KEY AUTOINCREMENT,",
        "    Name TEXT NOT NULL,",
        "    Email TEXT,",
        "    Phone TEXT,",
        "    Address TEXT",
        ")",
        "",
        "المنتجات",
        "PRODUCTS (",
        "    Product_ID INTEGER PRIMARY KEY AUTOINCREMENT,",
        "    Name TEXT NOT NULL UNIQUE,",
        "    Price REAL NOT NULL,",
        "    Stock_Quantity INTEGER NOT NULL",
        ")",
        "",
        "الفواتير",
        "INVOICES (",
        "    Invoice_ID INTEGER PRIMARY KEY AUTOINCREMENT,",
        "    Client_ID INTEGER NOT NULL,",
        "    Issue_Date TEXT NOT NULL,",
        "    Total_Amount REAL NOT NULL,",
        "    Status TEXT NOT NULL,",
        "    FOREIGN KEY (Client_ID) REFERENCES CLIENTS(Client_ID)",
        ")"
    ]
    for line in schema_part1:
        draw_text(c, line, width - 40, y_pos, 10, align="left")
        y_pos -= 15

    c.showPage()

    # Page 4: Relational Schema (Part 2)
    y_pos = height - 50
    schema_part2 = [
        "عناصر الفاتورة",
        "INVOICE_ITEMS (",
        "    Invoice_ID INTEGER,",
        "    Product_ID INTEGER,",
        "    Quantity INTEGER NOT NULL,",
        "    Unit_Price REAL NOT NULL,",
        "    PRIMARY KEY (Invoice_ID, Product_ID),",
        "    FOREIGN KEY (Invoice_ID) REFERENCES INVOICES(Invoice_ID),",
        "    FOREIGN KEY (Product_ID) REFERENCES PRODUCTS(Product_ID)",
        ")",
        "",
        "المدفوعات",
        "PAYMENTS (",
        "    Payment_ID INTEGER PRIMARY KEY AUTOINCREMENT,",
        "    Invoice_ID INTEGER NOT NULL,",
        "    Amount REAL NOT NULL,",
        "    Payment_Date TEXT NOT NULL,",
        "    Method TEXT NOT NULL,",
        "    FOREIGN KEY (Invoice_ID) REFERENCES INVOICES(Invoice_ID)",
        ")"
    ]
    for line in schema_part2:
        draw_text(c, line, width - 40, y_pos, 10, align="left")
        y_pos -= 15

    c.showPage()
    c.save()
    print(f"تم إنشاء ملف PDF بنجاح: {pdf_file}")

if __name__ == "__main__":
    create_pdf()