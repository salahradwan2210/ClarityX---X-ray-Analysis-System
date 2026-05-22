import sqlite3
import os
import re
import logging
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from tkcalendar import DateEntry
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, f"app_errors_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_error(message, exception=None):
    error_msg = f"{message}"
    if exception:
        error_msg += f" | Exception: {str(exception)}"
    logging.error(error_msg)

# Database setup
conn = None
cursor = None

def connect_to_db():
    global conn, cursor
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            conn = sqlite3.connect('warehouse.db')
            cursor = conn.cursor()
            create_tables()
            return True
        except sqlite3.Error as e:
            log_error(f"Database connection attempt {attempt + 1} failed", e)
            if attempt == max_attempts - 1:
                return False
    return False

def create_tables():
    try:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                phone TEXT,
                deleted INTEGER DEFAULT 0
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                deleted INTEGER DEFAULT 0
            )
        ''')
        conn.commit()
    except sqlite3.Error as e:
        log_error("Failed to create tables", e)
        raise

def get_all_customers():
    try:
        cursor.execute("SELECT id, name FROM customers WHERE deleted = 0 ORDER BY id ASC")
        return [f"{row[0]} - {row[1]}" for row in cursor.fetchall()]
    except sqlite3.Error as e:
        log_error("Failed to fetch customers", e)
        return []

def get_all_products():
    try:
        cursor.execute("SELECT name FROM products WHERE deleted = 0 ORDER BY id ASC")
        return [row[0] for row in cursor.fetchall()]
    except sqlite3.Error as e:
        log_error("Failed to fetch products", e)
        return []

def add_customer(name, phone):
    try:
        cursor.execute("SELECT id FROM customers WHERE name = ? AND deleted = 1", (name,))
        result = cursor.fetchone()
        if result:
            customer_id = result[0]
            cursor.execute("UPDATE customers SET phone = ?, deleted = 0 WHERE id = ?", (phone, customer_id))
        else:
            cursor.execute("INSERT INTO customers (name, phone) VALUES (?, ?)", (name, phone))
        conn.commit()
        return cursor.lastrowid if not result else customer_id
    except sqlite3.IntegrityError as e:
        conn.rollback()
        log_error(f"Integrity error adding customer: {name}", e)
        return False
    except sqlite3.Error as e:
        conn.rollback()
        log_error(f"Database error adding customer: {name}", e)
        return False

def close_db():
    if conn:
        conn.close()

# PDF utilities
def generate_invoice_pdf(invoice_data, pdf_file):
    try:
        c = canvas.Canvas(pdf_file, pagesize=letter)
        width, height = letter
        c.setFont('Helvetica-Bold', 24)
        
        c.drawString(40, height - 50, "Warehouse Company")
        
        c.setFont('Helvetica-Bold', 20)
        c.drawString(40, height - 90, "Invoice")

        c.setFont('Helvetica', 12)
        y_pos = height - 130
        customer_info = [
            f"Customer Name: {invoice_data['customer']}",
            f"Date: {invoice_data['date']}"
        ]
        if "customer_id" in invoice_data:
            customer_info.append(f"Customer ID: {invoice_data['customer_id']}")
        
        for info in customer_info:
            c.drawString(50, y_pos, info)
            y_pos -= 20

        headers = ["Product", "Quantity", "Price", "Total"]
        col_positions = [50, 150, 250, 350]
        for header, pos in zip(headers, col_positions):
            c.drawString(pos, y_pos, header)
        y_pos -= 30

        for item in invoice_data['items']:
            values = [
                item['name'],
                str(item['quantity']),
                f"${item['price']:.2f}",
                f"${item['quantity'] * item['price']:.2f}"
            ]
            for value, pos in zip(values, col_positions):
                c.drawString(pos, y_pos, value)
            y_pos -= 25

        y_pos = max(y_pos, height - 400)
        c.setFont('Helvetica-Bold', 14)
        summary_y = y_pos - 40
        total_line = f"Total: ${invoice_data['total']:.2f}"
        c.drawString(350, summary_y, total_line)

        if 'paid' in invoice_data and invoice_data['paid'] is not None:
            paid_line = f"Paid: ${invoice_data['paid']:.2f}"
            c.drawString(350, summary_y - 20, paid_line)

        if 'remaining' in invoice_data and invoice_data['remaining'] is not None:
            remaining_line = f"Balance: ${invoice_data['remaining']:.2f}"
            c.drawString(350, summary_y - 40, remaining_line)

        c.showPage()
        c.save()
        return True
    except Exception as e:
        log_error(f"Failed to generate PDF: {pdf_file}", e)
        return False

# GUI components
def validate_phone(phone):
    pattern = r'^\+?\d+$'
    return bool(re.match(pattern, phone))

class AutocompleteEntry(tk.Entry):
    def __init__(self, autocompleteList, *args, **kwargs):
        tk.Entry.__init__(self, *args, **kwargs)
        self.autocompleteList = autocompleteList
        self.var = kwargs.get("textvariable", None)
        if not self.var:
            self.var = tk.StringVar()
            self["textvariable"] = self.var
        self.var.trace("w", self.changed)
        self.bind("<Right>", self.selection)
        self.bind("<Down>", self.moveDown)
        self.bind("<Up>", self.moveUp)
        self.listboxUp = False

    def changed(self, name, index, mode):
        if self.var.get() == '':
            if self.listboxUp:
                self.listbox.destroy()
                self.listboxUp = False
        else:
            words = self.comparison()
            if words:
                if not self.listboxUp:
                    self.listbox = tk.Listbox(self.master, width=self["width"], height=5)
                    self.listbox.bind("<Double-Button-1>", self.selection)
                    self.listbox.bind("<Right>", self.selection)
                    self.listbox.place(x=self.winfo_rootx(), y=self.winfo_rooty() + self.winfo_height())
                    self.listboxUp = True
                self.listbox.delete(0, tk.END)
                for w in words:
                    self.listbox.insert(tk.END, w)
            else:
                if self.listboxUp:
                    self.listbox.destroy()
                    self.listboxUp = False

    def selection(self, event):
        if self.listboxUp:
            self.var.set(self.listbox.get(tk.ACTIVE))
            self.listbox.destroy()
            self.listboxUp = False
            self.icursor(tk.END)

    def moveUp(self, event):
        if self.listboxUp:
            if self.listbox.curselection():
                index = int(self.listbox.curselection()[0])
                if index != 0:
                    self.listbox.selection_clear(first=index)
                    index -= 1
                    self.listbox.selection_set(first=index)
                    self.listbox.activate(index)

    def moveDown(self, event):
        if self.listboxUp:
            if self.listbox.curselection():
                index = int(self.listbox.curselection()[0])
                if index < self.listbox.size() - 1:
                    self.listbox.selection_clear(first=index)
                    index += 1
                    self.listbox.selection_set(first=index)
                    self.listbox.activate(index)
            else:
                self.listbox.selection_set(first=0)
                self.listbox.activate(0)

    def comparison(self):
        pattern = self.var.get().lower()
        return [w for w in self.autocompleteList if w.lower().startswith(pattern)]

class CustomerSelectPopup(tk.Toplevel):
    def __init__(self, master, callback):
        super().__init__(master)
        self.title("Select Customer")
        self.geometry("600x400")
        self.callback = callback
        self.selected_customer = None
        self.setup_ui()
        self.grab_set()

    def setup_ui(self):
        search_frame = tk.Frame(self)
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        self.search_entry = ttk.Entry(search_frame, width=30)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(search_frame, text="Search", command=self.search).pack(side=tk.LEFT, padx=5)
        ttk.Button(search_frame, text="Confirm", command=self.apply_selection).pack(side=tk.LEFT, padx=5)

        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.tree = ttk.Treeview(tree_frame, columns=("ID", "Name"), show="headings")
        self.tree.heading("ID", text="Customer ID")
        self.tree.heading("Name", text="Customer Name")
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.populate_tree()

    def populate_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        customers = get_all_customers()
        for customer in customers:
            customer_id, customer_name = customer.split(" - ", 1)
            self.tree.insert("", tk.END, values=(customer_id, customer_name))

    def search(self):
        query = self.search_entry.get().lower()
        self.populate_tree()
        if query:
            for item in self.tree.get_children():
                values = self.tree.item(item, "values")
                if not (query in values[0].lower() or query in values[1].lower()):
                    self.tree.delete(item)

    def apply_selection(self):
        selected = self.tree.selection()
        if selected:
            self.selected_customer = self.tree.item(selected[0], "values")
            self.callback(self.selected_customer)
            self.destroy()

class WarehouseApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Warehouse Management System")
        self.geometry("800x600")
        if not connect_to_db():
            messagebox.showerror("Error", "Failed to connect to database.")
            self.destroy()
            return
        self.setup_ui()

    def setup_ui(self):
        self.content_frame = ttk.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        customer_frame = ttk.Frame(self.content_frame)
        customer_frame.pack(fill=tk.X, pady=5)
        ttk.Label(customer_frame, text="Customer:").pack(side=tk.LEFT)
        self.customer_var = tk.StringVar()
        self.customer_entry = AutocompleteEntry(get_all_customers(), customer_frame, textvariable=self.customer_var, width=30)
        self.customer_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(customer_frame, text="Select Customer", command=self.open_customer_popup).pack(side=tk.LEFT, padx=5)

        ttk.Label(customer_frame, text="New Customer Name:").pack(side=tk.LEFT, padx=5)
        self.new_customer_name = ttk.Entry(customer_frame, width=20)
        self.new_customer_name.pack(side=tk.LEFT, padx=5)
        ttk.Label(customer_frame, text="Phone:").pack(side=tk.LEFT, padx=5)
        self.new_customer_phone = ttk.Entry(customer_frame, width=15)
        self.new_customer_phone.pack(side=tk.LEFT, padx=5)
        ttk.Button(customer_frame, text="Add Customer", command=self.add_new_customer).pack(side=tk.LEFT, padx=5)

        items_frame = ttk.Frame(self.content_frame)
        items_frame.pack(fill=tk.X, pady=5)
        ttk.Label(items_frame, text="Product:").pack(side=tk.LEFT)
        self.product_var = tk.StringVar()
        self.product_entry = AutocompleteEntry(get_all_products(), items_frame, textvariable=self.product_var, width=30)
        self.product_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(items_frame, text="Quantity:").pack(side=tk.LEFT, padx=5)
        self.quantity_entry = ttk.Entry(items_frame, width=10)
        self.quantity_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(items_frame, text="Price:").pack(side=tk.LEFT, padx=5)
        self.price_entry = ttk.Entry(items_frame, width=10)
        self.price_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(items_frame, text="Add to Invoice", command=self.add_item).pack(side=tk.LEFT, padx=5)

        self.items_tree = ttk.Treeview(self.content_frame, columns=("Name", "Quantity", "Price", "Total"), show="headings")
        self.items_tree.heading("Name", text="Product")
        self.items_tree.heading("Quantity", text="Quantity")
        self.items_tree.heading("Price", text="Price")
        self.items_tree.heading("Total", text="Total")
        self.items_tree.pack(fill=tk.BOTH, expand=True, pady=5)

        invoice_frame = ttk.Frame(self.content_frame)
        invoice_frame.pack(fill=tk.X, pady=5)
        ttk.Label(invoice_frame, text="Invoice Date:").pack(side=tk.LEFT)
        self.date_entry = DateEntry(invoice_frame, date_pattern='yyyy-mm-dd')
        self.date_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(invoice_frame, text="Paid Amount:").pack(side=tk.LEFT, padx=5)
        self.paid_entry = ttk.Entry(invoice_frame, width=10)
        self.paid_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(invoice_frame, text="Generate Invoice", command=self.generate_invoice).pack(side=tk.LEFT, padx=5)

        self.items = []

    def open_customer_popup(self):
        CustomerSelectPopup(self, self.set_customer)

    def set_customer(self, customer):
        self.customer_var.set(f"{customer[0]} - {customer[1]}")

    def add_new_customer(self):
        name = self.new_customer_name.get().strip()
        phone = self.new_customer_phone.get().strip()
        if not name or not phone:
            messagebox.showwarning("Error", "Please enter customer name and phone number.")
            return
        if not validate_phone(phone):
            messagebox.showwarning("Error", "Invalid phone number.")
            return
        customer_id = add_customer(name, phone)
        if customer_id:
            messagebox.showinfo("Success", "Customer added successfully.")
            self.customer_entry.autocompleteList = get_all_customers()
            self.new_customer_name.delete(0, tk.END)
            self.new_customer_phone.delete(0, tk.END)
        else:
            messagebox.showerror("Error", "Failed to add customer.")

    def add_item(self):
        name = self.product_var.get().strip()
        quantity = self.quantity_entry.get().strip()
        price = self.price_entry.get().strip()
        try:
            quantity = int(quantity)
            price = float(price)
            if quantity <= 0 or price < 0:
                raise ValueError("Invalid quantity or price")
        except ValueError:
            messagebox.showwarning("Error", "Please enter valid quantity and price.")
            return
        if not name:
            messagebox.showwarning("Error", "Please enter product name.")
            return
        total = quantity * price
        self.items.append({"name": name, "quantity": quantity, "price": price})
        self.items_tree.insert("", tk.END, values=(name, quantity, f"${price:.2f}", f"${total:.2f}"))
        self.product_var.set("")
        self.quantity_entry.delete(0, tk.END)
        self.price_entry.delete(0, tk.END)

    def generate_invoice(self):
        if not self.customer_var.get():
            messagebox.showwarning("Error", "Please select a customer.")
            return
        if not self.items:
            messagebox.showwarning("Error", "Please add products to the invoice.")
            return
        customer_id, customer_name = self.customer_var.get().split(" - ", 1)
        total = sum(item["quantity"] * item["price"] for item in self.items)
        paid = self.paid_entry.get().strip()
        paid = float(paid) if paid else 0.0
        remaining = total - paid if paid else None

        invoice_data = {
            "customer": customer_name,
            "customer_id": customer_id,
            "date": self.date_entry.get(),
            "items": self.items,
            "total": total,
            "paid": paid,
            "remaining": remaining
        }

        default_filename = f"Invoice_{customer_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
        pdf_file = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            initialfile=default_filename,
            filetypes=[("PDF Files", "*.pdf")]
        )
        if not pdf_file:
            return

        if generate_invoice_pdf(invoice_data, pdf_file):
            messagebox.showinfo("Success", f"Invoice created: {pdf_file}")
            self.items = []
            for item in self.items_tree.get_children():
                self.items_tree.delete(item)
            self.customer_var.set("")
            self.paid_entry.delete(0, tk.END)
        else:
            messagebox.showerror("Error", "Failed to create invoice. Please check path or permissions.")

    def destroy(self):
        close_db()
        super().destroy()

if __name__ == "__main__":
    app = WarehouseApp()
    app.mainloop() 