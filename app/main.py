import ttkbootstrap as ttk
from gui import YOLOApp

if __name__ == "__main__":
    root = ttk.Window(themename="darkly")
    app = YOLOApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
