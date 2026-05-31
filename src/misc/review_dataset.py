"""
Dataset image reviewer — run from repo root: python review_dataset.py
Browse result/ad/ (by batch) and result/non_ad/ (by image), mark bad frames,
and move them to result/discarded/ad/ or result/discarded/non_ad/.
"""

import json
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    cwd = Path.cwd()
    default = json.loads((cwd / "default_config.json").read_text())
    cfg_path = cwd / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return default


# ---------------------------------------------------------------------------
# Scanning helpers
# ---------------------------------------------------------------------------

THUMB_W, THUMB_H = 200, 150
COLS = 4
PAGE_SIZE = 24  # non-ad images per page


def _batch_key(path: Path) -> str:
    """Group ad frames: strip the trailing _{counter} from the stem."""
    return path.stem.rsplit("_", 1)[0]


def _counter(path: Path) -> int:
    """Trailing _{counter} as an int, for numeric (not lexicographic) ordering."""
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def scan_ad(ad_dir: Path) -> dict[str, list[Path]]:
    files = sorted(ad_dir.glob("*.jpg"))
    batches: dict[str, list[Path]] = {}
    for f in files:
        key = _batch_key(f)
        batches.setdefault(key, []).append(f)
    for frames in batches.values():
        frames.sort(key=_counter)
    return batches


def scan_non_ad(non_ad_dir: Path) -> list[Path]:
    return sorted(non_ad_dir.glob("*.jpg"))


# ---------------------------------------------------------------------------
# Thumbnail helpers
# ---------------------------------------------------------------------------

_thumb_cache: dict[Path, ImageTk.PhotoImage] = {}


def _make_thumb(path: Path, selected: bool) -> ImageTk.PhotoImage:
    img = Image.open(path).convert("RGB")
    img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)

    if selected:
        overlay = Image.new("RGBA", img.size, (200, 0, 0, 120))
        base = img.convert("RGBA")
        merged = Image.alpha_composite(base, overlay).convert("RGB")
        draw = ImageDraw.Draw(merged)
        w, h = merged.size
        for t in range(4):
            draw.rectangle([t, t, w - 1 - t, h - 1 - t], outline=(220, 30, 30))
        img = merged

    return ImageTk.PhotoImage(img)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

class ReviewApp(tk.Tk):
    def __init__(self, dataset_dir: Path):
        super().__init__()
        self.title("Dataset Image Reviewer")
        self.geometry("1100x700")
        self.minsize(800, 500)
        self.state("zoomed")  # start maximized (Windows)

        self.dataset_dir = dataset_dir
        self.ad_dir = dataset_dir / "result" / "ad"
        self.non_ad_dir = dataset_dir / "result" / "non_ad"
        self.discard_ad_dir = dataset_dir / "result" / "discarded" / "ad"
        self.discard_non_ad_dir = dataset_dir / "result" / "discarded" / "non_ad"
        self.progress_path = dataset_dir / "result" / ".review_progress.json"

        # State
        self.mode = tk.StringVar(value="ad")
        self.ad_batches: dict[str, list[Path]] = {}
        self.ad_keys: list[str] = []
        self.current_batch_idx: int = 0

        self.non_ad_images: list[Path] = []
        self.current_page: int = 0

        self.selected: set[Path] = set()
        self.undo_stack: list[tuple[list[Path], list[Path]]] = []
        self._thumb_refs: list[ImageTk.PhotoImage] = []  # prevent GC

        self.discarded_ad_count = tk.IntVar(value=0)
        self.discarded_non_ad_count = tk.IntVar(value=0)

        # Saved review progress, applied once after the first directory scan.
        self._progress = self._load_progress()
        self._restored = False

        self._build_ui()
        self._reload_data()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Top bar
        top = tk.Frame(self, bd=1, relief=tk.FLAT, bg="#2b2b2b")
        top.pack(fill=tk.X, side=tk.TOP)

        tk.Label(top, text="Dataset Reviewer", fg="white", bg="#2b2b2b",
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=12, pady=6)

        mode_frame = tk.Frame(top, bg="#2b2b2b")
        mode_frame.pack(side=tk.LEFT, padx=20)
        tk.Radiobutton(mode_frame, text="AD Batches", variable=self.mode,
                       value="ad", command=self._on_mode_change,
                       bg="#2b2b2b", fg="white", selectcolor="#555",
                       activebackground="#2b2b2b", activeforeground="white").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="NON-AD", variable=self.mode,
                       value="non_ad", command=self._on_mode_change,
                       bg="#2b2b2b", fg="white", selectcolor="#555",
                       activebackground="#2b2b2b", activeforeground="white").pack(side=tk.LEFT, padx=10)

        # Main pane
        self.paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=5, bg="#444")
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Left sidebar
        self.sidebar = tk.Frame(self.paned, bg="#1e1e1e", width=300)
        self.paned.add(self.sidebar, minsize=200)

        self.sidebar_label = tk.Label(self.sidebar, text="Batches", fg="#aaa",
                                      bg="#1e1e1e", font=("Segoe UI", 9, "bold"))
        self.sidebar_label.pack(anchor=tk.W, padx=8, pady=(8, 2))

        sb_list_frame = tk.Frame(self.sidebar, bg="#1e1e1e")
        sb_list_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.batch_listbox = tk.Listbox(sb_list_frame, bg="#252525", fg="#ddd",
                                         selectbackground="#3c6faf",
                                         selectforeground="white",
                                         activestyle="none",
                                         font=("Consolas", 8),
                                         relief=tk.FLAT, bd=0)
        self.batch_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_scroll = tk.Scrollbar(sb_list_frame, orient=tk.VERTICAL,
                                  command=self.batch_listbox.yview)
        sb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.batch_listbox.config(yscrollcommand=sb_scroll.set)
        self.batch_listbox.bind("<<ListboxSelect>>", self._on_batch_select)

        # Right panel
        right = tk.Frame(self.paned, bg="#2d2d2d")
        self.paned.add(right, minsize=400)

        # Batch header
        header = tk.Frame(right, bg="#333", pady=4)
        header.pack(fill=tk.X)

        self.header_label = tk.Label(header, text="", fg="white", bg="#333",
                                      font=("Segoe UI", 9, "bold"), anchor=tk.W)
        self.header_label.pack(side=tk.LEFT, padx=10)

        self.counter_label = tk.Label(header, text="", fg="#aaa", bg="#333",
                                       font=("Segoe UI", 9))
        self.counter_label.pack(side=tk.RIGHT, padx=10)

        btn_row = tk.Frame(right, bg="#2d2d2d", pady=3)
        btn_row.pack(fill=tk.X)
        self._btn(btn_row, "Select All  (A)", self.select_all, "#3a3a3a").pack(side=tk.LEFT, padx=6)
        self._btn(btn_row, "Deselect All", self.deselect_all, "#3a3a3a").pack(side=tk.LEFT)

        # Image canvas
        canvas_frame = tk.Frame(right, bg="#2d2d2d")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.canvas = tk.Canvas(canvas_frame, bg="#2d2d2d", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.config(yscrollcommand=v_scroll.set)
        self.canvas.bind("<Configure>", lambda e: self._reflow_grid())
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        self.grid_frame = tk.Frame(self.canvas, bg="#2d2d2d")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.grid_frame, anchor=tk.NW)
        self.grid_frame.bind("<Configure>", lambda e: self.canvas.config(
            scrollregion=self.canvas.bbox("all")))

        # Bottom bar
        bottom = tk.Frame(self, bg="#222", pady=5)
        bottom.pack(fill=tk.X, side=tk.BOTTOM)

        self._btn(bottom, "← Back  (H)", lambda: self.navigate(-1), "#3a3a3a").pack(side=tk.LEFT, padx=6)
        self._btn(bottom, "Skip →  (L)", lambda: self.navigate(1), "#3a3a3a").pack(side=tk.LEFT)
        self._btn(bottom, "Discard Selected & Next  (D)",
                  self.discard_and_next, "#8b2020").pack(side=tk.LEFT, padx=12)
        self._btn(bottom, "Undo Last  (Ctrl+Z)", self.undo_last, "#3a3a3a").pack(side=tk.LEFT)

        self.status_label = tk.Label(bottom, text="", fg="#888", bg="#222",
                                      font=("Segoe UI", 8))
        self.status_label.pack(side=tk.RIGHT, padx=12)

        # Keyboard shortcuts
        self.bind("<Right>", lambda e: self.navigate(1))
        self.bind("<Left>", lambda e: self.navigate(-1))
        self.bind("l", lambda e: self.navigate(1))
        self.bind("h", lambda e: self.navigate(-1))
        self.bind("a", lambda e: self.select_all())
        self.bind("d", lambda e: self.discard_and_next())
        self.bind("<Control-z>", lambda e: self.undo_last())

    def _btn(self, parent, text, cmd, bg="#3a3a3a"):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg="white", relief=tk.FLAT,
                         activebackground="#555", activeforeground="white",
                         padx=8, pady=4, font=("Segoe UI", 8), cursor="hand2")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _reload_data(self):
        self.selected.clear()
        if self.ad_dir.exists():
            self.ad_batches = scan_ad(self.ad_dir)
        else:
            self.ad_batches = {}
        self.ad_keys = list(self.ad_batches.keys())

        if self.non_ad_dir.exists():
            self.non_ad_images = scan_non_ad(self.non_ad_dir)
        else:
            self.non_ad_images = []

        if not self._restored:
            self._apply_saved_progress()
            self._restored = True

        self._populate_sidebar()
        self._show_current()

    # ------------------------------------------------------------------
    # Progress persistence
    # ------------------------------------------------------------------

    def _load_progress(self) -> dict:
        try:
            return json.loads(self.progress_path.read_text())
        except Exception:
            return {}

    def _save_progress(self):
        if self.ad_keys and 0 <= self.current_batch_idx < len(self.ad_keys):
            ad_key = self.ad_keys[self.current_batch_idx]
        else:
            ad_key = None
        data = {
            "mode": self.mode.get(),
            "ad_batch_key": ad_key,
            "non_ad_page": self.current_page,
        }
        try:
            self.progress_path.parent.mkdir(parents=True, exist_ok=True)
            self.progress_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _apply_saved_progress(self):
        """Restore the last reviewed position from disk (called once at startup)."""
        p = self._progress
        if not p:
            return
        mode = p.get("mode")
        if mode in ("ad", "non_ad"):
            self.mode.set(mode)
        # Resolve the AD batch by name so discards shifting indices don't matter.
        key = p.get("ad_batch_key")
        if key in self.ad_keys:
            self.current_batch_idx = self.ad_keys.index(key)
        page = p.get("non_ad_page")
        if isinstance(page, int):
            total_pages = max(1, (len(self.non_ad_images) + PAGE_SIZE - 1) // PAGE_SIZE)
            self.current_page = max(0, min(total_pages - 1, page))

    def _on_close(self):
        self._save_progress()
        self.destroy()

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------

    def _populate_sidebar(self):
        self.batch_listbox.delete(0, tk.END)
        if self.mode.get() == "ad":
            self.sidebar_label.config(text=f"Batches ({len(self.ad_keys)})")
            for key in self.ad_keys:
                count = len(self.ad_batches[key])
                # Show only the cod part to keep it short
                short = key.split("_processed_")[-1] if "_processed_" in key else key[-20:]
                prefix = key.replace(f"_processed_{short}", "")
                display = f"{prefix[-16:]}…{short}  ({count})"
                self.batch_listbox.insert(tk.END, display)
            if self.ad_keys:
                idx = min(self.current_batch_idx, len(self.ad_keys) - 1)
                self.batch_listbox.selection_clear(0, tk.END)
                self.batch_listbox.selection_set(idx)
                self.batch_listbox.see(idx)
        else:
            total_pages = max(1, (len(self.non_ad_images) + PAGE_SIZE - 1) // PAGE_SIZE)
            self.sidebar_label.config(text=f"NON-AD Images ({len(self.non_ad_images)})")
            for p in range(total_pages):
                start = p * PAGE_SIZE + 1
                end = min((p + 1) * PAGE_SIZE, len(self.non_ad_images))
                self.batch_listbox.insert(tk.END, f"Page {p + 1}  ({start}–{end})")
            if total_pages:
                pg = min(self.current_page, total_pages - 1)
                self.batch_listbox.selection_clear(0, tk.END)
                self.batch_listbox.selection_set(pg)
                self.batch_listbox.see(pg)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _on_mode_change(self):
        self.selected.clear()
        self.current_batch_idx = 0
        self.current_page = 0
        self._populate_sidebar()
        self._show_current()
        self._save_progress()

    def _on_batch_select(self, event):
        sel = self.batch_listbox.curselection()
        if not sel:
            return
        self.selected.clear()
        if self.mode.get() == "ad":
            self.current_batch_idx = sel[0]
        else:
            self.current_page = sel[0]
        self._show_current()
        self._save_progress()

    def navigate(self, delta: int):
        self.selected.clear()
        if self.mode.get() == "ad":
            if not self.ad_keys:
                return
            self.current_batch_idx = max(0, min(len(self.ad_keys) - 1,
                                                 self.current_batch_idx + delta))
        else:
            total_pages = max(1, (len(self.non_ad_images) + PAGE_SIZE - 1) // PAGE_SIZE)
            self.current_page = max(0, min(total_pages - 1, self.current_page + delta))
        self._populate_sidebar()
        self._show_current()
        self._save_progress()

    def _show_current(self):
        if self.mode.get() == "ad":
            self._show_ad_batch()
        else:
            self._show_non_ad_page()

    def _show_ad_batch(self):
        if not self.ad_keys:
            self._render_grid([])
            self.header_label.config(text="No AD images found")
            self.counter_label.config(text="")
            return
        idx = self.current_batch_idx
        key = self.ad_keys[idx]
        images = self.ad_batches[key]
        self.header_label.config(text=key)
        self.counter_label.config(text=f"Batch {idx + 1} of {len(self.ad_keys)}")
        self._render_grid(images)

    def _show_non_ad_page(self):
        if not self.non_ad_images:
            self._render_grid([])
            self.header_label.config(text="No NON-AD images found")
            self.counter_label.config(text="")
            return
        total_pages = max(1, (len(self.non_ad_images) + PAGE_SIZE - 1) // PAGE_SIZE)
        pg = self.current_page
        start = pg * PAGE_SIZE
        images = self.non_ad_images[start: start + PAGE_SIZE]
        self.header_label.config(text="NON-AD images")
        self.counter_label.config(text=f"Page {pg + 1} of {total_pages}  "
                                        f"({start + 1}–{start + len(images)} of {len(self.non_ad_images)})")
        self._render_grid(images)

    # ------------------------------------------------------------------
    # Grid rendering
    # ------------------------------------------------------------------

    def _current_cols(self) -> int:
        w = self.canvas.winfo_width()
        cols = max(1, w // (THUMB_W + 12))
        return cols

    def _reflow_grid(self):
        # Re-render when window resizes — only if grid has children
        if self.grid_frame.winfo_children():
            self._show_current()

    def _render_grid(self, paths: list[Path]):
        # Clear previous
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        self._thumb_refs.clear()
        self.canvas.yview_moveto(0)

        self.canvas.update_idletasks()
        cols = self._current_cols()

        for i, path in enumerate(paths):
            row, col = divmod(i, cols)
            selected = path in self.selected
            try:
                thumb = _make_thumb(path, selected)
            except Exception:
                continue
            self._thumb_refs.append(thumb)

            cell = tk.Frame(self.grid_frame, bg="#2d2d2d", padx=4, pady=4)
            cell.grid(row=row, column=col, sticky=tk.NSEW)

            lbl = tk.Label(cell, image=thumb, bg="#2d2d2d", cursor="hand2")
            lbl.pack()
            lbl.bind("<Button-1>", lambda e, p=path: self._toggle(p))

            name_lbl = tk.Label(cell, text=path.name, fg="#999",
                                 bg="#2d2d2d", font=("Consolas", 7), wraplength=THUMB_W)
            name_lbl.pack()

        self._update_status()
        # Expand canvas window to full width
        self.canvas.update_idletasks()
        self.canvas.itemconfig(self.canvas_window,
                                width=self.canvas.winfo_width())

    def _toggle(self, path: Path):
        if path in self.selected:
            self.selected.discard(path)
        else:
            self.selected.add(path)
        self._show_current()

    # ------------------------------------------------------------------
    # Selection actions
    # ------------------------------------------------------------------

    def select_all(self):
        if self.mode.get() == "ad":
            if not self.ad_keys:
                return
            self.selected.update(self.ad_batches[self.ad_keys[self.current_batch_idx]])
        else:
            pg = self.current_page
            self.selected.update(self.non_ad_images[pg * PAGE_SIZE:(pg + 1) * PAGE_SIZE])
        self._show_current()

    def deselect_all(self):
        self.selected.clear()
        self._show_current()

    # ------------------------------------------------------------------
    # Discard
    # ------------------------------------------------------------------

    def discard_and_next(self):
        if not self.selected:
            self.navigate(1)
            return

        mode = self.mode.get()
        dest_dir = self.discard_ad_dir if mode == "ad" else self.discard_non_ad_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        moved_src: list[Path] = []
        moved_dst: list[Path] = []
        for src in list(self.selected):
            dst = dest_dir / src.name
            shutil.move(str(src), str(dst))
            moved_src.append(src)
            moved_dst.append(dst)

        self.undo_stack.append((moved_src, moved_dst))

        if mode == "ad":
            self.discarded_ad_count.set(self.discarded_ad_count.get() + len(moved_src))
        else:
            self.discarded_non_ad_count.set(self.discarded_non_ad_count.get() + len(moved_src))

        self.selected.clear()
        self._reload_data_soft()
        self.navigate(1)

    def _reload_data_soft(self):
        """Rescan directories without resetting navigation position."""
        if self.ad_dir.exists():
            self.ad_batches = scan_ad(self.ad_dir)
        else:
            self.ad_batches = {}
        self.ad_keys = list(self.ad_batches.keys())

        if self.non_ad_dir.exists():
            self.non_ad_images = scan_non_ad(self.non_ad_dir)
        else:
            self.non_ad_images = []

        # Clamp indices
        if self.ad_keys:
            self.current_batch_idx = min(self.current_batch_idx, len(self.ad_keys) - 1)
        else:
            self.current_batch_idx = 0

        total_pages = max(1, (len(self.non_ad_images) + PAGE_SIZE - 1) // PAGE_SIZE)
        self.current_page = min(self.current_page, total_pages - 1)

    # ------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------

    def undo_last(self):
        if not self.undo_stack:
            return
        moved_src, moved_dst = self.undo_stack.pop()
        for src, dst in zip(moved_src, moved_dst):
            if dst.exists():
                shutil.move(str(dst), str(src))

        mode = self.mode.get()
        if mode == "ad":
            self.discarded_ad_count.set(
                max(0, self.discarded_ad_count.get() - len(moved_src)))
        else:
            self.discarded_non_ad_count.set(
                max(0, self.discarded_non_ad_count.get() - len(moved_src)))

        self._reload_data_soft()
        self._populate_sidebar()
        self._show_current()
        self._save_progress()

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _update_status(self):
        sel_count = len(self.selected)
        mode = self.mode.get()
        if mode == "ad":
            current = (self.ad_batches.get(self.ad_keys[self.current_batch_idx], [])
                       if self.ad_keys else [])
            total_str = f"{len(current)} images in batch"
        else:
            pg = self.current_page
            current = self.non_ad_images[pg * PAGE_SIZE:(pg + 1) * PAGE_SIZE]
            total_str = f"{len(current)} images on page"

        discarded = (f"Discarded: {self.discarded_ad_count.get()} ad  |  "
                     f"{self.discarded_non_ad_count.get()} non-ad")
        sel_str = f"  |  {sel_count} selected" if sel_count else ""
        self.status_label.config(text=f"{total_str}{sel_str}   ·   {discarded}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    try:
        cfg = _load_config()
    except Exception as e:
        messagebox.showerror("Config error", str(e))
        sys.exit(1)

    dataset_dir = Path(cfg["path"]["dataset"]).resolve()

    if not dataset_dir.exists():
        messagebox.showerror("Error", f"Dataset directory not found:\n{dataset_dir}")
        sys.exit(1)

    result_dir = dataset_dir / "result"
    if not result_dir.exists():
        messagebox.showwarning("Warning", f"No result/ directory found at:\n{result_dir}\n\n"
                                           "Run the pipeline first to generate images.")

    app = ReviewApp(dataset_dir)
    app.mainloop()


if __name__ == "__main__":
    main()
