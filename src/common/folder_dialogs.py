"""Native folder pickers. Windows supports selecting multiple directories."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def ask_directories(
    *,
    parent: Any = None,
    title: str = "選擇資料夾",
    initialdir: str | Path | None = None,
) -> list[Path]:
    """Return selected folders, or an empty list if the user cancels.

    On Windows this uses the Explorer folder dialog with multi-select
    (Ctrl / Shift). Other platforms fall back to a single-folder picker.
    """
    start = str(initialdir) if initialdir else None
    if os.name == "nt":
        try:
            chosen = _ask_directories_windows(
                parent=parent,
                title=title,
                initialdir=start,
            )
        except Exception:
            chosen = None
        if chosen is not None:
            return chosen
    from tkinter import filedialog

    folder = filedialog.askdirectory(
        parent=parent,
        title=title,
        initialdir=start or "",
    )
    if not folder:
        return []
    return [Path(folder)]


def _owner_hwnd(parent: Any) -> int:
    if parent is None:
        return 0
    try:
        hwnd = int(parent.winfo_toplevel().winfo_id())
    except Exception:
        return 0
    try:
        import ctypes

        root = int(ctypes.windll.user32.GetAncestor(hwnd, 2))  # GA_ROOT
        return root or hwnd
    except Exception:
        return hwnd


def _ask_directories_windows(
    *,
    parent: Any,
    title: str,
    initialdir: str | None,
) -> list[Path] | None:
    """IFileOpenDialog with folder + multi-select. None means the dialog could not be created."""
    import ctypes
    from ctypes import wintypes

    clsctx_inproc_server = 1
    fos_nochangedir = 0x00000008
    fos_pickfolders = 0x00000020
    fos_forcefilesystem = 0x00000040
    fos_allowmultiselect = 0x00000200
    sigdn_filesyspath = 0x80058000

    ole32 = ctypes.windll.ole32
    shell32 = ctypes.windll.shell32

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_ulong),
            ("Data2", ctypes.c_ushort),
            ("Data3", ctypes.c_ushort),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    def guid_from_string(value: str) -> GUID:
        parsed = GUID()
        hr = ole32.CLSIDFromString(wintypes.LPCWSTR(value), ctypes.byref(parsed))
        if hr != 0:
            raise OSError(f"CLSIDFromString failed: 0x{hr & 0xFFFFFFFF:08X}")
        return parsed

    def vtable(ptr: ctypes.c_void_p, size: int) -> Any:
        return ctypes.cast(
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))[0],
            ctypes.POINTER(ctypes.c_void_p * size),
        ).contents

    def com_release(ptr: ctypes.c_void_p | None) -> None:
        if not ptr:
            return
        release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable(ptr, 3)[2])
        release(ptr)

    def display_name(item: ctypes.c_void_p) -> str | None:
        get_name = ctypes.WINFUNCTYPE(
            ctypes.HRESULT,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.POINTER(wintypes.LPWSTR),
        )(vtable(item, 6)[5])
        path_ptr = wintypes.LPWSTR()
        hr = get_name(item, sigdn_filesyspath, ctypes.byref(path_ptr))
        if hr != 0 or not path_ptr:
            return None
        try:
            return path_ptr.value
        finally:
            ole32.CoTaskMemFree(path_ptr)

    ole32.CoInitialize(None)

    clsid = guid_from_string("{DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7}")
    iid_dialog = guid_from_string("{D57C7288-D4AD-4768-BE02-9D969532D960}")
    iid_item = guid_from_string("{43826D1E-E718-42EE-BC55-A1E261C37BFE}")

    dialog = ctypes.c_void_p()
    hr = ole32.CoCreateInstance(
        ctypes.byref(clsid),
        None,
        clsctx_inproc_server,
        ctypes.byref(iid_dialog),
        ctypes.byref(dialog),
    )
    if hr != 0 or not dialog:
        return None

    # IUnknown 0-2, IModalWindow 3, IFileDialog 4-26, IFileOpenDialog 27-28
    dlg_vtbl = vtable(dialog, 29)
    try:
        get_options = ctypes.WINFUNCTYPE(
            ctypes.HRESULT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)
        )(dlg_vtbl[10])
        set_options = ctypes.WINFUNCTYPE(
            ctypes.HRESULT, ctypes.c_void_p, ctypes.c_uint
        )(dlg_vtbl[9])
        opts = ctypes.c_uint()
        get_options(dialog, ctypes.byref(opts))
        set_options(
            dialog,
            opts.value
            | fos_pickfolders
            | fos_allowmultiselect
            | fos_forcefilesystem
            | fos_nochangedir,
        )

        if title:
            set_title = ctypes.WINFUNCTYPE(
                ctypes.HRESULT, ctypes.c_void_p, wintypes.LPCWSTR
            )(dlg_vtbl[17])
            set_title(dialog, title)

        if initialdir and os.path.isdir(initialdir):
            folder_item = ctypes.c_void_p()
            shell32.SHCreateItemFromParsingName.restype = ctypes.c_long
            hr = shell32.SHCreateItemFromParsingName(
                wintypes.LPCWSTR(initialdir),
                None,
                ctypes.byref(iid_item),
                ctypes.byref(folder_item),
            )
            if hr == 0 and folder_item:
                set_folder = ctypes.WINFUNCTYPE(
                    ctypes.HRESULT, ctypes.c_void_p, ctypes.c_void_p
                )(dlg_vtbl[12])
                set_folder(dialog, folder_item)
                com_release(folder_item)

        show = ctypes.WINFUNCTYPE(
            ctypes.HRESULT, ctypes.c_void_p, wintypes.HWND
        )(dlg_vtbl[3])
        hr = show(dialog, _owner_hwnd(parent))
        # After Show() the user already saw a dialog; never fall back to a second picker.
        try:
            if hr != 0:
                return []

            get_results = ctypes.WINFUNCTYPE(
                ctypes.HRESULT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            )(dlg_vtbl[27])
            items = ctypes.c_void_p()
            hr = get_results(dialog, ctypes.byref(items))
            if hr != 0 or not items:
                return []

            try:
                # IShellItemArray: GetCount=7, GetItemAt=8
                arr_vtbl = vtable(items, 9)
                get_count = ctypes.WINFUNCTYPE(
                    ctypes.HRESULT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)
                )(arr_vtbl[7])
                get_item_at = ctypes.WINFUNCTYPE(
                    ctypes.HRESULT,
                    ctypes.c_void_p,
                    ctypes.c_uint,
                    ctypes.POINTER(ctypes.c_void_p),
                )(arr_vtbl[8])
                count = ctypes.c_uint()
                get_count(items, ctypes.byref(count))
                paths: list[Path] = []
                for index in range(count.value):
                    item = ctypes.c_void_p()
                    item_hr = get_item_at(items, index, ctypes.byref(item))
                    if item_hr != 0 or not item:
                        continue
                    try:
                        text = display_name(item)
                        if text:
                            paths.append(Path(text))
                    finally:
                        com_release(item)
                return paths
            finally:
                com_release(items)
        except Exception:
            return []
    finally:
        com_release(dialog)
