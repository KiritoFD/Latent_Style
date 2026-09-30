import pathlib, os, ctypes
from ctypes import wintypes

p = pathlib.Path(r"I:\Github\Latent_Style\exp_samam\training\samam_distinct5_512_scratch_7k_250eval_remote\curve_eval_hf_750_batched\step_000250\images")
imgs = sorted(p.glob("*.png"))
print(f"num images: {len(imgs)}")
print(f"first: {imgs[0].name}")
print(f"path len: {len(str(imgs[0]))}")

kernel32 = ctypes.windll.kernel32
GENERIC_READ = 0x80000000
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x80
INVALID_HANDLE_VALUE = -1  # ctypes.c_void_p(-1).value is not -1 on 64-bit

# Try multiple path formats
norm_path = os.path.normpath(str(imgs[0]))
formats = [
    ("\\\\?\\", "\\\\?\\" + norm_path),
    ("\\\\.\\", "\\\\.\\" + norm_path),
    ("raw", norm_path),
]

for label, lp in formats:
    print(f"\nTrying {label}: {lp[:80]}...")
    handle = kernel32.CreateFileW(
        ctypes.c_wchar_p(lp), GENERIC_READ, 0, None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None
    )
    if handle == INVALID_HANDLE_VALUE:
        err = kernel32.GetLastError()
        print(f"  CreateFileW failed: error {err}")
    else:
        print(f"  CreateFileW OK, handle={handle}")
        buf = ctypes.create_string_buffer(4096)
        bytes_read = wintypes.DWORD()
        if kernel32.ReadFile(handle, buf, 4096, ctypes.byref(bytes_read), None):
            print(f"  ReadFile OK, {bytes_read.value} bytes")
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(buf.raw[:bytes_read.value])).convert("RGB")
            print(f"  PIL OK: {img.size}")
        kernel32.CloseHandle(handle)
        break