import pathlib, os, ctypes

# Check each directory level
dirs = [
    pathlib.Path(r"I:\Github\Latent_Style\exp_samam\training\samam_distinct5_512_scratch_7k_250eval_remote"),
    pathlib.Path(r"I:\Github\Latent_Style\exp_samam\training\samam_distinct5_512_scratch_7k_250eval_remote\c"),
    pathlib.Path(r"I:\Github\Latent_Style\exp_samam\training\samam_distinct5_512_scratch_7k_250eval_remote\c\step_000250"),
    pathlib.Path(r"I:\Github\Latent_Style\exp_samam\training\samam_distinct5_512_scratch_7k_250eval_remote\c\step_000250\images"),
]

for d in dirs:
    print(f"\n{d.name}: len={len(str(d))}")
    attrs = ctypes.windll.kernel32.GetFileAttributesW(str(d))
    print(f"  attrs: {attrs} (reparse: {bool(attrs & 0x400)}, dir: {bool(attrs & 0x10)})")
    try:
        s = d.stat()
        print(f"  stat: OK, uid={s.st_uid}, gid={s.st_gid}")
    except Exception as e:
        print(f"  stat: {e}")
    try:
        items = list(d.iterdir())[:3]
        print(f"  iterdir: OK, {len(items)} items")
    except Exception as e:
        print(f"  iterdir: {e}")

# Compare with working dir
print("\n=== Working dir comparison ===")
wd = pathlib.Path(r"I:\Github\Latent_Style\exp_samam\training\samam_distinct5_512_scratch_7k_250eval_remote\curve_eval_30src\last\images")
attrs = ctypes.windll.kernel32.GetFileAttributesW(str(wd))
print(f"  attrs: {attrs}")
s = wd.stat()
print(f"  stat: uid={s.st_uid}, gid={s.st_gid}")

# Try using \\?\ prefix with CreateFileW on the non-working file
p2 = pathlib.Path(r"I:\Github\Latent_Style\exp_samam\training\samam_distinct5_512_scratch_7k_250eval_remote\c\step_000250\images\Early_Renaissance__Early_Renaissance__andrea-mantegna_adoration-of-the-magi-central-panel-from-the-altarpiece__to__Early_Renaissance.png")
print(f"\n=== Trying \\\\?\\ on non-working file ===")
kernel32 = ctypes.windll.kernel32
long_path = "\\\\?\\" + os.path.normpath(str(p2))
print(f"  long_path len: {len(long_path)}")
handle = kernel32.CreateFileW(
    long_path, 0x80000000, 0, None, 3, 0x80, None
)
if handle == -1:
    err = kernel32.GetLastError()
    print(f"  CreateFileW failed: {err}")
else:
    print(f"  CreateFileW OK: {handle}")
    kernel32.CloseHandle(handle)