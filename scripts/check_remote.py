import subprocess, json, time

def remote_cmd(cmd):
    r = subprocess.run(['ssh', '-p', '2222', '-o', 'LogLevel=ERROR', '-o', 'ConnectTimeout=10', 'administrator@100.115.18.62', cmd], capture_output=True, text=True, timeout=60, errors='replace')
    return r.stdout.strip(), r.stderr.strip()

# Wait for eval to complete (5 minutes max)
for i in range(10):
    time.sleep(60)
    out, _ = remote_cmd('wsl -- cat /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/620_spatial_bridge/620_intrinsic_v2/full_eval/curve_summary.json 2>/dev/null')
    if out:
        try:
            data = json.loads(out)
            row_count = data.get("row_count", 0)
            print("Eval check %d: rows=%d" % (i+1, row_count))
            if row_count >= 6:
                break
        except:
            pass
    else:
        print("Eval check %d: no data yet" % (i+1))

# Final results
out, _ = remote_cmd('wsl -- cat /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/620_spatial_bridge/620_intrinsic_v2/full_eval/curve_summary.json 2>/dev/null')
if out:
    try:
        data = json.loads(out)
        print("\n=== H6 INTRINSIC RESULTS ===")
        bt = data["best_transfer"]
        print("Best clip_style:", round(bt["transfer_clip_style"], 4))
        print("Best lpips:", round(bt["transfer_content_lpips"], 4))
        print("\nAll epochs:")
        for row in data["rows"]:
            print("  Epoch %d: clip_style=%.4f, lpips=%.4f" % (row["epoch_int"], row["transfer_clip_style"], row["transfer_content_lpips"]))
    except Exception as e:
        print("Parse error:", e)
else:
    print("No eval results")
