@echo off
setlocal
set SRC=g:\GitHub\Latent_Style\eval_cache\manual_clip\openai-clip-vit-base-patch32
set DST=administrator@100.115.18.62:"I:/Github/Latent_Style/eval_cache/hf/models--openai--clip-vit-base-patch32/snapshots/c237dc49a33fc61debc9276459120b7eac67e7ef/"

echo === Uploading 7 processor files ===
scp -P 2222 -o LogLevel=ERROR "%SRC%\config.json" %DST%
scp -P 2222 -o LogLevel=ERROR "%SRC%\merges.txt" %DST%
scp -P 2222 -o LogLevel=ERROR "%SRC%\preprocessor_config.json" %DST%
scp -P 2222 -o LogLevel=ERROR "%SRC%\special_tokens_map.json" %DST%
scp -P 2222 -o LogLevel=ERROR "%SRC%\tokenizer.json" %DST%
scp -P 2222 -o LogLevel=ERROR "%SRC%\tokenizer_config.json" %DST%
scp -P 2222 -o LogLevel=ERROR "%SRC%\vocab.json" %DST%
echo === DONE ===
endlocal
