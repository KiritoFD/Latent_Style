from modelscope.hub.api import HubApi
api = HubApi()
print("Listing files in 'google/t5-v1_1-large' repository:")
try:
    files = api.get_model_files(model_id='google/t5-v1_1-large')
    for f in files:
        print(f" - {f.get('Name')} ({f.get('Size')} bytes)")
except Exception as e:
    print("Error:", e)
