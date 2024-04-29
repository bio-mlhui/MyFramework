# import objaverse.xl as oxl


# founded = []
# modified = []
# missing_object = []

# def handle_missing_object(file_identifier, sha256, metadata):
#     missing_object.append((file_identifier, sha256, metadata))

# oxl.download_objects(
#     # Base parameters:
#     objects=oxl.get_annotations(download_dir='~/datasets/oxl'),
#     download_dir='~/datasets/oxl',
#     handle_missing_object = handle_missing_object,
# )

# print(missing_object)
# import json
# with open('./missing.json', 'w') as f:
#     json.dump(missing_object, f)

# 
import json
import objaverse
uids = objaverse.load_uids()
obj_paths = objaverse.load_objects(
    uids=uids,
    download_processes=64
)
with open('./downloaded.json', 'w') as f:
    json.dump(obj_paths, f)
