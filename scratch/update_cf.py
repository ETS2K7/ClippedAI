import json

with open("dist-config.json", "r") as f:
    data = json.load(f)

config = data["DistributionConfig"]
etag = data["ETag"]

# Update origin
config["Origins"]["Items"][0]["DomainName"] = "clippedai-7137.s3.us-east-1.amazonaws.com"
config["Origins"]["Items"][0]["Id"] = "S3-clippedai-7137"

# Also update DefaultCacheBehavior TargetOriginId
if config["DefaultCacheBehavior"]["TargetOriginId"] == "S3-clippedai-ap-south-1":
    config["DefaultCacheBehavior"]["TargetOriginId"] = "S3-clippedai-7137"

with open("modified-config.json", "w") as f:
    json.dump(config, f)

with open("etag.txt", "w") as f:
    f.write(etag)
